# src/loadbalancer/proxy.py
import asyncio
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import cast

import aiohttp
from aiohttp import web
from multidict import CIMultiDict

from loadbalancer.balancer import pick
from loadbalancer.config import Config, load_config
from loadbalancer.telemetry import Telemetry

logger = logging.getLogger("loadbalancer.proxy")

State = dict[str, dict[str, int | float | bool | None] | None]

# Transient status codes that are safe to retry (rule 1)
RETRYABLE_STATUS = {500, 502, 503, 504}

# Hop-by-hop headers describe the upstream connection only and must not be
# relayed to a different connection (RFC 9110 7.6.1).
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
# aiohttp has already decoded the body, so relaying the upstream encoding or
# length would describe bytes that are no longer on the wire.
ENTITY_HEADERS = frozenset({"content-encoding", "content-length"})

# Absolute ceiling on a buffered request body, independent of `max_body_size`,
# so a single pathological request cannot exhaust memory. The configured limit
# (which may change at runtime) is enforced in Proxy._read_body.
HARD_BODY_CEILING = 1 << 30  # 1 GiB

# Operational endpoints stay reachable without a token: a liveness or
# readiness probe has to work while the balancer is being restarted, and none
# of them reveals backend data.
UNPROTECTED = frozenset({"/", "/health", "/ready", "/metrics"})

# The balancer's own credential. It is stripped before forwarding, so it never
# reaches a backend. `Authorization` is deliberately *not* listed: that is the
# backend's credential slot (the OpenAI key) and is relayed unchanged.
LB_TOKEN_HEADER = "X-Auth-Token"


def _presented_token(request: web.Request) -> str | None:
    """The token the client presents, from either accepted header."""
    dedicated = request.headers.get(LB_TOKEN_HEADER)
    if dedicated:
        return dedicated.strip()
    authorization = request.headers.get("Authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None


@web.middleware
async def require_token(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    """Reject unauthenticated traffic when `auth_token` is configured.

    The comparison is constant-time so a wrong token cannot be recovered one
    character at a time from response timing.
    """
    config = request.app[CONFIG_KEY]
    token = config.auth_token
    if not token or request.path in UNPROTECTED:
        return await handler(request)
    presented = _presented_token(request)
    if presented is None or not hmac.compare_digest(presented, token):
        logger.warning("rejected unauthenticated request to %s", request.path)
        return web.json_response(
            {"error": {"type": "unauthorized", "message": "missing or invalid token"}},
            status=401,
        )
    return await handler(request)


def _forwardable_headers(headers: Mapping[str, str]) -> CIMultiDict[str]:
    """Response headers that may be relayed to the client unchanged.

    A `CIMultiDict` preserves repeated header names (several `Set-Cookie`, for
    example). A plain dict keyed by name keeps only the last value, so a
    client would silently receive a partial set.
    """
    return CIMultiDict(
        (key, value)
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP and key.lower() not in ENTITY_HEADERS
    )


def _is_stream(resp: aiohttp.ClientResponse, request: web.Request, wants_stream: bool) -> bool:
    """Whether the response has to be relayed chunk by chunk.

    The upstream content type is authoritative; `stream` in the request body or
    query is honoured for backends that answer before setting it.
    """
    return (
        wants_stream
        or request.query.get("stream") == "true"
        or "text/event-stream" in resp.headers.get("content-type", "")
    )


def _body_fields(body: bytes) -> tuple[str | None, bool]:
    """(model id, stream flag) from a JSON request body.

    GETs, non-JSON bodies and bodies without a string `model` field yield
    (None, False) for the model (such requests are not model-routed). Parsed
    exactly once per request, since a body can be up to `max_body_size` bytes.
    """
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None, False
    if not isinstance(data, dict):
        return None, False
    model = data.get("model")
    name = model if isinstance(model, str) and model else None
    return name, data.get("stream") is True


def preferred_candidates(
    states: State, tried: list[str]
) -> dict[str, dict[str, int | float | bool | None]]:
    """Healthy candidates for a pick.

    Prefers instances that have not been tried yet AND are awake AND have no
    recent preemptions; falls back to any healthy instance if no preferred
    one exists (a sleeping or preempting fleet never renders all instances
    unavailable).
    """
    healthy = {u: s for u, s in states.items() if s is not None}
    fresh = {u: s for u, s in healthy.items() if u not in tried}
    base = fresh if fresh else healthy
    preferred = {
        u: s
        for u, s in base.items()
        if not s.get("sleeping", False) and s.get("preemptions", 0) == 0
    }
    return preferred if preferred else base


class Proxy:
    def __init__(
        self,
        config: Config,
        states: State,
        model_map: dict[str, list[str]] | None = None,
    ) -> None:
        self.config = config
        self.states = states  # url -> metrics dict or None
        # model id -> endpoints hosting that model (queried from /v1/models at startup)
        self.model_map = model_map if model_map is not None else {}
        # locally in-flight requests per instance (see balancer.pick); the
        # upstream /metrics gauges lag by up to one poll interval
        self.inflight: dict[str, int] = {}
        self.telemetry = Telemetry()
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """The shared upstream session, created on first use.

        One session -- and therefore one connector and one connection pool --
        serves the whole app. A session per request would drop keep-alive and
        pay a TCP plus TLS handshake for every single call. `close()` releases
        it when the app shuts down.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the shared session (idempotent)."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _pool_for(self, model: str | None) -> State:
        """Candidate pool for a request.

        If the request names a model the balancer knows, only the endpoints
        hosting that model are candidates. Unknown models (and requests without
        a model) fall back to all instances.
        """
        if model is not None and model in self.model_map:
            urls = set(self.model_map[model])
            return {u: s for u, s in self.states.items() if u in urls}
        return self.states

    def _is_overloaded(self, url: str) -> bool:
        """Rule 4: skip retry when the target instance is overloaded.

        Overloaded = waiting queue at/above the threshold OR the KV-cache
        usage at/above the KV threshold (a leading saturation signal). Absent
        KV metrics never trigger the shortcut.
        """
        st = self.states.get(url)
        if st is None:
            return False
        waiting = st.get("waiting")
        if waiting is not None and float(waiting) >= self.config.overload_threshold:
            return True
        kv = st.get("kv_cache_usage_perc")
        return kv is not None and float(kv) >= self.config.kv_cache_overload_threshold

    def _upstream_timeout(self, stream: bool) -> aiohttp.ClientTimeout:
        """Streamed completions may outlive `timeout` seconds; use an idle
        timeout (reset per chunk) for them, a total timeout otherwise."""
        if stream:
            return aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.config.timeout)
        return aiohttp.ClientTimeout(total=self.config.timeout)

    async def _read_body(self, request: web.Request) -> bytes:
        """The request body, rejecting anything over the *current* limit.

        The limit is read per request so a hot reload takes effect without a
        restart. `client_max_size` is only a hard memory backstop (a config
        value may be raised above it), and Content-Length is client-supplied,
        so the bytes are counted as they arrive rather than trusted.
        """
        limit = self.config.max_body_size
        declared = request.content_length
        if declared is not None and declared > limit:
            raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=declared)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await request.content.readany()
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > limit:
                raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=total)
            chunks.append(chunk)

    async def _forward(self, request: web.Request, path: str) -> web.StreamResponse:
        body = await self._read_body(request)
        model, wants_stream = _body_fields(body)
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", LB_TOKEN_HEADER.lower())
        }
        timeout = self._upstream_timeout(wants_stream)
        tried: list[str] = []
        last_error: str | None = None
        self.telemetry.requests_total += 1

        for attempt in range(self.config.max_retries + 1):
            # Rule 3: on retry, prefer an instance that has not been tried
            # yet; fall back to any healthy one (incl. the same instance)
            # if no fresh one exists. Sleeping and recently-preempting
            # instances are deprioritized unless nothing else is left.
            candidates = preferred_candidates(self._pool_for(model), tried)
            target = pick(
                cast(Mapping[str, dict[str, int | float] | None], candidates),
                self.config.queue_time_weight,
                inflight=self.inflight,
            )
            if target is None:
                if model is not None and model in self.model_map:
                    return web.Response(status=503, text=f"no healthy instance for model {model}")
                return web.Response(status=503, text="no healthy vllm instance")

            # Rule 4: don't retry into an overloaded instance
            if attempt > 0 and self._is_overloaded(target):
                tried.append(target)
                last_error = f"overloaded ({target})"
                await asyncio.sleep(self.config.retry_backoff)
                continue

            url = f"{target.rstrip('/')}{path}"
            self.inflight[target] = self.inflight.get(target, 0) + 1
            # Once the response is committed the client has already seen (part
            # of) it: retrying would hand it a second, different answer on top
            # of the first and pay for the inference twice.
            committed = False
            try:
                async with self.session.request(
                    request.method,
                    url,
                    data=body,
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    if resp.status in RETRYABLE_STATUS and attempt < self.config.max_retries:
                        # Rule 1: transient error, retry. Drain the body so
                        # the connection can return to the pool.
                        await resp.read()
                        tried.append(target)
                        self.telemetry.retries_total += 1
                        last_error = f"status {resp.status} ({target})"
                        await asyncio.sleep(self.config.retry_backoff)
                        continue
                    forwarded = _forwardable_headers(resp.headers)
                    if not _is_stream(resp, request, wants_stream):
                        # Nothing is on the wire yet, so a read error is still
                        # a transient error and stays inside the retry path.
                        data = await resp.read()
                        committed = True
                        return web.Response(status=resp.status, body=data, headers=forwarded)
                    committed = True
                    return await self._pump(resp, request, forwarded)
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                if committed:
                    logger.warning(
                        "upstream failed after the response was committed (%s): %s: %s",
                        target,
                        type(exc).__name__,
                        exc,
                    )
                    raise
                # Rule 1: connection error / timeout is transient, retry
                if attempt < self.config.max_retries:
                    tried.append(target)
                    self.telemetry.retries_total += 1
                    last_error = f"{type(exc).__name__} ({target})"
                    await asyncio.sleep(self.config.retry_backoff)
                    continue
                self.telemetry.upstream_errors_total += 1
                logger.warning("upstream error: %s: %s (%s)", type(exc).__name__, exc, target)
                return web.Response(
                    status=504, text=f"upstream error: {type(exc).__name__} ({target})"
                )
            finally:
                # the in-flight slot is released once the response is fully
                # delivered (incl. streamed bodies) or the attempt failed
                remaining = self.inflight[target] - 1
                if remaining:
                    self.inflight[target] = remaining
                else:
                    del self.inflight[target]

        return web.Response(status=504, text=f"upstream error: {last_error}")

    async def _pump(
        self,
        resp: aiohttp.ClientResponse,
        request: web.Request,
        headers: CIMultiDict[str],
    ) -> web.StreamResponse:
        """Relay a response body chunk by chunk (SSE).

        A stream that breaks mid-body is never retried (the client already
        holds part of this answer, and a second attempt would splice a
        different answer onto a half-written body and pay for the inference
        twice). The break is instead made explicit: SSE clients get a terminal
        `error` event, and the body is never ended cleanly, so a raw client
        cannot mistake a truncated answer for a finished one.
        """
        response = web.StreamResponse(status=resp.status, headers=headers)
        await response.prepare(request)
        is_sse = "text/event-stream" in headers.get("content-type", "")
        try:
            async for chunk in resp.content.iter_any():
                await response.write(chunk)
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            self.telemetry.stream_errors_total += 1
            logger.warning("stream broke mid-response: %s: %s", type(exc).__name__, exc)
            if is_sse:
                # Best effort: the client may already be gone, and losing the
                # event then costs nothing because the body stays incomplete.
                with contextlib.suppress(aiohttp.ClientError, OSError, RuntimeError):
                    await response.write(
                        b"event: error\ndata: "
                        + json.dumps(
                            {
                                "error": {
                                    "type": "upstream_stream_error",
                                    "message": f"upstream stream broke mid-response: {exc!s}",
                                }
                            }
                        ).encode()
                        + b"\n\n"
                    )
            raise
        await response.write_eof()
        return response


PROXY_KEY: web.AppKey[Proxy] = web.AppKey("proxy", Proxy)
CONFIG_KEY: web.AppKey[Config] = web.AppKey("config", Config)


async def _fetch_models(
    session: aiohttp.ClientSession, url: str
) -> tuple[list[dict[str, object]], str | None]:
    """(models, error) from one backend. Never raises: a failing backend is
    skipped so the aggregate still serves what the healthy ones offer."""
    try:
        async with session.get(
            f"{url.rstrip('/')}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status != 200:
                return [], f"{url}: status {resp.status}"
            data = await resp.json()
    except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
        return [], f"{url}: {type(exc).__name__}"
    items = data.get("data", []) if isinstance(data, dict) else []
    return [item for item in items if isinstance(item, dict)], None


def build_app(
    config: Config | None = None,
    states: State | None = None,
    model_map: dict[str, list[str]] | None = None,
) -> web.Application:
    if config is None:
        config = load_config()
    if states is None:
        states = {url: None for url in config.vllm_urls}
    p = Proxy(config, states, model_map)
    # Only a memory backstop: the real, hot-reloadable limit is enforced per
    # request in Proxy._read_body, so a config value may be raised above this.
    app = web.Application(client_max_size=HARD_BODY_CEILING, middlewares=[require_token])
    app[PROXY_KEY] = p
    app[CONFIG_KEY] = config
    if not config.auth_token:
        logger.warning(
            "AUTH_TOKEN is not set: every client that can reach this port may use the backends"
        )

    async def upstream_session(_app: web.Application) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await p.close()

    app.cleanup_ctx.append(upstream_session)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def ready(_request: web.Request) -> web.Response:
        """Readiness: usable only while at least one backend reports metrics.

        Separate from /health on purpose. /health answers "is this process
        alive", which is what a liveness probe needs and stays true during a
        backend outage; a readiness probe has to fail so the balancer is taken
        out of rotation instead of accepting traffic it cannot serve.
        """
        healthy = sum(1 for state in states.values() if state is not None)
        status = 200 if healthy else 503
        return web.json_response(
            {"status": "ready" if healthy else "not ready", "healthy_backends": healthy},
            status=status,
        )

    async def metrics(_request: web.Request) -> web.Response:
        # The version parameter is part of the Prometheus text exposition
        # contract; `web.Response(content_type=...)` cannot express it.
        return web.Response(
            body=p.telemetry.render(states, p.inflight).encode(),
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
        )

    async def root(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "api": "vLLM OpenAI-compatible API (proxied to the least-loaded instance)",
                "endpoints": [
                    "GET /health",
                    "GET /ready",
                    "GET /v1/models",
                    "POST /v1/chat/completions",
                    "POST /v1/completions",
                    "POST /v1/responses",
                    "POST /v1/responses/{id}/cancel",
                    "GET /metrics (the balancer's own counters)",
                ],
            }
        )

    async def list_models(_request: web.Request) -> web.Response:
        # Aggregiert /v1/models live von allen Backends (einzige Quelle der
        # Wahrheit); ausgefallene Instanzen werden übersprungen. Die Backends
        # werden parallel abgefragt, damit ein langsames sie nicht die
        # Antwort verzögert.
        results = await asyncio.gather(*(_fetch_models(p.session, url) for url in config.vllm_urls))
        entries: list[dict[str, object]] = []
        seen: set[str] = set()
        last_error: str | None = None
        for models, error in results:
            if error is not None:
                last_error = error
                continue
            for item in models:
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id not in seen:
                    seen.add(model_id)
                    entries.append(item)
        if not entries:
            return web.Response(status=503, text=f"no models available ({last_error})")
        return web.json_response({"object": "list", "data": entries})

    async def catchall(request: web.Request) -> web.StreamResponse:
        # path_qs, not path: the query string is part of the request target and
        # backends take parameters through it.
        return await p._forward(request, request.path_qs)

    app.router.add_get("/", root)
    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/v1/models", list_models)
    # Generischer Catch-all: leitet jeden Pfad (chat/completions, responses, tool calling) weiter
    app.router.add_route("*", "/{path:.*}", catchall)
    return app
