# src/loadbalancer/proxy.py
import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import cast

import aiohttp
from aiohttp import web

from loadbalancer.balancer import pick
from loadbalancer.config import Config, load_config, upstream_ssl_context

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


def _forwardable_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Response headers that may be relayed to the client unchanged."""
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP and key.lower() not in ENTITY_HEADERS
    }


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
        self.ssl_ctx = upstream_ssl_context()
        # locally in-flight requests per instance (see balancer.pick); the
        # upstream /metrics gauges lag by up to one poll interval
        self.inflight: dict[str, int] = {}
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
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_ctx))
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

    async def _forward(self, request: web.Request, path: str) -> web.StreamResponse:
        body = await request.read()
        model, wants_stream = _body_fields(body)
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        timeout = self._upstream_timeout(wants_stream)
        tried: list[str] = []
        last_error: str | None = None

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
                    last_error = f"{type(exc).__name__} ({target})"
                    await asyncio.sleep(self.config.retry_backoff)
                    continue
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
        headers: dict[str, str],
    ) -> web.StreamResponse:
        """Relay a response body chunk by chunk (SSE); never retried afterwards."""
        response = web.StreamResponse(status=resp.status, headers=headers)
        await response.prepare(request)
        async for chunk in resp.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        return response


PROXY_KEY: web.AppKey[Proxy] = web.AppKey("proxy", Proxy)


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
    app = web.Application(client_max_size=config.max_body_size)
    app[PROXY_KEY] = p

    async def upstream_session(_app: web.Application) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await p.close()

    app.cleanup_ctx.append(upstream_session)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def root(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "api": "vLLM OpenAI-compatible API (proxied to the least-loaded instance)",
                "endpoints": [
                    "GET /health",
                    "GET /v1/models",
                    "POST /v1/chat/completions",
                    "POST /v1/completions",
                    "POST /v1/responses",
                    "POST /v1/responses/{id}/cancel",
                    "GET /metrics",
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
    app.router.add_get("/v1/models", list_models)
    # Generischer Catch-all: leitet jeden Pfad (chat/completions, responses, tool calling) weiter
    app.router.add_route("*", "/{path:.*}", catchall)
    return app
