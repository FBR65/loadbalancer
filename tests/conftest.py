from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

import pytest_asyncio
from aiohttp import web
from aiohttp.client import ClientResponse
from aiohttp.test_utils import TestServer

from loadbalancer.config import Config
from loadbalancer.proxy import build_app

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

_IDLE: dict[str, Any] = {
    "running": 0.0,
    "waiting": 0.0,
    "queue_time_sum": 0.0,
    "kv_cache_usage_perc": 0.1,
    "preemptions": 0.0,
    "sleeping": False,
}


def state(**overrides: Any) -> dict[str, Any]:
    return {**_IDLE, **overrides}


class Fleet:
    """Mock vLLM backends and a load balancer, all on real sockets."""

    def __init__(self) -> None:
        self._servers: list[TestServer] = []
        self._urls: dict[str, str] = {}
        self.hits: dict[str, list[str]] = {}
        self.lb: TestServer | None = None

    async def backend(self, name: str, handler: Handler) -> str:
        hits = self.hits.setdefault(name, [])

        async def record(request: web.Request) -> web.StreamResponse:
            hits.append(request.rel_url.raw_path_qs)
            return await handler(request)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", record)
        server = TestServer(app)
        await server.start_server()
        self._servers.append(server)
        self._urls[name] = f"http://127.0.0.1:{server.port}"
        return self._urls[name]

    def url(self, name: str) -> str:
        return self._urls[name]

    def states(self, **per_backend: Mapping[str, Any]) -> dict[str, Any]:
        """States keyed by backend name, e.g. `states(a=state(queue_time_sum=0))`."""
        out: dict[str, Any] = {}
        for name, overrides in per_backend.items():
            out[self.url(name)] = state(**overrides)
        return out

    async def start_lb(
        self,
        names: list[str],
        *,
        config: Config | None = None,
        states: dict[str, Any] | None = None,
        model_map: dict[str, list[str]] | None = None,
    ) -> TestServer:
        urls = [self._urls[n] for n in names]
        if states is None:
            states = {u: state() for u in urls}
        cfg = config or Config(vllm_urls=urls, max_retries=2, retry_backoff=0.01, timeout=5)
        self.lb = TestServer(build_app(cfg, states, model_map))
        await self.lb.start_server()
        return self.lb

    async def close(self) -> None:
        for server in [*self._servers, self.lb]:
            if server is not None:
                await server.close()


@pytest_asyncio.fixture
async def fleet() -> AsyncIterator[Fleet]:
    f = Fleet()
    try:
        yield f
    finally:
        await f.close()


async def read_leniently(response: ClientResponse) -> bytes:
    """Read a body that may be truncated because the upstream died mid-stream."""
    try:
        return await response.read()
    except Exception:  # noqa: BLE001 - a truncated stream is a valid outcome here
        return b""


__all__ = ["ClientResponse", "Fleet", "Handler", "read_leniently", "state"]
