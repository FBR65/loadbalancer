"""SC-4: one ClientSession for the whole app lifetime; /v1/models fans out."""

import asyncio
import time

from aiohttp import web
from aiohttp.test_utils import TestClient
from conftest import Fleet

from loadbalancer.proxy import PROXY_KEY


async def _ok(request: web.Request) -> web.StreamResponse:
    return web.json_response({"ok": True})


async def test_all_requests_share_one_session(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))
    server = fleet.lb
    assert server is not None

    async with TestClient(lb) as client:
        await client.post("/v1/chat/completions", json={"model": "m"})
        first = server.app[PROXY_KEY].session
        await client.post("/v1/chat/completions", json={"model": "m"})
        second = server.app[PROXY_KEY].session

    assert first is second, "each request built its own ClientSession/TCP connector"


async def test_session_is_closed_when_the_app_shuts_down(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))
    server = fleet.lb
    assert server is not None

    async with TestClient(lb) as client:
        await client.post("/v1/chat/completions", json={"model": "m"})
        session = server.app[PROXY_KEY].session
        assert not session.closed

    assert session.closed, "session leaked past app cleanup"


async def test_list_models_queries_backends_concurrently(fleet: Fleet) -> None:
    for i in range(3):
        model_id = f"m{i}"

        async def handler(request: web.Request, m: str = model_id) -> web.StreamResponse:
            await asyncio.sleep(0.3)
            return web.json_response({"object": "list", "data": [{"id": m}]})

        await fleet.backend(f"b{i}", handler)

    lb = await fleet.start_lb(["b0", "b1", "b2"])

    async with TestClient(lb) as client:
        started = time.monotonic()
        response = await client.get("/v1/models")
        elapsed = time.monotonic() - started
        data = await response.json()

    assert response.status == 200
    assert {m["id"] for m in data["data"]} == {"m0", "m1", "m2"}
    assert elapsed < 0.7, f"backends queried serially ({elapsed:.2f}s for 3x0.3s)"


async def test_list_models_skips_broken_backends(fleet: Fleet) -> None:
    async def ok(request: web.Request) -> web.StreamResponse:
        return web.json_response({"object": "list", "data": [{"id": "good"}]})

    async def broken(request: web.Request) -> web.StreamResponse:
        raise web.HTTPInternalServerError()

    await fleet.backend("ok", ok)
    await fleet.backend("broken", broken)
    lb = await fleet.start_lb(["ok", "broken"])

    async with TestClient(lb) as client:
        response = await client.get("/v1/models")
        data = await response.json()

    assert response.status == 200
    assert [m["id"] for m in data["data"]] == ["good"]


async def test_list_models_503_when_every_backend_fails(fleet: Fleet) -> None:
    async def broken(request: web.Request) -> web.StreamResponse:
        return web.Response(status=500, text="boom")

    await fleet.backend("a", broken)
    lb = await fleet.start_lb(["a"])

    async with TestClient(lb) as client:
        response = await client.get("/v1/models")
        body = await response.read()

    assert response.status == 503
    assert b"no models available" in body


async def test_list_models_survives_a_backend_answering_garbage(fleet: Fleet) -> None:
    async def broken(request: web.Request) -> web.StreamResponse:
        return web.Response(text="not json", content_type="application/json")

    async def ok(request: web.Request) -> web.StreamResponse:
        return web.json_response({"object": "list", "data": [{"id": "good"}]})

    await fleet.backend("broken", broken)
    await fleet.backend("ok", ok)
    lb = await fleet.start_lb(["broken", "ok"])

    async with TestClient(lb) as client:
        response = await client.get("/v1/models")
        data = await response.json()

    assert response.status == 200
    assert [m["id"] for m in data["data"]] == ["good"]
