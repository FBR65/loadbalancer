"""SC-4: `max_body_size` and `listen_port` follow the config without a restart."""

import asyncio
import socket

import pytest
from aiohttp import ClientConnectorError, ClientSession, web
from aiohttp.test_utils import TestClient
from conftest import Fleet

from loadbalancer.config import Config
from loadbalancer.main import rebind_port
from loadbalancer.proxy import build_app

PAYLOAD = b'{"model": "m", "pad": "' + b"x" * 4096 + b'"}'


async def _ok(_request: web.Request) -> web.StreamResponse:
    return web.json_response({"ok": True})


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _lb_config(url: str, max_body_size: int) -> Config:
    return Config(
        vllm_urls=[url], max_retries=0, retry_backoff=0.01, timeout=5, max_body_size=max_body_size
    )


async def test_a_body_over_max_body_size_is_rejected_with_413(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    config = _lb_config(fleet.url("a"), 256)
    lb = await fleet.start_lb(["a"], config=config, states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post(
            "/v1/chat/completions", data=PAYLOAD, headers={"content-type": "application/json"}
        )

    assert response.status == 413
    assert fleet.hits["a"] == [], "an oversized body must not reach a backend"


async def test_max_body_size_is_read_per_request_not_frozen_at_startup(fleet: Fleet) -> None:
    """Lowering the limit at runtime must take effect without a new process."""
    await fleet.backend("a", _ok)
    config = _lb_config(fleet.url("a"), 1 << 20)
    lb = await fleet.start_lb(["a"], config=config, states=fleet.states(a={}))

    async with TestClient(lb) as client:
        before = await client.post("/v1/chat/completions", data=PAYLOAD)
        config.max_body_size = 128  # hot reload
        after = await client.post("/v1/chat/completions", data=PAYLOAD)

    assert before.status == 200
    assert after.status == 413


async def test_raising_max_body_size_at_runtime_admits_the_larger_body(fleet: Fleet) -> None:
    """A limit raised above the startup value must not stay capped by it."""
    await fleet.backend("a", _ok)
    config = _lb_config(fleet.url("a"), 128)
    lb = await fleet.start_lb(["a"], config=config, states=fleet.states(a={}))

    async with TestClient(lb) as client:
        before = await client.post("/v1/chat/completions", data=PAYLOAD)
        config.max_body_size = 1 << 20
        after = await client.post("/v1/chat/completions", data=PAYLOAD)

    assert before.status == 413
    assert after.status == 200, "the raised limit must take effect, not the startup one"


async def test_a_body_without_content_length_is_still_limited(fleet: Fleet) -> None:
    """A chunked request declares no length, so the bytes must be counted.

    Sent over a raw socket because aiohttp's client always writes a
    Content-Length, which would make the declared-length path the only one
    under test.
    """
    await fleet.backend("a", _ok)
    config = _lb_config(fleet.url("a"), 256)
    lb = await fleet.start_lb(["a"], config=config, states=fleet.states(a={}))

    reader, writer = await asyncio.open_connection("127.0.0.1", lb.port)
    try:
        writer.write(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: lb\r\n"
            b"content-type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        writer.write(f"{len(PAYLOAD):x}\r\n".encode() + PAYLOAD + b"\r\n0\r\n\r\n")
        await writer.drain()
        status_line = await asyncio.wait_for(reader.readline(), 5)
    finally:
        writer.close()
        await writer.wait_closed()

    assert status_line.startswith(b"HTTP/1.1 413"), status_line
    assert fleet.hits["a"] == []


async def test_rebind_port_moves_the_listener_and_frees_the_old_port() -> None:
    app = web.Application()
    app.router.add_get("/", _ok)
    runner = web.AppRunner(app)
    await runner.setup()
    first, second = _free_port(), _free_port()
    site = web.TCPSite(runner, "127.0.0.1", first)
    await site.start()
    moved: web.TCPSite | None = None
    try:
        moved = await rebind_port(runner, site, second)

        assert moved is not site
        async with ClientSession() as client:
            reached = await client.get(f"http://127.0.0.1:{second}/")
            assert reached.status == 200
            with pytest.raises(ClientConnectorError):
                await client.get(f"http://127.0.0.1:{first}/")
    finally:
        await (moved or site).stop()
        await runner.cleanup()


async def test_rebind_port_keeps_serving_when_the_new_port_cannot_be_bound() -> None:
    """A typo in the config must not take the balancer down."""
    app = web.Application()
    app.router.add_get("/", _ok)
    runner = web.AppRunner(app)
    await runner.setup()
    first = _free_port()
    site = web.TCPSite(runner, "127.0.0.1", first)
    await site.start()
    try:
        result = await rebind_port(runner, site, 1)  # port 1 needs root

        assert result is site, "the old site must stay up when the new bind fails"
        async with ClientSession() as client:
            reached = await client.get(f"http://127.0.0.1:{first}/")
            assert reached.status == 200
    finally:
        await site.stop()
        await runner.cleanup()


async def test_build_app_does_not_freeze_the_configured_body_limit() -> None:
    """`client_max_size` is a backstop only; the real limit must not be baked in."""
    url = "http://127.0.0.1:1"
    tight = build_app(_lb_config(url, 128))
    roomy = build_app(_lb_config(url, 1 << 20))

    assert tight is not roomy
    assert roomy._client_max_size >= (1 << 20), (
        "a raised limit must not be capped by the startup backstop"
    )
