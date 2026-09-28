"""SC-2: response headers forwarded to the client must describe the real body."""

import gzip

from aiohttp import web
from aiohttp.test_utils import TestClient
from conftest import Fleet

from vllm_lb.proxy import HOP_BY_HOP, _forwardable_headers

# `upgrade`/`proxy-*` change how aiohttp's own server answers, so they are
# covered by the unit test below rather than end to end.
TOLERATED_BY_AIOHTTP = ("connection", "keep-alive", "te", "trailer")


async def test_gzip_body_is_delivered_decoded_without_encoding_header(fleet: Fleet) -> None:
    async def gzipped(request: web.Request) -> web.StreamResponse:
        return web.Response(
            body=gzip.compress(b'{"ok": true}'),
            headers={"content-encoding": "gzip", "content-type": "application/json"},
        )

    await fleet.backend("a", gzipped)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        body = await response.read()

    assert response.status == 200
    assert body == b'{"ok": true}'
    assert response.headers.get("content-encoding") is None
    assert response.headers.get("content-length") == str(len(body))


async def test_hop_by_hop_headers_are_not_forwarded(fleet: Fleet) -> None:
    async def noisy(request: web.Request) -> web.StreamResponse:
        return web.Response(
            body=b'{"ok": true}',
            headers={h: "x" for h in TOLERATED_BY_AIOHTTP} | {"content-type": "application/json"},
        )

    await fleet.backend("a", noisy)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        await response.read()

    leaked = {h for h in TOLERATED_BY_AIOHTTP if h in response.headers}
    assert leaked == set(), f"hop-by-hop headers leaked: {sorted(leaked)}"
    assert response.headers.get("content-type") == "application/json"


async def test_streamed_response_is_framed_as_chunked(fleet: Fleet) -> None:
    async def sse(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: 1\n\n")
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    await fleet.backend("a", sse)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m", "stream": True})
        body = await response.read()

    assert body == b"data: 1\n\ndata: [DONE]\n\n"
    assert "content-length" not in response.headers, "a stream has no known length"
    assert response.headers.get("transfer-encoding") == "chunked"


def test_forwardable_headers_drops_every_hop_by_hop_and_entity_header() -> None:
    given = {name: "x" for name in sorted(HOP_BY_HOP)}
    given |= {"content-encoding": "gzip", "content-length": "99", "x-keep": "1"}

    kept = _forwardable_headers(given)

    assert kept == {"x-keep": "1"}


def test_forwardable_headers_is_case_insensitive_and_idempotent() -> None:
    given = {"Connection": "close", "Content-Encoding": "gzip", "X-Trace": "t"}

    once = _forwardable_headers(given)

    assert once == {"X-Trace": "t"}
    assert _forwardable_headers(once) == once
