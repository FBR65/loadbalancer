"""SC-2: response headers forwarded to the client must describe the real body."""

import gzip

from aiohttp import web
from aiohttp.test_utils import TestClient

from conftest import Fleet

HOP_BY_HOP = (
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
)


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
            headers={h: "x" for h in HOP_BY_HOP} | {"content-type": "application/json"},
        )

    await fleet.backend("a", noisy)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        await response.read()

    leaked = {h for h in HOP_BY_HOP if h in response.headers}
    assert leaked == set(), f"hop-by-hop headers leaked: {sorted(leaked)}"
    assert response.headers.get("content-type") == "application/json"


async def test_streamed_response_has_no_stale_content_length(fleet: Fleet) -> None:
    async def sse(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={"content-type": "text/event-stream", "content-length": "0"},
        )
        await response.prepare(request)
        await response.write(b"data: 1\n\n")
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    await fleet.backend("a", sse)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "stream": True}
        )
        body = await response.read()

    assert body == b"data: 1\n\ndata: [DONE]\n\n"
    assert response.headers.get("content-length") != "0"
