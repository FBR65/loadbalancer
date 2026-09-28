"""SC-3: the request target (path *and* query) reaches the upstream verbatim."""

from aiohttp import web
from aiohttp.test_utils import TestClient
from conftest import Fleet


async def _echo(request: web.Request) -> web.StreamResponse:
    return web.json_response(
        {
            "raw_path_qs": request.rel_url.raw_path_qs,
            "path_qs": request.path_qs,
            "query": dict(request.query),
            "host": request.headers.get("host"),
        }
    )


async def test_query_string_is_forwarded(fleet: Fleet) -> None:
    await fleet.backend("a", _echo)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions?trace=abc&user=7", json={"model": "m"})
        echoed = await response.json()

    assert echoed["query"] == {"trace": "abc", "user": "7"}
    assert echoed["path_qs"] == "/v1/chat/completions?trace=abc&user=7"
    assert echoed["raw_path_qs"] == "/v1/chat/completions?trace=abc&user=7"


async def test_percent_encoded_path_and_query_are_forwarded_verbatim(fleet: Fleet) -> None:
    """The raw target must survive: no decoding, no double-encoding, no query loss."""
    await fleet.backend("a", _echo)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    target = "/v1/responses/abc%20def?x=1"
    async with TestClient(lb) as client:
        response = await client.get(target)
        echoed = await response.json()

    assert echoed["raw_path_qs"] == target, "path must not be decoded or double-encoded"
    assert echoed["query"] == {"x": "1"}


async def test_query_string_survives_a_retry(fleet: Fleet) -> None:
    calls: list[str] = []

    async def flaky(request: web.Request) -> web.StreamResponse:
        calls.append(request.rel_url.raw_path_qs)
        if len(calls) == 1:
            return web.Response(status=502, text="bad gateway")
        return web.json_response({"ok": True})

    await fleet.backend("a", flaky)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/completions?best_of=2", json={"model": "m"})

    assert response.status == 200
    assert calls == [
        "/v1/completions?best_of=2",
        "/v1/completions?best_of=2",
    ]


async def test_request_without_query_is_unchanged(fleet: Fleet) -> None:
    await fleet.backend("a", _echo)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        echoed = await response.json()

    assert echoed["raw_path_qs"] == "/v1/chat/completions"
    assert echoed["query"] == {}
