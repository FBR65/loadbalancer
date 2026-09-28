"""SC-1: no retry once the response has been committed to the client."""

from aiohttp import web
from aiohttp.test_utils import TestClient

from conftest import Fleet, read_leniently


async def _half_streamed_then_dead(request: web.Request) -> web.StreamResponse:
    """Writes one SSE chunk, then dies: the client already saw that chunk."""
    response = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
    await response.prepare(request)
    await response.write(b'data: {"tok": 1}\n\n')
    response.force_close()
    raise web.HTTPBadGateway()


async def _marks_itself(request: web.Request) -> web.StreamResponse:
    return web.json_response({"served_by": "healthy"})


async def test_stream_failure_does_not_retry_onto_another_instance(fleet: Fleet) -> None:
    """The broken stream is NOT retried: a second instance must not be asked."""
    await fleet.backend("flaky", _half_streamed_then_dead)
    await fleet.backend("healthy", _marks_itself)
    states = fleet.states(flaky={"queue_time_sum": 0.0}, healthy={"queue_time_sum": 50.0})
    lb = await fleet.start_lb(["flaky", "healthy"], states=states)

    async with TestClient(lb) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "stream": True}
        )
        body = await read_leniently(response)

    assert fleet.hits["flaky"] == ["/v1/chat/completions"], "flaky instance must be tried once"
    assert fleet.hits["healthy"] == [], "no retry may hit another instance after a commit"
    assert b"served_by" not in body, "client must never receive another instance's body"


async def test_truncated_stream_reaches_the_client_as_truncated(fleet: Fleet) -> None:
    """The client keeps the partial stream it already got; no 200+foreign-body."""
    await fleet.backend("flaky", _half_streamed_then_dead)
    states = fleet.states(flaky={"queue_time_sum": 0.0})
    lb = await fleet.start_lb(["flaky"], states=states)

    async with TestClient(lb) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "stream": True}
        )
        body = await read_leniently(response)

    assert b"served_by" not in body
    assert body in (b"", b'data: {"tok": 1}\n\n')


async def test_retryable_status_before_commit_is_still_retried(fleet: Fleet) -> None:
    """SC-1 must not weaken the existing retry budget (N6)."""
    calls: list[str] = []

    async def fails_twice_then_ok(request: web.Request) -> web.StreamResponse:
        calls.append(request.path_qs)
        if len(calls) < 3:
            return web.Response(status=503, text="busy")
        return web.json_response({"served_by": "healthy"})

    await fleet.backend("a", fails_twice_then_ok)
    states = fleet.states(a={})
    lb = await fleet.start_lb(["a"], states=states)

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        body = await response.json()

    assert len(calls) == 3, "1 attempt + MAX_RETRIES(2) retries"
    assert response.status == 200
    assert body == {"served_by": "healthy"}


async def test_client_error_is_never_retried(fleet: Fleet) -> None:
    """4xx stays a client error (README retry rule 1)."""
    calls: list[str] = []

    async def rejects(request: web.Request) -> web.StreamResponse:
        calls.append(request.path_qs)
        return web.Response(status=400, text="bad prompt")

    await fleet.backend("a", rejects)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})

    assert calls == ["/v1/chat/completions"]
    assert response.status == 400
