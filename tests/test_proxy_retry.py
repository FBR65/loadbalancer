"""SC-1: no retry once the response has been committed to the client."""

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient
from conftest import Fleet, read_leniently, read_outcome

from loadbalancer.config import Config

# A regression that makes the proxy retry after the response was committed
# leaves the client socket open forever (no second status line can be sent).
# These tests bound the read so such a regression shows up as a fast, clean
# assertion failure instead of a hung suite.
READ_TIMEOUT = 5


async def _half_streamed_then_dead(request: web.Request) -> web.StreamResponse:
    """Writes one SSE chunk, then dies mid-body: the client already saw it.

    A real backend that loses the process leaves a truncated chunked body, so
    the connection is aborted instead of a second status line being written.
    """
    response = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
    await response.prepare(request)
    await response.write(b'data: {"tok": 1}\n\n')
    transport = request.transport
    assert transport is not None
    transport.abort()
    return response


async def _marks_itself(request: web.Request) -> web.StreamResponse:
    return web.json_response({"served_by": "healthy"})


async def test_stream_failure_does_not_retry_onto_another_instance(fleet: Fleet) -> None:
    """The broken stream is NOT retried: a second instance must not be asked."""
    await fleet.backend("flaky", _half_streamed_then_dead)
    await fleet.backend("healthy", _marks_itself)
    states = fleet.states(flaky={"queue_time_sum": 0.0}, healthy={"queue_time_sum": 50.0})
    lb = await fleet.start_lb(["flaky", "healthy"], states=states)

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m", "stream": True})
        try:
            body = await asyncio.wait_for(read_leniently(response), READ_TIMEOUT)
        except TimeoutError:
            body = b""

    assert fleet.hits["flaky"] == ["/v1/chat/completions"], "flaky instance must be tried once"
    assert fleet.hits["healthy"] == [], "no retry may hit another instance after a commit"
    assert b"served_by" not in body, "client must never receive another instance's body"


async def test_truncated_stream_reaches_the_client_as_truncated(fleet: Fleet) -> None:
    """The client keeps the partial stream it already got; no 200+foreign-body."""
    await fleet.backend("flaky", _half_streamed_then_dead)
    states = fleet.states(flaky={"queue_time_sum": 0.0})
    lb = await fleet.start_lb(["flaky"], states=states)

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m", "stream": True})
        body = await read_leniently(response)

    assert b"served_by" not in body
    assert body in (b"", b'data: {"tok": 1}\n\n')


async def test_broken_stream_is_signalled_to_the_client_not_silently_truncated(
    fleet: Fleet,
) -> None:
    """A cut-off stream must be distinguishable from a normally finished one.

    Silence is the real defect: a client that reads a clean end of body cannot
    tell "the model finished" from "the backend died halfway" and will happily
    return a half answer as a complete one.
    """
    await fleet.backend("flaky", _half_streamed_then_dead)
    lb = await fleet.start_lb(["flaky"], states=fleet.states(flaky={"queue_time_sum": 0.0}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m", "stream": True})
        try:
            body, error = await asyncio.wait_for(read_outcome(response), READ_TIMEOUT)
        except TimeoutError:
            body, error = b"", TimeoutError()

    signalled = error is not None or b"event: error" in body
    assert signalled, "a broken stream must be observable by the client"
    assert b"data: [DONE]" not in body, "a cut-off stream must not look complete"


async def test_broken_sse_stream_ends_with_a_terminal_error_event(fleet: Fleet) -> None:
    """An SSE client must learn *what* failed, not just that the body ended.

    A raw client sees only an incomplete body. An SDK-based client parses the
    event stream, so the failure has to be a terminal `error` event to be
    actionable rather than an opaque transport error.
    """
    await fleet.backend("flaky", _half_streamed_then_dead)
    lb = await fleet.start_lb(["flaky"], states=fleet.states(flaky={"queue_time_sum": 0.0}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m", "stream": True})
        try:
            body, _ = await asyncio.wait_for(read_outcome(response), READ_TIMEOUT)
        except TimeoutError:
            body = b""

    assert b"event: error" in body, f"no terminal error event in {body!r}"
    assert b"upstream" in body.lower(), "the error event must name the failure"


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


async def test_504_names_the_upstream_error(fleet: Fleet) -> None:
    """An unreachable backend must not produce `upstream error: None`."""
    dead = "http://127.0.0.1:1"
    config = Config(vllm_urls=[dead], max_retries=0, retry_backoff=0.01, timeout=2)
    states = {
        dead: {
            "running": 0.0,
            "waiting": 0.0,
            "queue_time_sum": 0.0,
            "kv_cache_usage_perc": 0.1,
            "preemptions": 0.0,
            "sleeping": False,
        }
    }
    lb = await fleet.start_lb([], config=config, states=states)

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        body = await response.text()

    assert response.status == 504
    assert "None" not in body
    assert dead in body, f"the 504 must identify the failing instance: {body!r}"
