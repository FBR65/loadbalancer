"""SC-5: the balancer exports its own metrics and a truthful readiness probe."""

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient
from conftest import Fleet

from loadbalancer.telemetry import Telemetry, label_value

TEXT = "text/plain; version=0.0.4; charset=utf-8"


async def _ok(_request: web.Request) -> web.StreamResponse:
    return web.json_response({"ok": True})


def _value(body: str, name: str) -> float:
    line = next(line for line in body.splitlines() if line.startswith(name))
    return float(line.split()[-1])


async def test_metrics_are_served_by_the_balancer_itself(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.get("/metrics")
        body = await response.text()

    assert response.status == 200
    assert response.headers["Content-Type"] == TEXT
    assert "loadbalancer_requests_total" in body
    assert "loadbalancer_endpoints_healthy" in body


async def test_metrics_count_served_requests_and_per_endpoint_health(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    await fleet.backend("b", _ok)
    states = fleet.states(a={}, b={})
    states[fleet.url("b")] = None  # b never reported metrics -> unhealthy
    lb = await fleet.start_lb(["a", "b"], states=states)

    async with TestClient(lb) as client:
        await client.post("/v1/chat/completions", json={"model": "m"})
        body = await (await client.get("/metrics")).text()

    assert _value(body, "loadbalancer_requests_total") >= 1
    assert _value(body, "loadbalancer_endpoints_healthy") == 1
    assert f'loadbalancer_endpoint_healthy{{endpoint="{fleet.url("a")}"}} 1' in body
    assert f'loadbalancer_endpoint_healthy{{endpoint="{fleet.url("b")}"}} 0' in body


async def test_inflight_gauge_tracks_requests_in_flight(fleet: Fleet) -> None:
    release = asyncio.Event()

    async def slow(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: 1\n\n")
        await release.wait()
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    await fleet.backend("a", slow)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        task = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "m", "stream": True})
        )
        needle = f'loadbalancer_inflight_requests{{endpoint="{fleet.url("a")}"}} 1'
        body = ""
        for _ in range(500):
            body = await (await client.get("/metrics")).text()
            if needle in body:
                break
            await asyncio.sleep(0.01)
        release.set()
        await task

    assert needle in body, f"in-flight gauge never reached 1:\n{body}"


async def test_ready_reports_ready_while_a_backend_is_healthy(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.get("/ready")
        payload = await response.json()

    assert response.status == 200
    assert payload["status"] == "ready"
    assert payload["healthy_backends"] == 1


async def test_ready_fails_when_every_backend_is_down(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    states = fleet.states(a={})
    states[fleet.url("a")] = None
    lb = await fleet.start_lb(["a"], states=states)

    async with TestClient(lb) as client:
        response = await client.get("/ready")
        payload = await response.json()

    assert response.status == 503
    assert payload["status"] == "not ready"


async def test_health_stays_a_liveness_probe_when_backends_are_down(fleet: Fleet) -> None:
    """/health must not become the readiness probe: the process is alive."""
    lb = await fleet.start_lb([], states={"http://127.0.0.1:1": None})

    async with TestClient(lb) as client:
        response = await client.get("/health")
        payload = await response.json()

    assert response.status == 200
    assert payload["status"] == "ok"


async def test_metrics_endpoint_is_not_proxied_to_a_backend(fleet: Fleet) -> None:
    """/metrics used to be relayed; it now answers the balancer's own counters."""
    await fleet.backend("a", _ok)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        body = await (await client.get("/metrics")).text()

    assert "served_by" not in body
    assert fleet.hits["a"] == [], "/metrics must not be forwarded upstream"


def test_counter_starts_at_zero_and_renders_help_and_type() -> None:
    body = Telemetry().render({}, {})

    assert "# HELP loadbalancer_requests_total" in body
    assert "# TYPE loadbalancer_requests_total counter" in body
    assert "loadbalancer_requests_total 0" in body


def test_label_value_escapes_characters_that_would_break_the_format() -> None:
    """An endpoint URL is config, not a constant. A raw quote would emit a
    second label and silently change the meaning of every later series."""
    hostile = 'http://h/"} evil{x="'

    escaped = label_value(hostile)

    assert escaped == 'http://h/\\"} evil{x=\\"'
    assert "\n" not in escaped
    line = Telemetry().render({hostile: {"running": 0.0}}, {}).splitlines()[-1]
    assert line == f'loadbalancer_inflight_requests{{endpoint="{escaped}"}} 0'


def test_label_value_escapes_backslash_before_quotes() -> None:
    assert label_value('a\\"b') == 'a\\\\\\"b'


def test_render_stays_valid_when_a_backend_never_reported_metrics() -> None:
    body = Telemetry().render({"http://a": None, "http://b": None}, {})

    assert "loadbalancer_endpoints_healthy 0" in body
    assert 'loadbalancer_endpoint_healthy{endpoint="http://a"} 0' in body
    assert 'loadbalancer_endpoint_healthy{endpoint="http://b"} 0' in body


def test_render_marks_a_reporting_backend_healthy() -> None:
    body = Telemetry().render({"http://a": {"running": 0.0}}, {})

    assert "loadbalancer_endpoints_healthy 1" in body
    assert 'loadbalancer_endpoint_healthy{endpoint="http://a"} 1' in body
