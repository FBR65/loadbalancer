"""Routing/refusal behaviour of _forward, and the retry rules it must preserve."""

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient
from conftest import Fleet, state

from loadbalancer.config import Config
from loadbalancer.proxy import PROXY_KEY


async def _ok(request: web.Request) -> web.StreamResponse:
    return web.json_response({"ok": True})


def _states(urls: list[str], per: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {u: state(**per.get(u, {})) for u in urls}


async def test_non_dict_json_body_is_not_model_routed(fleet: Fleet) -> None:
    seen: list[bytes] = []

    async def echo(request: web.Request) -> web.StreamResponse:
        seen.append(await request.read())
        return web.json_response({"ok": True})

    await fleet.backend("a", echo)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", data=json.dumps([1, 2, 3]))
        body = await response.json()

    assert response.status == 200
    assert body == {"ok": True}
    assert seen == [b"[1, 2, 3]"]


async def test_503_names_the_model_when_its_group_is_down(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    url = fleet.url("a")
    states = {url: None}  # unhealthy: no metrics
    lb = await fleet.start_lb(["a"], states=states, model_map={"gpt-oss": [url]})

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "gpt-oss"})
        text = await response.text()

    assert response.status == 503
    assert "no healthy instance for model gpt-oss" in text


async def test_503_when_every_instance_is_down(fleet: Fleet) -> None:
    await fleet.backend("a", _ok)
    url = fleet.url("a")
    lb = await fleet.start_lb(["a"], states={url: None})

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "gpt-oss"})
        text = await response.text()

    assert response.status == 503
    assert "no healthy vllm instance" in text


async def test_retry_skips_an_overloaded_instance(fleet: Fleet) -> None:
    """Rule 4: a 503 must not be retried into an instance that is saturated."""
    calls: list[str] = []

    async def busy(request: web.Request) -> web.StreamResponse:
        calls.append("a")
        return web.Response(status=503, text="busy")

    await fleet.backend("a", busy)
    await fleet.backend("b", _ok)
    states = _states(
        [fleet.url("a"), fleet.url("b")],
        {
            fleet.url("a"): {"queue_time_sum": 0.0},
            fleet.url("b"): {"queue_time_sum": 1.0, "waiting": 99},
        },
    )
    config = Config(
        vllm_urls=[fleet.url("a"), fleet.url("b")],
        max_retries=1,
        retry_backoff=0.01,
        timeout=5,
    )
    lb = await fleet.start_lb([], config=config, states=states)

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        text = await response.text()

    assert response.status == 504
    assert "overloaded" in text
    assert fleet.hits["a"] == ["/v1/chat/completions"]
    assert fleet.hits["b"] == [], "an overloaded instance must not be retried into"


async def test_connection_error_is_retried_on_a_fresh_instance(fleet: Fleet) -> None:
    async def served(request: web.Request) -> web.StreamResponse:
        return web.json_response({"served_by": "good"})

    await fleet.backend("good", served)
    good = fleet.url("good")
    dead = "http://127.0.0.1:1"  # nothing listens: connect fails immediately
    states = _states(
        [dead, good],
        {dead: {"queue_time_sum": 0.0}, good: {"queue_time_sum": 50.0}},
    )
    config = Config(vllm_urls=[dead, good], max_retries=2, retry_backoff=0.01, timeout=5)
    lb = await fleet.start_lb([], config=config, states=states)

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})
        body = await response.json()

    assert response.status == 200
    assert body == {"served_by": "good"}
    assert fleet.hits["good"] == ["/v1/chat/completions"]


async def test_inflight_slot_is_released_after_every_request(fleet: Fleet) -> None:
    """Two overlapping requests on one instance must not leak in-flight slots."""
    gate = asyncio.Event()
    seen: list[int] = []

    async def gated(request: web.Request) -> web.StreamResponse:
        seen.append(1)
        await gate.wait()
        return web.json_response({"ok": True})

    await fleet.backend("a", gated)
    lb = await fleet.start_lb(["a"], states=fleet.states(a={}))
    server = fleet.lb
    assert server is not None
    inflight = server.app[PROXY_KEY].inflight

    async with TestClient(lb) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "m"}))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "m"}))
        await asyncio.sleep(0.05)
        gate.set()
        responses = await asyncio.gather(first, second)

    assert [r.status for r in responses] == [200, 200]
    assert len(seen) == 2, "both requests must overlap on the same instance"
    assert inflight == {}, f"in-flight slots leaked: {inflight}"
