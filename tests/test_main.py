"""main.py: metrics URL candidates, model map, state derivation, poll loop."""

import asyncio
import contextlib
from typing import cast

import aiohttp
import pytest
from aiohttp import web
from conftest import Fleet

from loadbalancer.config import Config
from loadbalancer.main import (
    State,
    _compute_state,
    fetch_model_map,
    metrics_urls,
    poll_metrics,
    write_model_map,
)
from loadbalancer.metrics import MetricsResult


def test_metrics_urls_plain_base() -> None:
    assert metrics_urls("http://vllm1:8000") == ["http://vllm1:8000/metrics"]


def test_metrics_urls_strips_trailing_slash() -> None:
    assert metrics_urls("http://vllm1:8000/") == ["http://vllm1:8000/metrics"]


def test_metrics_urls_adds_root_fallback_for_path_prefix() -> None:
    assert metrics_urls("https://host/v1") == ["https://host/v1/metrics", "https://host/metrics"]


def test_metrics_urls_no_fallback_for_bare_root() -> None:
    assert metrics_urls("https://host") == ["https://host/metrics"]
    assert metrics_urls("https://host/") == ["https://host/metrics"]


async def test_fetch_model_map_merges_endpoints(fleet: Fleet) -> None:
    for name, model in (("a", "m1"), ("b", "m1"), ("c", "m2")):

        async def models(request: web.Request, m: str = model) -> web.StreamResponse:
            return web.json_response({"data": [{"id": m}]})

        await fleet.backend(name, models)
    cfg = Config(vllm_urls=[fleet.url("a"), fleet.url("b"), fleet.url("c")])
    async with aiohttp.ClientSession() as session:
        mapping = await fetch_model_map(cfg, session)
    assert mapping == {"m1": [fleet.url("a"), fleet.url("b")], "m2": [fleet.url("c")]}


def test_write_model_map_roundtrips() -> None:
    import json
    import os

    mapping = {"m1": ["http://a:1"]}
    path = write_model_map(mapping)
    try:
        with open(path) as f:
            assert json.load(f) == mapping
    finally:
        os.unlink(path)


async def _poll_until(
    config: Config,
    states: State,
    session: aiohttp.ClientSession,
    predicate: object,
) -> None:
    task = asyncio.create_task(poll_metrics(config, states, session))
    try:
        for _ in range(200):
            if predicate():  # type: ignore[operator]
                return
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_poll_metrics_marks_healthy(fleet: Fleet) -> None:
    async def ok(request: web.Request) -> web.StreamResponse:
        return web.Response(text="vllm:num_requests_running 1.0\nvllm:num_requests_waiting 2.0\n")

    await fleet.backend("a", ok)
    url = fleet.url("a")
    states: State = {url: None}
    async with aiohttp.ClientSession() as session:
        await _poll_until(
            Config(vllm_urls=[url], poll_interval=0.05),
            states,
            session,
            lambda: states.get(url) is not None,
        )
    assert states[url] is not None
    assert states[url]["running"] == 1.0  # type: ignore[index]
    assert states[url]["waiting"] == 2.0  # type: ignore[index]


async def test_removed_instance_is_dropped_from_states(fleet: Fleet) -> None:
    """A URL removed by hot reload must leave the routing pool (states)."""

    async def ok(request: web.Request) -> web.StreamResponse:
        return web.Response(text="vllm:num_requests_running 0.0\n")

    await fleet.backend("a", ok)
    await fleet.backend("b", ok)
    a, b = fleet.url("a"), fleet.url("b")
    config = Config(vllm_urls=[a, b], poll_interval=0.05, timeout=1)
    states: State = {a: None, b: None}
    async with aiohttp.ClientSession() as session:
        task = asyncio.create_task(poll_metrics(config, states, session))
        try:
            for _ in range(200):
                if states.get(a) is not None and states.get(b) is not None:
                    break
                await asyncio.sleep(0.02)
            assert states.get(b) is not None, "both backends should start healthy"
            config.vllm_urls = [a]  # b leaves the fleet
            for _ in range(200):
                if b not in states:
                    break
                await asyncio.sleep(0.02)
            assert b not in states, "removed instance still routable via states"
            assert a in states
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize(
    "base,expected",
    [
        ("http://h:1", ["http://h:1/metrics"]),
        ("http://h:1/pfx", ["http://h:1/pfx/metrics", "http://h:1/metrics"]),
    ],
)
def test_metrics_urls_parametrized(base: str, expected: list[str]) -> None:
    assert metrics_urls(base) == expected


# --- state derivation (pure function, no I/O) -------------------------------

SAMPLE: MetricsResult = {
    "running": 1.0,
    "waiting": 4.0,
    "queue_time_sum": 10.0,
    "queue_time_count": 2,
    "waiting_by_reason": {},
    "kv_cache_usage_perc": 0.3,
    "preemptions": 0.0,
    "sleep_state": None,
}


def _metrics(**over: object) -> MetricsResult:
    return cast(MetricsResult, {**SAMPLE, **over})


def test_queue_time_uses_delta_not_raw_counter() -> None:
    """The raw _sum grows forever; only the per-poll delta is a load signal."""
    prev: dict[str, tuple[float, int]] = {}
    prev_p: dict[str, float] = {}
    states: State = {}
    first = _compute_state("u", _metrics(), states, prev, prev_p)
    assert first["queue_time_sum"] == 0.0, "no baseline yet -> no rate"
    second = _compute_state(
        "u",
        _metrics(queue_time_sum=30.0, queue_time_count=4),
        states,
        prev,
        prev_p,
    )
    assert second["queue_time_sum"] == 10.0, "(30-10)/(4-2) == 10"


def test_counter_reset_yields_zero_rate() -> None:
    """A vLLM restart resets the counters; a negative rate must never reach the score."""
    prev: dict[str, tuple[float, int]] = {}
    prev_p: dict[str, float] = {}
    states: State = {}
    _compute_state("u", _metrics(queue_time_sum=100.0, queue_time_count=10), states, prev, prev_p)
    after_reset = _compute_state(
        "u",
        _metrics(queue_time_sum=5.0, queue_time_count=1),
        states,
        prev,
        prev_p,
    )
    assert after_reset["queue_time_sum"] == 0.0


def test_capacity_waiting_is_load_but_deferred_is_not() -> None:
    state = _compute_state(
        "u",
        _metrics(
            waiting_by_reason={"capacity": 7.0, "engine_busy": 3.0, "kv_budget": 2.0},
        ),
        {},
        {},
        {},
    )
    assert state["waiting"] == 7.0
    assert state["waiting_deferred"] == 5.0


def test_waiting_falls_back_to_total_when_no_reasons() -> None:
    state = _compute_state("u", _metrics(waiting=9.0, waiting_by_reason={}), {}, {}, {})
    assert state["waiting"] == 9.0
    assert state["waiting_deferred"] == 0.0


def test_preemption_delta_and_reset() -> None:
    prev_p: dict[str, float] = {}
    states: State = {}
    prev: dict[str, tuple[float, int]] = {}
    _compute_state("u", _metrics(preemptions=5.0), states, prev, prev_p)
    grew = _compute_state("u", _metrics(preemptions=8.0), states, prev, prev_p)
    assert grew["preemptions"] == 3.0
    reset = _compute_state("u", _metrics(preemptions=1.0), states, prev, prev_p)
    assert reset["preemptions"] == 0.0, "counter reset must not produce a negative delta"


def test_sleeping_flag_from_engine_sleep_state() -> None:
    states: State = {}
    prev: dict[str, tuple[float, int]] = {}
    prev_p: dict[str, float] = {}
    assert _compute_state("u", _metrics(sleep_state=1.0), states, prev, prev_p)["sleeping"] is True
    assert _compute_state("u", _metrics(sleep_state=0.0), states, prev, prev_p)["sleeping"] is False
    assert (
        _compute_state("u", _metrics(sleep_state=None), states, prev, prev_p)["sleeping"] is False
    )


# --- concurrency ------------------------------------------------------------


async def test_poll_metrics_queries_backends_concurrently(fleet: Fleet) -> None:
    """All backends are polled at once.

    The barrier only releases when *both* handlers have arrived, so a serial poll
    deadlocks and the test fails via its own timeout -- no wall-clock assertion.
    """
    barrier = asyncio.Barrier(2)

    async def gated(request: web.Request) -> web.StreamResponse:
        await barrier.wait()
        return web.Response(text="vllm:num_requests_running 0.0\n")

    await fleet.backend("a", gated)
    await fleet.backend("b", gated)
    a, b = fleet.url("a"), fleet.url("b")
    states: State = {a: None, b: None}
    config = Config(vllm_urls=[a, b], poll_interval=0.05, timeout=1)

    async with aiohttp.ClientSession() as session:
        task = asyncio.create_task(poll_metrics(config, states, session))
        try:
            await asyncio.wait_for(_both_seen(states, a, b), timeout=5)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _both_seen(states: State, a: str, b: str) -> None:
    while not (states.get(a) is not None and states.get(b) is not None):
        await asyncio.sleep(0.02)
