"""Property layer: invariants that must hold for arbitrary header/state sets."""

import random
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient

from conftest import Fleet
from vllm_lb.balancer import pick
from vllm_lb.proxy import preferred_candidates

SEED = 20260928
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
ENTITY = frozenset({"content-encoding", "content-length"})
HEADER_POOL = sorted(HOP_BY_HOP | ENTITY | {"x-trace", "x-request-id", "content-type"})
# headers the web server itself manages or rejects
UNSAFE_TO_SET = frozenset({"transfer-encoding", "content-length"})


def _random_header_sets(n: int) -> list[dict[str, str]]:
    rng = random.Random(SEED)
    sets: list[dict[str, str]] = []
    for _ in range(n):
        size = rng.randint(1, 6)
        names = rng.sample(HEADER_POOL, size)
        chosen = [nm for nm in names if nm not in UNSAFE_TO_SET]
        if not chosen:
            continue
        sets.append({nm: "x" for nm in chosen})
    return sets


async def test_no_request_header_ever_leaks_through_the_proxy(fleet: Fleet) -> None:
    """For arbitrary upstream header sets, none of them is forbidden downstream."""
    sets = _random_header_sets(40)
    assert len(sets) >= 30

    for headers in sets:
        async def handler(request: web.Request, h: dict[str, str] = headers) -> web.StreamResponse:
            return web.Response(body=b"{}", headers=h | {"content-type": "application/json"})

        name = f"b{headers!r}"
        await fleet.backend(name, handler)
        lb = await fleet.start_lb([name], states=fleet.states(**{name: {}}))

        async with TestClient(lb) as client:
            response = await client.get("/v1/chat/completions")
            await response.read()

        leaked = {
            key
            for key in response.headers
            if key.lower() in HOP_BY_HOP | ENTITY and key.lower() != "content-length"
        }
        assert leaked == set(), f"leaked {sorted(leaked)} for upstream headers {headers}"
        assert response.headers.get("content-length") == "2", "wrong body length forwarded"

        await fleet.lb.close()  # type: ignore[union-attr]
        fleet.lb = None


def test_candidate_selection_never_invents_or_drops_instances() -> None:
    """preferred_candidates: a subset of the healthy set, empty only if that is empty."""
    rng = random.Random(SEED)
    for _ in range(2000):
        size = rng.randint(0, 6)
        names = [f"u{i}" for i in range(size)]
        states: dict[str, Any] = {}
        for name in names:
            roll = rng.random()
            if roll < 0.2:
                states[name] = None
            else:
                states[name] = {
                    "running": rng.uniform(0, 10),
                    "waiting": rng.uniform(0, 10),
                    "queue_time_sum": rng.uniform(0, 10),
                    "sleeping": rng.random() < 0.3,
                    "preemptions": rng.choice([0, 0, 0, 1, 2]),
                }
        healthy = {u for u, s in states.items() if s is not None}
        tried = rng.sample(names, rng.randint(0, len(names)))

        result = preferred_candidates(states, tried)  # type: ignore[arg-type]

        assert set(result) <= healthy, "returned an unhealthy instance"
        assert bool(result) == bool(healthy), "empty result while instances are healthy"


def test_pick_only_returns_offered_candidates() -> None:
    rng = random.Random(SEED)
    for _ in range(2000):
        size = rng.randint(0, 6)
        states = {
            f"u{i}": (None if rng.random() < 0.2 else
                      {"running": rng.uniform(0, 5), "waiting": 0.0, "queue_time_sum": 0.0})
            for i in range(size)
        }
        chosen = pick(states, 1.0, rng=random.Random(rng.random()))  # type: ignore[arg-type]
        if chosen is None:
            assert all(s is None for s in states.values())
        else:
            assert chosen in states and states[chosen] is not None
