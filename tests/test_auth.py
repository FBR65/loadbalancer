"""SC-3: the balancer authenticates clients when a token is configured."""

import logging
from collections.abc import Awaitable, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient
from conftest import Fleet

from loadbalancer.config import Config

TOKEN = "s3cr3t-token-value"
PROBED = ("/", "/health", "/ready", "/metrics")


async def _ok(_request: web.Request) -> web.StreamResponse:
    return web.json_response({"ok": True})


async def _echo(request: web.Request) -> web.StreamResponse:
    return web.json_response(
        {"auth": request.headers.get("authorization"), "lb": request.headers.get("x-auth-token")}
    )


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


async def _authed_lb(
    fleet: Fleet, handler: Handler = _ok
) -> TestClient[web.Request, web.Application]:
    await fleet.backend("a", handler)
    config = Config(
        vllm_urls=[fleet.url("a")],
        max_retries=0,
        retry_backoff=0.01,
        timeout=5,
        auth_token=TOKEN,
    )
    lb = await fleet.start_lb(["a"], config=config, states=fleet.states(a={}))
    return TestClient(lb)


async def test_a_request_without_a_token_is_rejected(fleet: Fleet) -> None:
    client = await _authed_lb(fleet)
    async with client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})

    assert response.status == 401
    assert fleet.hits["a"] == [], "an unauthenticated request must not reach a backend"


async def test_a_valid_bearer_token_is_accepted(fleet: Fleet) -> None:
    client = await _authed_lb(fleet)
    async with client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status == 200


async def test_a_valid_x_auth_token_is_accepted(fleet: Fleet) -> None:
    client = await _authed_lb(fleet)
    async with client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m"}, headers={"X-Auth-Token": TOKEN}
        )

    assert response.status == 200


async def test_a_wrong_token_is_rejected(fleet: Fleet) -> None:
    client = await _authed_lb(fleet)
    async with client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m"}, headers={"X-Auth-Token": "wrong"}
        )

    assert response.status == 401


async def test_a_prefix_of_the_token_is_rejected(fleet: Fleet) -> None:
    """A prefix must not pass; a length- or prefix-leak would be a real bypass."""
    client = await _authed_lb(fleet)
    async with client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m"},
            headers={"X-Auth-Token": TOKEN[:-1]},
        )

    assert response.status == 401


async def test_operational_endpoints_stay_reachable_without_a_token(fleet: Fleet) -> None:
    """Probes must work while the balancer itself is unhealthy."""
    client = await _authed_lb(fleet)
    async with client:
        statuses = {path: (await client.get(path)).status for path in PROBED}

    assert statuses == dict.fromkeys(PROBED, 200), statuses


async def _models(_request: web.Request) -> web.StreamResponse:
    return web.json_response({"object": "list", "data": [{"id": "m"}]})


async def test_model_listing_requires_the_token(fleet: Fleet) -> None:
    client = await _authed_lb(fleet, _models)
    async with client:
        assert (await client.get("/v1/models")).status == 401
        ok = await client.get("/v1/models", headers={"X-Auth-Token": TOKEN})

    assert ok.status == 200


async def test_the_balancers_own_token_is_not_forwarded_to_the_backend(fleet: Fleet) -> None:
    """The balancer credential is not a backend credential."""
    client = await _authed_lb(fleet, _echo)
    async with client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m"},
            headers={"X-Auth-Token": TOKEN, "Authorization": f"Bearer {TOKEN}"},
        )
        relayed = await response.json()

    assert relayed["lb"] is None, "the balancer's own credential must be stripped upstream"


async def test_without_a_token_the_balancer_stays_open_and_warns(
    fleet: Fleet, caplog: pytest.LogCaptureFixture
) -> None:
    await fleet.backend("a", _ok)
    config = Config(vllm_urls=[fleet.url("a")], max_retries=0, retry_backoff=0.01, timeout=5)
    lb = await fleet.start_lb(["a"], config=config, states=fleet.states(a={}))

    with caplog.at_level(logging.WARNING):
        async with TestClient(lb) as client:
            response = await client.post("/v1/chat/completions", json={"model": "m"})

    assert response.status == 200, "an unconfigured token must not break existing clients"
    assert any("auth" in record.message.lower() for record in caplog.records)


async def test_an_empty_token_configured_is_treated_as_no_token(fleet: Fleet) -> None:
    """An empty value must not become a token that "" satisfies."""
    await fleet.backend("a", _ok)
    config = Config(
        vllm_urls=[fleet.url("a")], max_retries=0, retry_backoff=0.01, timeout=5, auth_token=""
    )
    lb = await fleet.start_lb(["a"], config=config, states=fleet.states(a={}))

    async with TestClient(lb) as client:
        response = await client.post("/v1/chat/completions", json={"model": "m"})

    assert response.status == 200
