# src/loadbalancer/main.py
import asyncio
import json
import logging
import os
import signal
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from loadbalancer.config import Config, load_config, load_dotenv
from loadbalancer.metrics import MetricsResult, parse_metrics
from loadbalancer.proxy import build_app
from loadbalancer.watchdog import ConfigWatcher

State = dict[str, dict[str, int | float | bool | None] | None]

logger = logging.getLogger("loadbalancer")


def metrics_urls(base: str) -> list[str]:
    """Candidate /metrics URLs for a base URL.

    vLLM serves /metrics at the server root. If the configured base URL carries a
    path (e.g. .../v1), the root is added as a fallback candidate.
    """
    base = base.rstrip("/")
    urls = [f"{base}/metrics"]
    parsed = urlsplit(base)
    if parsed.path not in ("", "/") and parsed.scheme and parsed.netloc:
        urls.append(f"{parsed.scheme}://{parsed.netloc}/metrics")
    return urls


async def _fetch_one(
    session: aiohttp.ClientSession, url: str
) -> tuple[MetricsResult | None, str | None]:
    """(metrics, error) for one instance, trying its /metrics URL candidates.

    Never raises: an unreachable instance yields (None, reason) so one bad
    backend cannot abort the whole poll cycle.
    """
    last_error: str | None = None
    for metrics_url in metrics_urls(url):
        try:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return parse_metrics(await resp.text(), on_malformed=_log_malformed), None
                last_error = f"status {resp.status} ({metrics_url})"
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            last_error = f"{type(exc).__name__} ({metrics_url})"
    return None, last_error


def _compute_state(
    url: str,
    metrics: MetricsResult,
    states: State,
    prev: dict[str, tuple[float, int]],
    prev_preemptions: dict[str, float],
) -> dict[str, int | float | bool | None]:
    """Derive this instance's load state from a metrics sample.

    Counter-based values (queue time, preemptions) are turned into per-poll
    *deltas* against the previous sample, so the score reflects recent load and
    a counter reset (vLLM restart) yields 0 instead of a negative rate.
    """
    p = prev.get(url)
    if p is not None and metrics["queue_time_count"] > p[1]:
        if metrics["queue_time_sum"] < p[0]:
            rate = 0.0  # counter reset (e.g. vLLM restart)
        else:
            rate = (metrics["queue_time_sum"] - p[0]) / (metrics["queue_time_count"] - p[1])
    else:
        rate = 0.0
    prev[url] = (metrics["queue_time_sum"], metrics["queue_time_count"])

    # Effective load: only capacity-waiting counts as a load signal;
    # deferred waiters (LoRA/KV budget/blocked) are not real load.
    reasons = metrics["waiting_by_reason"]
    if reasons:
        capacity = reasons.get("capacity", 0.0)
        deferred = sum(v for k, v in reasons.items() if k != "capacity")
    else:
        capacity = metrics["waiting"]
        deferred = 0.0

    raw_preemptions = metrics["preemptions"]
    baseline = prev_preemptions.get(url)
    preemption_delta = 0.0
    if baseline is not None and raw_preemptions >= baseline:
        preemption_delta = raw_preemptions - baseline
    prev_preemptions[url] = raw_preemptions

    sleep_state = metrics["sleep_state"]
    return {
        "running": metrics["running"],
        "waiting": capacity,
        "waiting_deferred": deferred,
        "queue_time_sum": rate,
        "kv_cache_usage_perc": metrics["kv_cache_usage_perc"],
        "preemptions": preemption_delta,
        "sleeping": sleep_state is not None and sleep_state > 0,
    }


async def poll_metrics(
    config: Config,
    states: State,
    session: aiohttp.ClientSession,
    model_map: dict[str, list[str]] | None = None,
) -> None:
    # Per-instance counter baselines for delta-rate computation.
    prev: dict[str, tuple[float, int]] = {}
    prev_preemptions: dict[str, float] = {}
    # The startup model map (fetched by main) already covers the current URL
    # set; re-fetch it whenever the configured URLs change (hot reload of
    # modelle.json via the config watchdog).
    seen_urls = tuple(config.vllm_urls)
    while True:
        urls = tuple(config.vllm_urls)
        if urls != seen_urls:
            seen_urls = urls
            # Drop instances that left the fleet: `states` is the proxy's
            # routing pool, so a stale key would keep routing to a removed
            # backend forever.
            for gone in set(states) - set(urls):
                del states[gone]
                prev.pop(gone, None)
                prev_preemptions.pop(gone, None)
                logger.info("instance removed: %s", gone)
            if model_map is not None:
                new_map = await fetch_model_map(config, session)
                model_map.clear()
                model_map.update(new_map)
                logger.info(
                    "model map refreshed: %d model(s), %d endpoint(s)", len(new_map), len(urls)
                )
        # All instances are polled concurrently: a slow backend must not delay
        # the others (serially, 9 unreachable instances would stretch one
        # 2s poll cycle to 45s). State computation stays sequential below, on
        # this single event-loop thread, because the counter baselines are
        # order-dependent.
        fetched = await asyncio.gather(*(_fetch_one(session, url) for url in urls))
        for url, (metrics, last_error) in zip(urls, fetched):
            state: dict[str, int | float | bool | None] | None = None
            if metrics is not None:
                state = _compute_state(url, metrics, states, prev, prev_preemptions)
            if state is None and states.get(url) is not None:
                logger.warning("instance unhealthy: %s (%s)", url, last_error)
                prev.pop(url, None)
                prev_preemptions.pop(url, None)
            elif state is not None and states.get(url) is None:
                logger.info("instance healthy: %s", url)
            states[url] = state
        await asyncio.sleep(config.poll_interval)


def _log_malformed(line: str) -> None:
    logger.warning("malformed metric line: %s", line)


async def fetch_model_map(config: Config, session: aiohttp.ClientSession) -> dict[str, list[str]]:
    """Query /v1/models on every configured endpoint and map model id -> endpoints."""
    mapping: dict[str, list[str]] = {}
    for url in config.vllm_urls:
        try:
            async with session.get(
                f"{url.rstrip('/')}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    logger.warning("model fetch failed: %s (status %s)", url, resp.status)
                    continue
                data = await resp.json()
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            logger.warning("model fetch failed: %s (%s)", url, type(exc).__name__)
            continue
        for item in data.get("data", []):
            model_id = item.get("id")
            if model_id:
                mapping.setdefault(str(model_id), []).append(url)
    return mapping


def write_model_map(mapping: dict[str, list[str]]) -> str:
    """Persist the model -> endpoint mapping to a temp file; returns its path."""
    fd, path = tempfile.mkstemp(prefix="vllm-lb-models-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("model map written to %s", path)
    return path


async def rebind_port(runner: web.AppRunner, current: web.TCPSite, wanted: int) -> web.TCPSite:
    """Move the listener to `wanted`, keeping the old one until it works.

    The new socket is bound *before* the old one is closed: a config typo (a
    privileged port, a port already in use) must not take the balancer down.
    Requests already in flight on the old port are dropped, which is the same
    consequence a restart would have.
    """
    site = web.TCPSite(runner, "0.0.0.0", wanted)
    try:
        await site.start()
    except OSError as exc:
        logger.error("cannot listen on port %s (%s); keeping the current port", wanted, exc)
        return current
    await current.stop()
    logger.info("now listening on port %s", wanted)
    return site


async def follow_listen_port(runner: web.AppRunner, config: Config, site: web.TCPSite) -> None:
    """Rebind whenever `listen_port` changes in the config files."""
    current = site
    port = config.listen_port
    while True:
        await asyncio.sleep(1.0)
        if config.listen_port == port:
            continue
        current = await rebind_port(runner, current, config.listen_port)
        port = config.listen_port


async def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config_dir = Path(
        os.getenv("CONFIG_DIR", str(Path(__file__).resolve().parent.parent.parent / "config"))
    )
    config = load_config(config_dir)
    logger.info("starting load balancer on port %s", config.listen_port)
    for url in config.vllm_urls:
        logger.info("backend: %s (metrics: %s)", url, ", ".join(metrics_urls(url)))
    states: State = {url: None for url in config.vllm_urls}
    model_map_path: str | None = None
    watcher = ConfigWatcher(config, config_dir)
    watcher.start()
    async with aiohttp.ClientSession() as session:
        model_map = await fetch_model_map(config, session)
        if model_map:
            model_map_path = write_model_map(model_map)
        else:
            logger.warning("no models discovered; skipping model map file")
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        asyncio.create_task(poll_metrics(config, states, session, model_map))
        app = build_app(config, states, model_map)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", config.listen_port)
        await site.start()
        asyncio.create_task(follow_listen_port(runner, config, site))
        await stop.wait()
        watcher.stop()
        await runner.cleanup()
        if model_map_path is not None and os.path.exists(model_map_path):
            os.unlink(model_map_path)
            logger.info("model map removed: %s", model_map_path)


def run() -> None:
    """Console-script entry point; `main` is a coroutine and cannot be one."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
