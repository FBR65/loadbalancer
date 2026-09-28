# src/loadbalancer/metrics.py
import re
from collections.abc import Callable
from typing import Any, TypedDict, cast

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


class MetricsResult(TypedDict):
    running: float
    waiting: float
    queue_time_sum: float
    queue_time_count: int
    waiting_by_reason: dict[str, float]
    kv_cache_usage_perc: float | None
    preemptions: float
    sleep_state: float | None


def _series(prefix: str) -> re.Pattern[str]:
    """Match `prefix` with optional `{...}` labels and a numeric value."""
    return re.compile(rf"^{prefix}(?:{{.*}})?\s+({_NUM})$")


_REASON = re.compile(
    r'^vllm:num_requests_waiting_by_reason\{.*?reason="([^"]+)".*\}\s+(' + _NUM + r")$"
)

# OpenMetrics counters are exported as `<name>_total` (+ a `<name>_created`
# timestamp line); older vLLM versions use the bare name.
_PREEMPTIONS = re.compile(rf"^vllm:num_preemptions(?:_total)?(?:{{.*}})?\s+({_NUM})$")

_KNOWN = re.compile(
    r"^(vllm:num_requests_running|vllm:num_requests_waiting|"
    r"vllm:num_requests_waiting_by_reason|vllm:kv_cache_usage_perc|"
    r"vllm:num_preemptions|vllm:engine_sleep_state|"
    r"vllm:request_queue_time_seconds_sum|vllm:request_queue_time_seconds_count|"
    r"llamacpp:requests_processing|llamacpp:requests_deferred|"
    r"llamacpp_n_requests_running|llamacpp_n_requests_queued)(?=[\s{])"
)


def _add(key: str) -> Callable[[dict[str, Any], re.Match[str]], None]:
    def handler(results: dict[str, Any], m: re.Match[str]) -> None:
        results[key] += float(m.group(1))

    return handler


def _add_count(results: dict[str, Any], m: re.Match[str]) -> None:
    results["queue_time_count"] += int(float(m.group(1)))


def _max(results: dict[str, Any], m: re.Match[str]) -> None:
    value = float(m.group(1))
    current = results["kv_cache_usage_perc"]
    if current is None or value > current:
        results["kv_cache_usage_perc"] = value


def _set(results: dict[str, Any], m: re.Match[str]) -> None:
    results["sleep_state"] = float(m.group(1))


def _add_reason(results: dict[str, Any], m: re.Match[str]) -> None:
    reason = m.group(1)
    results["waiting_by_reason"][reason] = results["waiting_by_reason"].get(reason, 0.0) + float(
        m.group(2)
    )


_PARSERS: list[tuple[re.Pattern[str], Callable[[dict[str, Any], re.Match[str]], None]]] = [
    (_series("vllm:num_requests_running"), _add("running")),
    (_series("vllm:num_requests_waiting"), _add("waiting")),
    (_REASON, _add_reason),
    (_series("vllm:kv_cache_usage_perc"), _max),
    (_PREEMPTIONS, _add("preemptions")),
    (_series("vllm:engine_sleep_state"), _set),
    (_series("vllm:request_queue_time_seconds_sum"), _add("queue_time_sum")),
    (_series("vllm:request_queue_time_seconds_count"), _add_count),
    # llama.cpp (llama-server): current `llamacpp:` names plus the legacy
    # `llamacpp_` names of older releases. Deferred requests are queued (slot
    # waiters) and count as load, like capacity waiters in vLLM.
    (_series("llamacpp:requests_processing"), _add("running")),
    (_series("llamacpp:requests_deferred"), _add("waiting")),
    (_series("llamacpp_n_requests_running"), _add("running")),
    (_series("llamacpp_n_requests_queued"), _add("waiting")),
]


def parse_metrics(text: str, on_malformed: Callable[[str], None] | None = None) -> MetricsResult:
    """Parse vLLM /metrics output. Known metrics whose value cannot be parsed
    are reported via `on_malformed` and keep their default (they never render
    an instance unhealthy by themselves)."""
    results: MetricsResult = {
        "running": 0.0,
        "waiting": 0.0,
        "queue_time_sum": 0.0,
        "queue_time_count": 0,
        "waiting_by_reason": {},
        "kv_cache_usage_perc": None,
        "preemptions": 0.0,
        "sleep_state": None,
    }
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        matched = False
        for pattern, handler in _PARSERS:
            m = pattern.fullmatch(line)
            if m:
                handler(cast(dict[str, Any], results), m)
                matched = True
                break
        if not matched and on_malformed is not None and _KNOWN.match(line):
            on_malformed(line)
    return results
