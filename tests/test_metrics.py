"""Prometheus /metrics parsing: the signals the balancer actually relies on."""

from loadbalancer.metrics import MetricsResult, parse_metrics


def test_parses_core_vllm_gauges() -> None:
    r = parse_metrics(
        "vllm:num_requests_running 3\n"
        "vllm:num_requests_waiting 2\n"
        "vllm:request_queue_time_seconds_sum 12.5\n"
        "vllm:request_queue_time_seconds_count 5\n"
        "vllm:kv_cache_usage_perc 0.4\n"
        "vllm:num_preemptions_total 7\n"
        "vllm:engine_sleep_state 1\n"
    )
    assert r["running"] == 3.0
    assert r["waiting"] == 2.0
    assert r["queue_time_sum"] == 12.5
    assert r["queue_time_count"] == 5
    assert r["kv_cache_usage_perc"] == 0.4
    assert r["preemptions"] == 7.0
    assert r["sleep_state"] == 1.0


def test_sums_running_across_label_series() -> None:
    r = parse_metrics(
        'vllm:num_requests_running{engine="0"} 2\nvllm:num_requests_running{engine="1"} 3\n'
    )
    assert r["running"] == 5.0


def test_kv_cache_uses_max_not_sum() -> None:
    """Summing two engines' fractional usage would falsely trip the 0.95 threshold."""
    r = parse_metrics(
        'vllm:kv_cache_usage_perc{engine="0"} 0.5\nvllm:kv_cache_usage_perc{engine="1"} 0.6\n'
    )
    assert r["kv_cache_usage_perc"] == 0.6


def test_waiting_by_reason() -> None:
    r = parse_metrics(
        'vllm:num_requests_waiting_by_reason{reason="capacity"} 4\n'
        'vllm:num_requests_waiting_by_reason{reason="engine_busy"} 1\n'
    )
    assert r["waiting_by_reason"] == {"capacity": 4.0, "engine_busy": 1.0}


def test_bare_preemption_name_is_accepted() -> None:
    assert parse_metrics("vllm:num_preemptions 2\n")["preemptions"] == 2.0
    assert parse_metrics("vllm:num_preemptions_total 2\n")["preemptions"] == 2.0


def test_created_timestamp_lines_are_ignored() -> None:
    r = parse_metrics("vllm:num_preemptions_total_created 1.2e9\nvllm:num_preemptions_total 4\n")
    assert r["preemptions"] == 4.0


def test_comments_and_blank_lines_are_skipped() -> None:
    r = parse_metrics("# HELP vllm:num_requests_running x\n\nvllm:num_requests_running 1\n")
    assert r["running"] == 1.0


def test_scientific_notation_is_accepted() -> None:
    assert parse_metrics("vllm:num_requests_running 1.2e1\n")["running"] == 12.0


def test_llamacpp_modern_names() -> None:
    r = parse_metrics("llamacpp:requests_processing 2\nllamacpp:requests_deferred 3\n")
    assert r["running"] == 2.0
    assert r["waiting"] == 3.0


def test_llamacpp_legacy_names() -> None:
    r = parse_metrics("llamacpp_n_requests_running 5\nllamacpp_n_requests_queued 6\n")
    assert r["running"] == 5.0
    assert r["waiting"] == 6.0


def test_malformed_known_metric_is_reported_but_keeps_default() -> None:
    seen: list[str] = []
    r = parse_metrics("vllm:num_requests_running abc\n", on_malformed=seen.append)
    assert seen == ["vllm:num_requests_running abc"]
    assert r["running"] == 0.0


def test_unknown_metric_is_not_reported_as_malformed() -> None:
    seen: list[str] = []
    parse_metrics(
        "some_other:metric 1\nvllm:prefix_matching_but_unknown 2\n", on_malformed=seen.append
    )
    assert seen == []


def test_empty_input_returns_defaults() -> None:
    r: MetricsResult = parse_metrics("")
    assert r == {
        "running": 0.0,
        "waiting": 0.0,
        "queue_time_sum": 0.0,
        "queue_time_count": 0,
        "waiting_by_reason": {},
        "kv_cache_usage_perc": None,
        "preemptions": 0.0,
        "sleep_state": None,
    }
