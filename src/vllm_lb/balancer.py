# src/vllm_lb/balancer.py
import random
from collections.abc import Mapping

MetricsState = dict[str, int | float]


def score(running: float, waiting: float, queue_time_sum: float, weight: float = 1.0) -> float:
    return running + waiting + (queue_time_sum * weight)


def pick(
    states: Mapping[str, MetricsState | None],
    weight: float = 1.0,
    exclude: str | None = None,
    inflight: Mapping[str, int] | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Least-loaded healthy instance.

    `inflight` (locally in-flight requests per instance) covers the gap until
    the next /metrics poll, so bursts are spread instead of piling up on the
    first instance. The effective running count is max(upstream, inflight) so
    requests are never counted twice. Ties on the lowest score are broken
    randomly.
    """
    inflight = inflight if inflight is not None else {}
    best_score = None
    tied: list[str] = []
    for url, st in states.items():
        if url == exclude:
            continue
        if st is None:
            continue
        running = max(st["running"], inflight.get(url, 0))
        s = score(running, st["waiting"], st["queue_time_sum"], weight)
        if best_score is None or s < best_score:
            best_score, tied = s, [url]
        elif s == best_score:
            tied.append(url)
    if not tied:
        return None
    if rng is None:
        return random.choice(tied)
    return rng.choice(tied)
