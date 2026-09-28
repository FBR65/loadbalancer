# src/loadbalancer/telemetry.py
"""The balancer's own counters, exposed on `GET /metrics`.

A load balancer that relays `/metrics` to a backend gives an operator no way
to see the balancer itself. These counters are the minimum needed to tell
"the balancer is slow" from "the backends are slow".

Plain integers are safe here: aiohttp runs handlers on one event loop and
`+= 1` contains no await, so a counter cannot be lost to a context switch.
"""

from collections.abc import Mapping

State = dict[str, dict[str, int | float | bool | None] | None]


def label_value(value: str) -> str:
    """A metric label value, safe to embed in quotes.

    Endpoints come from configuration, so a value containing a quote or a
    backslash would otherwise emit a second label and silently redefine every
    series after it.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape(value: str) -> str:
    """An error message, safe to embed in a quoted label."""
    return label_value(value)


class Telemetry:
    """Counters and gauges rendered in Prometheus text exposition format."""

    def __init__(self) -> None:
        self.requests_total = 0
        self.retries_total = 0
        self.upstream_errors_total = 0
        self.stream_errors_total = 0

    def render(self, states: Mapping[str, object], inflight: Mapping[str, int]) -> str:
        """The exposition body for the current `states` / in-flight counts."""
        healthy = sum(1 for value in states.values() if value is not None)
        lines: list[str] = []
        for name, value in (
            ("loadbalancer_requests_total", self.requests_total),
            ("loadbalancer_retries_total", self.retries_total),
            ("loadbalancer_upstream_errors_total", self.upstream_errors_total),
            ("loadbalancer_stream_errors_total", self.stream_errors_total),
        ):
            lines.append(f"# HELP {name} Requests observed by the load balancer.")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        lines.append(
            "# HELP loadbalancer_endpoints_healthy Configured endpoints reporting metrics."
        )
        lines.append("# TYPE loadbalancer_endpoints_healthy gauge")
        lines.append(f"loadbalancer_endpoints_healthy {healthy}")

        lines.append("# HELP loadbalancer_endpoint_healthy 1 if the endpoint reported metrics.")
        lines.append("# TYPE loadbalancer_endpoint_healthy gauge")
        for endpoint in sorted(states):
            up = 1 if states[endpoint] is not None else 0
            lines.append(
                f'loadbalancer_endpoint_healthy{{endpoint="{label_value(endpoint)}"}} {up}'
            )

        lines.append("# HELP loadbalancer_inflight_requests Requests the balancer is proxying now.")
        lines.append("# TYPE loadbalancer_inflight_requests gauge")
        for endpoint in sorted(states):
            lines.append(
                f'loadbalancer_inflight_requests{{endpoint="{label_value(endpoint)}"}} '
                f"{inflight.get(endpoint, 0)}"
            )
        return "\n".join(lines) + "\n"


__all__ = ["State", "Telemetry", "label_value"]
