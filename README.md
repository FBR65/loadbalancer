# Load Balancer

An async load balancer for vLLM and llama.cpp (`llama-server`) inference endpoints, written in Python (aiohttp). It routes requests to the least-loaded endpoint of each model's group based on the endpoints' `/metrics` output — not on connection counts.

## Why

A single vLLM instance can return 504s under load. Running two instances of the same model behind a load balancer distributes the load. Instead of naive round-robin or `least_conn`, this balancer reads each instance's real load from vLLM's Prometheus `/metrics` endpoint and routes to the freest one.

## Architecture

```
Client → https://fqdn.de:443 → TLS-terminating reverse proxy (existing)
                                    → Load Balancer (Docker, aiohttp, async, port 8000)
                                        ├── vLLM instance 1 (VLLM_1_URL, model gpt-oss-120b)
                                        └── vLLM instance 2 (VLLM_2_URL, model gpt-oss-120b)
```

TLS (443) is handled by an existing upstream reverse proxy. The load balancer listens on port 8000 internally.

## Routing

The balancer polls `/metrics` on each instance and computes a load score:

```
score = num_requests_running + num_requests_waiting + (avg_queue_time_per_request × weight)
```

`avg_queue_time_per_request` is the delta of the cumulative
`request_queue_time_seconds_sum` divided by the delta of
`request_queue_time_seconds_count` since the previous poll — i.e. the mean
queue time of recently completed requests (the raw `_sum` is a monotonically
growing counter and would bias the score over time).

A new request goes to the instance with the lowest score. Instances that fail to report metrics are marked unhealthy and skipped.

### In-flight tracking and tie-breaking

The `/metrics` gauges lag by up to one poll interval, so the balancer also
tracks its own in-flight requests locally (incremented before forwarding,
released once the response is fully delivered). The effective running count is
`max(upstream, inflight)` — requests are never counted twice, and a burst of
concurrent requests is spread across the group instead of piling onto the
first instance while the metrics are still stale. Ties on the lowest score are
broken randomly.

### Model-based routing

If the request body contains a `model` field and that model is known from the
[model map](#model-map), the candidate pool is restricted to the endpoints that
host that model — the least-loaded one of *those* is picked. If all of a
model's endpoints are unhealthy, the balancer returns `503` (it never sends a
request to an instance that does not host the model). Requests without a model
field (e.g. `GET`s) or with an unknown model fall back to all healthy
instances.

### Saturation signals

Beyond the queue-time score, the balancer reads leading saturation indicators
from `/metrics` and reacts before queues build up:

- **`waiting` counts capacity waiters only** — the `num_requests_waiting` gauge
  is ambiguous in vLLM (it includes up to 256 requests queued on the GPU), so
  the load score uses only the `reason="capacity"` series of
  `num_requests_waiting_by_reason` when present (the remaining reasons —
  LoRA, KV budget, blocked requests — are summed into `waiting_deferred` and
  do **not** count as load). If `num_requests_waiting_by_reason` is missing,
  the balancer falls back to the total `num_requests_waiting`.
- **KV-cache saturation** (`kv_cache_usage_perc`) — an instance at/above
  `KV_CACHE_OVERLOAD_THRESHOLD` (default `0.95`) is treated as overloaded for
  retry purposes (rule 4).
- **Preemptions** (`num_preemptions` delta between polls) — recently
  preempting instances are deprioritized during candidate selection.
- **Engine sleep** (`engine_sleep_state`) — a sleeping engine is deprioritized.

### llama.cpp backends

`llama-server` endpoints are first-class backends (they are OpenAI-compatible,
so the proxy works unchanged). Their `/metrics` output maps onto the same load
signals: `llamacpp:requests_processing` → running, `llamacpp:requests_deferred`
(queued slot waiters) → waiting — plus the legacy `llamacpp_n_requests_running`
/ `llamacpp_n_requests_queued` names of older releases. Signals llama.cpp does
not export (queue time, KV-cache usage, preemptions, sleep state) stay absent;
a `llama-server` started **without** `--metrics` serves no `/metrics` and is
treated as unhealthy, like any endpoint that fails to report.

Malformed lines from known metrics are logged as warnings; they never render an
instance unhealthy on their own. Scientific notation (`1.2e1`) and missing
`{...}` label blocks are accepted (metrics from vLLM typically lack labels).

## Features

| Feature | Support |
|---|---|
| Chat Completions (`/v1/chat/completions`) | ✅ |
| Completions (`/v1/completions`) | ✅ |
| Responses API (`/v1/responses`, `/v1/responses/{id}`, `/v1/responses/{id}/cancel`) | ✅ |
| Tool calling | ✅ (generic catch-all passes `tools` payload through) |
| Streaming (SSE) | ✅ (chunk-by-chunk passthrough) |
| Non-streaming responses | ✅ |
| 5xx fallback to other instance | ✅ |
| 503 when no healthy instance | ✅ |
| Health endpoint (`/health`) | ✅ |

The proxy uses a generic catch-all route, so any path and HTTP method is forwarded to the selected instance.

## Retry behavior

On transient errors the balancer retries before giving up:

1. **Only transient errors are retried** — HTTP `500/502/503/504` and connection errors/timeouts. Client errors (`4xx`) are never retried.
2. **Bounded retries** — at most `MAX_RETRIES` (default 2) attempts with `RETRY_BACKOFF` (default 0.2s) between them.
3. **Prefer a fresh instance** — on retry, the balancer prefers an instance that has not been tried yet for this request, falling back to already-tried (or the same) instance only if no fresh healthy instance exists.
4. **No retry storm** — an instance whose waiting queue is at/above `OVERLOAD_THRESHOLD`, or whose KV-cache usage is at/above `KV_CACHE_OVERLOAD_THRESHOLD`, is not retried into.

If all retries are exhausted, the balancer returns `504`.

## Model map

At startup the balancer queries `/v1/models` on every configured endpoint and
builds the `model_id -> [endpoints]` mapping used for [model-based
routing](#model-based-routing): endpoints hosting the same model form a group,
and requests naming that model are distributed within the group by load. If
the endpoint list in `modelle.json` changes at runtime (config watchdog hot
reload), the mapping is re-fetched automatically on the next metrics poll.

The mapping is additionally written to a temporary file
(`vllm-lb-models-*.json` in the system temp directory, e.g. via `TMPDIR`).
The file is removed when the balancer shuts down (SIGINT/SIGTERM) and is
intended for debugging / external inspection only.

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `VLLM_1_URL` | `http://vllm1:8000` | Internal URL of vLLM instance 1 |
| `VLLM_2_URL` | `http://vllm2:8000` | Internal URL of vLLM instance 2 |
| `POLL_INTERVAL` | `2.0` | Seconds between `/metrics` polls |
| `TIMEOUT` | `300.0` | Upstream request timeout in seconds (total for regular requests; idle/`sock_read` for streamed responses, so slow but active streams are not cut off) |
| `LISTEN_PORT` | `8000` | Port the load balancer listens on |
| `QUEUE_TIME_WEIGHT` | `1.0` | Weight for `request_queue_time_seconds` in the score |
| `MAX_RETRIES` | `2` | Max retries on transient errors (502/503/504, connection errors/timeouts) |
| `RETRY_BACKOFF` | `0.2` | Backoff between retries in seconds |
| `OVERLOAD_THRESHOLD` | `10` | Do not retry into an instance whose waiting queue is at/above this threshold |
| `KV_CACHE_OVERLOAD_THRESHOLD` | `0.95` | Do not retry into an instance whose `kv_cache_usage_perc` is at/above this fraction (leading saturation signal; absolut `None` → signal ignored) |
| `MAX_BODY_SIZE` | `67108864` | Max request body size in bytes (default 64 MiB); larger bodies are rejected with `413` (also `max_body_size` in `einstellung.json`) |
| `SSL_CERT_FILE` | `certs/itzbund-ca.pem` | Extra CA bundle layered onto the system trust store for upstream TLS. The Docker image sets this to a merged bundle (system CAs + ITZBund internal CA) so internal backend certs verify while public ones still work. |

## Usage

### Local

```bash
uv sync
uv run python -m loadbalancer.main
```

### Docker

```bash
docker build -t vllm-lb .
docker run -p 8000:8000 --env-file .env vllm-lb
```

## Development

```bash
uv sync
uv run pytest -q                    # tests (33)
uv run pytest -q --cov=src/loadbalancer  # coverage
uv run ruff check .                 # lint
uv run ruff format --check .        # format
uv run mypy src/ tests/             # strict type check
uv run python scripts/mutants.py    # mutation testing (7 mutants, all must die)
uv run python scripts/smoke.py      # boots the real console script against real backends
```

## Project layout

```
src/loadbalancer/
├── config.py        # environment configuration
├── metrics.py       # /metrics parser (Prometheus format)
├── balancer.py      # load score + instance selection
├── proxy.py         # async reverse proxy + streaming
├── watchdog.py      # hot reload of einstellung.json / modelle.json
└── main.py          # metrics polling + server startup
scripts/
├── mutants.py       # manual mutation testing
└── smoke.py         # real-execution smoke test
```

## Known limits

- A response that breaks **after** the proxy already wrote bytes to the client
  (mid-SSE disconnect) is never retried — the client keeps a truncated stream
  rather than a second, spliced-in answer from another instance.
- Repeated upstream headers (e.g. several `Set-Cookie`) collapse to the last
  one when relayed; vLLM does not use them.
- Hot reload applies to routing/scoring settings; `listen_port` and
  `max_body_size` are read at startup and need a restart.
- `GET /metrics` is proxied to a backend; the balancer exposes no metrics of
  its own, and `GET /health` reports `ok` even when every backend is down.

## License

MIT
