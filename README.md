# Load Balancer

An async load balancer for vLLM and llama.cpp (`llama-server`) inference endpoints, written in Python (aiohttp). It groups endpoints by the models they serve, reads each endpoint's real load from its Prometheus `/metrics` output, and routes every request to the least-loaded endpoint of that model's group — not by round-robin or connection count.

## Why

A single vLLM instance returns 504s under load. Running several instances of the same model behind a load balancer distributes that load. The balancer does not guess: it polls each endpoint's `/metrics` and routes to the one with the lowest real load, and it retries elsewhere when an endpoint is saturated or fails.

## Architecture

```
Client → https://fqdn.de:443 → TLS-terminating reverse proxy (existing)
                                    → Load Balancer (aiohttp, async, port 8000)
                                        ├── endpoint A  (gpt-oss-120b)
                                        ├── endpoint B  (gpt-oss-120b)
                                        └── endpoint C  (paddle, llama-server)
```

TLS (443) terminates at an existing upstream reverse proxy; the balancer listens on port 8000 internally. The fleet is not fixed to two instances — `modelle.json` lists any number of endpoints, and endpoints serving the same model form a routing group (see [Model-based routing](#model-based-routing)).

## Routing

The balancer polls `/metrics` on each endpoint and computes a load score:

```
score = running + waiting + (avg_queue_time_per_request × QUEUE_TIME_WEIGHT)
```

`avg_queue_time_per_request` is the delta of the cumulative `request_queue_time_seconds_sum` divided by the delta of `request_queue_time_seconds_count` since the previous poll — the mean queue time of recently completed requests. The raw `_sum` is a monotonically growing counter and would bias the score over time; a counter reset (e.g. a vLLM restart) yields `0` rather than a negative rate.

A new request goes to the endpoint with the lowest score. Endpoints that fail to report metrics are marked unhealthy and skipped.

All endpoints are polled **concurrently** each cycle, so one slow or unreachable backend never delays the others (serially, N unreachable endpoints would stretch a single poll interval to N × timeout). When a URL is removed from `modelle.json` at runtime, it also leaves the routing pool on the next poll.

### In-flight tracking and tie-breaking

The `/metrics` gauges lag by up to one poll interval, so the balancer also tracks its own in-flight requests locally (incremented before forwarding, released once the response is fully delivered). The effective running count is `max(upstream, inflight)` — requests are never counted twice, and a burst of concurrent requests is spread across the group instead of piling onto the first endpoint while the metrics are still stale. Ties on the lowest score are broken randomly.

### Model-based routing

If the request body contains a `model` field and that model is known from the [model map](#model-map), the candidate pool is restricted to the endpoints that host that model — the least-loaded one of *those* is picked. If every endpoint of a model is unhealthy, the balancer returns `503` (it never sends a request to an endpoint that does not host the model). Requests without a model field (e.g. `GET`s) or with an unknown model fall back to all healthy endpoints.

### Saturation signals

Beyond the queue-time score, the balancer reads leading saturation indicators from `/metrics` and reacts before queues build up:

- **`waiting` counts capacity waiters only** — the `num_requests_waiting` gauge is ambiguous in vLLM (it includes requests queued on the GPU), so the score uses only the `reason="capacity"` series of `num_requests_waiting_by_reason` when present. The remaining reasons (LoRA, KV budget, blocked requests) are summed into `waiting_deferred` and do **not** count as load. If `num_requests_waiting_by_reason` is missing, the balancer falls back to the total `num_requests_waiting`.
- **KV-cache saturation** (`kv_cache_usage_perc`) — an endpoint at/above `KV_CACHE_OVERLOAD_THRESHOLD` (default `0.95`) counts as overloaded for retry purposes (rule 4). The gauge is reduced with `max`, not `sum`, so several engines cannot push the value past the threshold artificially.
- **Preemptions** (`num_preemptions` delta between polls) — recently preempting endpoints are deprioritized during candidate selection.
- **Engine sleep** (`engine_sleep_state`) — a sleeping engine is deprioritized.

### llama.cpp backends

`llama-server` endpoints are first-class backends (they are OpenAI-compatible, so the proxy works unchanged). Their `/metrics` output maps onto the same load signals: `llamacpp:requests_processing` → running, `llamacpp:requests_deferred` (queued slot waiters) → waiting — plus the legacy `llamacpp_n_requests_running` / `llamacpp_n_requests_queued` names of older releases. Signals llama.cpp does not export (queue time, KV-cache usage, preemptions, sleep state) stay absent. A `llama-server` started **without** `--metrics` serves no `/metrics` and is treated as unhealthy, like any endpoint that fails to report.

Malformed lines from known metrics are logged as warnings; they never render an endpoint unhealthy on their own. Scientific notation (`1.2e1`) and missing `{...}` label blocks are accepted (metrics from vLLM typically lack labels).

## Endpoints

The balancer answers these itself:

| Endpoint | Behaviour |
|---|---|
| `GET /` | Service banner and the list of API routes it serves (not the configured backends) |
| `GET /health` | Liveness — `ok` whenever the process is alive, even with every backend down. Used by the container `HEALTHCHECK` |
| `GET /ready` | Readiness — `503` while no backend reports metrics, so a balancer with nothing to serve is taken out of rotation |
| `GET /metrics` | The balancer's **own** Prometheus counters (no longer relayed upstream) |
| `GET /v1/models` | Live aggregate of `/v1/models` from all backends; `503` if none answers |

Every other path and method is forwarded. `/v1/chat/completions`, `/v1/completions`, `/v1/responses` (incl. `/{id}` and `/{id}/cancel`), tool calling, and streaming (SSE) therefore work unchanged; transient `500/502/503/504` responses fall back to another endpoint of the same group. `/v1/models` is served here rather than forwarded, but aggregates the backends live.

## Retry behavior

On transient errors the balancer retries before giving up:

1. **Only transient errors are retried** — HTTP `500/502/503/504` and connection errors/timeouts. Client errors (`4xx`) are never retried.
2. **Bounded retries** — `MAX_RETRIES` (default 2) retries *after* the first attempt, i.e. up to 3 upstream requests, with `RETRY_BACKOFF` (default 0.2s) between them.
3. **Prefer a fresh endpoint** — on retry, the balancer prefers one that has not been tried yet for this request, falling back to an already-tried (or the same) endpoint only if no fresh healthy one exists.
4. **No retry storm** — an endpoint whose waiting queue is at/above `OVERLOAD_THRESHOLD`, or whose KV-cache usage is at/above `KV_CACHE_OVERLOAD_THRESHOLD`, is not retried into.
5. **Never after a commit** — once any byte of the response has reached the client, there is no retry. A second attempt could only append a different answer to a half-delivered body and would pay for the inference twice.

A stream that breaks mid-body is therefore **reported, never silently truncated**: an SSE client receives a terminal `event: error` naming the failure, and the body is left incomplete, so a raw client cannot mistake a cut-off answer for a finished one. A stream that completes normally is closed cleanly and is distinguishable from a broken one.

If all retries are exhausted, the balancer returns `504` naming the failing endpoint.

## Model map

At startup the balancer queries `/v1/models` on every configured endpoint and builds the `model_id → [endpoints]` mapping used for [model-based routing](#model-based-routing): endpoints hosting the same model form a group, and requests naming that model are distributed within the group by load. If the endpoint list in `modelle.json` changes at runtime (config watchdog hot reload), the mapping is re-fetched automatically on the next poll.

The mapping is additionally written to a temporary file (`vllm-lb-models-*.json` in the system temp directory, e.g. via `TMPDIR`) for debugging / external inspection. It is removed on clean shutdown (SIGINT/SIGTERM); after a `SIGKILL` it is left behind.

## Configuration

Configuration comes from two sources. **Environment variables provide the defaults; the JSON files in `CONFIG_DIR` override them** for every key they define.

| Variable | Default | Description |
|---|---|---|
| `CONFIG_DIR` | `<repo>/config` | Directory holding `einstellung.json` and `modelle.json` |
| `VLLM_1_URL` | `http://vllm1:8000` | Default endpoint 1 (used when `modelle.json` lists no `vllm_urls`) |
| `VLLM_2_URL` | `http://vllm2:8000` | Default endpoint 2 (ditto) |
| `POLL_INTERVAL` | `2.0` | Seconds between `/metrics` poll cycles |
| `TIMEOUT` | `300.0` | Upstream timeout in seconds (total for regular requests; idle/`sock_read` for streamed responses, so a slow but active stream is not cut off) |
| `LISTEN_PORT` | `8000` | Port the balancer listens on. Hot-reloadable: the new port is bound before the old one is closed |
| `QUEUE_TIME_WEIGHT` | `1.0` | Weight of the queue-time term in the score |
| `MAX_RETRIES` | `2` | Retries after the first attempt on transient errors |
| `RETRY_BACKOFF` | `0.2` | Delay between retries, in seconds |
| `OVERLOAD_THRESHOLD` | `10` | Do not retry into an endpoint whose waiting queue is at/above this |
| `KV_CACHE_OVERLOAD_THRESHOLD` | `0.95` | Do not retry into an endpoint whose `kv_cache_usage_perc` is at/above this fraction; an absent value means the signal is ignored |
| `MAX_BODY_SIZE` | `67108864` | Max request body in bytes (64 MiB); larger bodies get `413`. Enforced per request, so it hot-reloads |
| `AUTH_TOKEN` | *(empty)* | When set, clients must present this token (see [Authentication](#authentication)); empty means open |

### `modelle.json` — the fleet

```json
{ "vllm_urls": ["https://a.example", "https://b.example", "http://c.example:8000"] }
```

### `einstellung.json` — the settings

Keys mirror the environment variables above, in lower case. The file is read as an excerpt — every key is optional, and a key that is absent keeps its current value:

```json
{
  "poll_interval": 2.0,
  "timeout": 300.0,
  "queue_time_weight": 1.0,
  "max_retries": 2,
  "retry_backoff": 0.2,
  "overload_threshold": 10,
  "kv_cache_overload_threshold": 0.95,
  "max_body_size": 67108864,
  "listen_port": 8000,
  "auth_token": ""
}
```

Both files are hot-reloaded on change. Values are **validated**: anything out of range, non-finite (`NaN` / `Infinity`, which `json.loads` otherwise accepts) or a `bool` is rejected — the previous value is kept and a warning is logged. The two files are read independently, so a broken `einstellung.json` no longer prevents a valid `modelle.json` from loading.

**Every** setting is hot-reloadable, including `listen_port` and `max_body_size`. Two details matter:

- Deleting a key does **not** reset it — the previous value is kept. Set `"auth_token": ""` to switch authentication off again.
- `listen_port` is moved by binding the new port **before** closing the old listener, so a typo or a port that is already taken logs an error and leaves the balancer running. Requests in flight on the old port are dropped, exactly as on a restart.

## Authentication

Set `auth_token` (or `AUTH_TOKEN`) to require a token. Everything except the operational endpoints then answers `401` without one:

| Header | Meaning |
|---|---|
| `X-Auth-Token: <token>` | The balancer's own credential. **Stripped before forwarding**, so it never reaches a backend |
| `Authorization: Bearer <token>` | Relayed to the backend unchanged — this is the backend's own credential slot (the OpenAI key) — and also accepted as the balancer's token |

Tokens are compared with `hmac.compare_digest`, so a wrong token cannot be recovered character by character from response timing. `/`, `/health`, `/ready` and `/metrics` stay open: probes must keep working while the balancer is unhealthy, and none of them reveals backend data.

> **Enabling `auth_token` breaks stock OpenAI clients.** They send only `Authorization: Bearer <backend key>`, and if that key differs from the balancer token the request is rejected with `401` before it ever reaches a backend. Clients therefore have to send **both** headers when the two tokens differ:
>
> ```bash
> curl -H "X-Auth-Token: $LB_TOKEN" \
>      -H "Authorization: Bearer $VLLM_KEY" \
>      http://lb:8000/v1/chat/completions -d '{"model":"gpt-oss-120b"}'
> ```
>
> If the balancer token and the backend key are the same value, a single `Authorization` header is enough.

**Without a token the balancer is open**, and it says so in a warning at startup. Anything that can reach the port can use the backends, so put it behind your reverse proxy or configure a token.

## Usage

```bash
uv sync
uv run loadbalancer              # console script
uv run python -m loadbalancer.main   # equivalent
```

The balancer reads `.env` from the working directory on startup (existing variables always win) and then applies `CONFIG_DIR`.

### Container

```bash
docker build -t loadbalancer .
docker run --rm -p 8000:8000 \
  -v "$PWD/config:/app/config:ro" \
  -e AUTH_TOKEN=... \
  loadbalancer
```

The image pins `python:3.12-slim`, installs `ca-certificates` for HTTPS to backends (the custom CA bundle is gone), runs as the unprivileged `balancer` user (uid 10001) and declares a `HEALTHCHECK` against `/health` — liveness, not `/ready`, because restarting the container cannot repair a dead backend and a readiness probe would restart-loop. The config directory is mounted read-only; hot reload then applies to the mounted files.

## Development

```bash
uv sync
uv run pytest -q                       # 122 tests
uv run pytest -q --cov=loadbalancer    # coverage (89 %)
uv run ruff check .                    # lint
uv run ruff format --check .           # format
uv run mypy src tests scripts          # strict type check
uv run python scripts/mutants.py       # mutation testing (16 mutants, all must die)
uv run python scripts/smoke.py         # boots the real console script against real backends (14 checks)
```

The proxy tests drive real aiohttp servers over real sockets, so keep-alive, streaming and mid-stream disconnects are exercised rather than mocked; the scoring, config and parser layers are covered by fast unit and property tests.

## Project layout

```
src/loadbalancer/
├── config.py        # configuration: env defaults, JSON files, validation
├── metrics.py       # /metrics parser (Prometheus format)
├── balancer.py      # load score + instance selection
├── proxy.py         # async reverse proxy + streaming + auth
├── telemetry.py     # the balancer's own Prometheus counters
├── watchdog.py      # hot reload of einstellung.json / modelle.json
└── main.py          # metrics polling + server startup + port rebinding
config/              # einstellung.json, modelle.json (hot-reloaded)
Dockerfile           # container image (non-root, HEALTHCHECK)
scripts/
├── mutants.py       # manual mutation testing
└── smoke.py         # real-execution smoke test
tests/               # unit, integration (real sockets), property, auth, container
```

## Known limits

- A broken stream is **reported, not repaired**. After bytes have reached the client there is no retry — a second attempt would splice a different answer onto a half-written body and pay for the inference twice. SSE clients get a terminal `event: error`; the bytes already sent stay incomplete, so the partial answer has to be discarded by the client.
- `listen_port` is changed by re-binding the socket, which drops requests in flight on the old port. A config value the process cannot bind is refused (the old listener keeps serving), but a *successfully* bound wrong port still takes traffic away.
- `X-Auth-Token` is stripped before forwarding, but `Authorization` is relayed. Stock OpenAI clients therefore get `401` once a token is configured unless they also send `X-Auth-Token` (see [Authentication](#authentication)).
- Request bodies are buffered in memory to enforce `max_body_size` (64 MiB by default). There is no streaming upload path.
- Readiness reflects "a backend reported metrics", not "a backend can serve this model"; a model whose endpoints are all unhealthy still yields `503` only once the model is requested.
- The healthcheck probes liveness, so a balancer with every backend down stays `healthy` in Docker. Use `/ready` for load-balancer or orchestration readiness.
- `GET /metrics` used to be relayed to a backend; it now serves the balancer's own counters. Upstream metrics remain available on each backend's own URL.

## License

MIT
