#!/usr/bin/env python
"""Real-execution smoke test: boots the installed console script and drives it
over real sockets with curl-style HTTP requests.

    uv run python scripts/smoke.py

Exits non-zero on the first failed check. Nothing here is mocked below the
load balancer: the two vLLM stand-ins are real aiohttp servers, one of which
gzip-encodes its answer (the case that used to return 400 through the proxy).
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

PORT = 8471
BACKENDS = 2
TOKEN = "smoke-token-9f3a"


def backend(port: int, gzip_body: bool) -> subprocess.Popen[bytes]:
    code = f"""
import gzip
from aiohttp import web
async def h(request):
    body = gzip.compress(b'{{"ok":true,"backend":{port}}}') if {gzip_body} else b'{{"ok":true,"backend":{port}}}'
    headers = {{"content-type": "application/json"}}
    if {gzip_body}:
        headers["content-encoding"] = "gzip"
    resp = web.Response(body=body, headers=headers)
    resp.headers.add("Set-Cookie", "a=1; Path=/")
    resp.headers.add("Set-Cookie", "b=2; Path=/")
    return resp
async def models(request):
    return web.json_response({{"object": "list", "data": [{{"id": "model-{port}"}}]}})
app = web.Application()
app.router.add_get("/v1/models", models)
app.router.add_route("*", "/{{p:.*}}", h)
web.run_app(app, host="127.0.0.1", port={port}, print=None)
"""
    return subprocess.Popen([sys.executable, "-c", code])


def request(
    path: str, data: bytes | None = None, headers: dict[str, str] | None = None
) -> tuple[int, bytes, "Message[str]"]:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data)
    req.add_header("content-type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    opener = urllib.request.build_opener()
    try:
        with opener.open(req, timeout=10) as resp:
            return resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers


def main() -> int:
    procs = [backend(8500 + i, gzip_body=(i == 0)) for i in range(BACKENDS)]
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "modelle.json").write_text(
            json.dumps({"vllm_urls": [f"http://127.0.0.1:{8500 + i}" for i in range(BACKENDS)]})
        )
        Path(tmp, "einstellung.json").write_text(json.dumps({"poll_interval": 0.5}))
        env = {
            **os.environ,
            "CONFIG_DIR": tmp,
            "LISTEN_PORT": str(PORT),
            "POLL_INTERVAL": "0.5",
            "MAX_RETRIES": "1",
        }
        lb = subprocess.Popen(["uv", "run", "loadbalancer"], env=env)
        try:
            for _ in range(100):
                try:
                    status, _, _ = request("/health")
                    if status == 200:
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                print("FAIL: loadbalancer never became reachable")
                return 1

            checks: list[tuple[str, bool, str]] = []
            status, body, _ = request("/health")
            checks.append(("GET /health -> 200", status == 200, str(status)))

            status, body, _ = request("/v1/models")
            ids = {m["id"] for m in json.loads(body)["data"]} if status == 200 else set()
            checks.append(
                (
                    "GET /v1/models aggregates both backends",
                    ids == {"model-8500", "model-8501"},
                    f"{status} {sorted(ids)}",
                )
            )

            # gzip-encoding backend: used to fail with 400 ContentEncodingError
            status, body, headers = request(
                "/v1/chat/completions?trace=smoke&user=1",
                json.dumps({"model": "m"}).encode(),
            )
            checks.append(
                ("POST with query + gzip backend -> 200", status == 200, f"{status} {body[:80]!r}")
            )
            checks.append(
                (
                    "no content-encoding leaked",
                    "content-encoding" not in {k.lower() for k in headers},
                    str(headers.get("Content-Encoding")),
                )
            )
            checks.append(
                (
                    "content-length matches body",
                    headers.get("Content-Length") == str(len(body)),
                    f"{headers.get('Content-Length')} vs {len(body)}",
                )
            )
            checks.append(
                (
                    "body is the decoded json",
                    json.loads(body).get("ok") is True,
                    body[:80].decode(errors="replace"),
                )
            )

            # --- the balancer's own operational surface -------------------
            status, body, headers = request("/ready")
            checks.append(("GET /ready -> 200 while backends are up", status == 200, str(status)))

            status, body, metrics_headers = request("/metrics")
            text = body.decode()
            content_type = metrics_headers.get("Content-Type", "")
            checks.append(
                (
                    "GET /metrics serves the balancer's own counters",
                    status == 200
                    and "loadbalancer_requests_total" in text
                    and "version=0.0.4" in content_type,
                    f"{status} {content_type}",
                )
            )
            # Parse the value: a substring check would also pass when the
            # series is missing altogether.
            served = 0.0
            for line in text.splitlines():
                if line.startswith("loadbalancer_requests_total "):
                    served = float(line.split()[1])
            checks.append(
                (
                    # counts proxied requests only: one POST so far, while
                    # /health, /ready and /metrics are served locally
                    "metrics count exactly the proxied request",
                    served == 1,
                    f"requests_total={served}",
                )
            )

            # --- auth: a configured token must be enforced -----------------
            # A token is configured through the watched config file, so this
            # covers hot reload and auth enforcement in one real run.
            Path(tmp, "einstellung.json").write_text(
                json.dumps({"poll_interval": 0.5, "auth_token": TOKEN})
            )
            auth_status = 0
            for _ in range(50):
                time.sleep(0.1)
                auth_status, _, _ = request("/v1/models")
                if auth_status == 401:
                    break
            checks.append(
                (
                    "auth_token hot-reloaded: unauthenticated -> 401",
                    auth_status == 401,
                    str(auth_status),
                )
            )
            status, _, _ = request("/v1/models", headers={"X-Auth-Token": TOKEN})
            checks.append(("authenticated request -> 200", status == 200, str(status)))
            status, _, _ = request("/v1/models", headers={"X-Auth-Token": TOKEN[:-1]})
            checks.append(("token prefix -> 401", status == 401, str(status)))
            # A removed key keeps the previous value, so auth is turned off by
            # setting it to the empty string.
            Path(tmp, "einstellung.json").write_text(
                json.dumps({"poll_interval": 0.5, "auth_token": ""})
            )
            for _ in range(50):
                time.sleep(0.1)
                status, _, _ = request("/v1/models")
                if status == 200:
                    break
            checks.append(("auth_token cleared: open again -> 200", status == 200, str(status)))

            # --- repeated response headers survive the relay ---------------
            status, _, relayed = request(
                "/v1/chat/completions", json.dumps({"model": "m"}).encode()
            )
            cookies = relayed.get_all("Set-Cookie", [])
            checks.append(
                (
                    "repeated Set-Cookie headers are relayed",
                    len(cookies) == 2,
                    str(cookies),
                )
            )

            failures = 0
            for name, ok, detail in checks:
                print(f"  {'ok  ' if ok else 'FAIL'}  {name}  [{detail}]")
                failures += not ok
            print(f"\n{len(checks) - failures}/{len(checks)} smoke checks passed")
            return 1 if failures else 0
        finally:
            lb.send_signal(signal.SIGTERM)
            try:
                lb.wait(timeout=10)
            except subprocess.TimeoutExpired:
                lb.kill()
            for proc in procs:
                proc.terminate()
                proc.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
