# syntax=docker/dockerfile:1
#
# Container image for the load balancer. `docker build -t loadbalancer .`
# `docker run --rm -p 8000:8000 -v "$PWD/config:/app/config:ro" loadbalancer`
#
# The config directory is mounted read-only so the image ships no mutable
# state; hot reload then applies to the mounted files.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    CONFIG_DIR=/app/config \
    LISTEN_PORT=8000 \
    PATH="/app/.venv/bin:$PATH"

# ca-certificates is what the upstream HTTPS calls need; the balancer no longer
# ships a custom CA bundle, it uses the system trust store.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first: a source-only change then reuses this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# Unprivileged runtime user; the config mount only has to be readable.
RUN useradd --system --create-home --uid 10001 balancer \
    && chown -R balancer:balancer /app
USER balancer

EXPOSE 8000

# Liveness, not readiness: restarting the container cannot repair a dead
# backend, so a failing /ready must not trigger a restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

ENTRYPOINT ["loadbalancer"]
