# src/vllm_lb/config.py
import json
import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path

# CA bundle for upstream (vLLM) connections. The container (Dockerfile) sets
# SSL_CERT_FILE to a merged bundle (system CAs + ITZBund); locally it falls back
# to the repo's internal CA. Overridable via the SSL_CERT_FILE env var.
_DEFAULT_CA_BUNDLE = Path(__file__).resolve().parents[2] / "certs" / "itzbund-ca.pem"
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", str(_DEFAULT_CA_BUNDLE))


def upstream_ssl_context() -> ssl.SSLContext:
    """SSL context for upstream (vLLM) connections.

    System CA store plus the SSL_CERT_FILE bundle (e.g. the ITZBund internal CA),
    layered on top so both public and internal certificates verify. The bundle
    is *added* to the system store, never used instead of it, so a single
    internal CA does not drop the public CAs. http backends are unaffected
    (the context only applies to https).
    """
    # create_default_context() reads SSL_CERT_FILE and would use it as the sole
    # store, losing the public CAs if the bundle is internal-only. Clear it
    # temporarily so we start from the true system store, then layer on top.
    saved = os.environ.pop("SSL_CERT_FILE", None)
    try:
        ctx = ssl.create_default_context()
    finally:
        if saved is not None:
            os.environ["SSL_CERT_FILE"] = saved
    bundle = Path(SSL_CERT_FILE)
    if bundle.is_file():
        ctx.load_verify_locations(bundle)
    return ctx


@dataclass
class Config:
    vllm_urls: list[str] = field(default_factory=list)
    poll_interval: float = 2.0
    timeout: float = 300.0
    listen_port: int = 8000
    queue_time_weight: float = 1.0
    max_retries: int = 2
    retry_backoff: float = 0.2
    overload_threshold: int = 10
    kv_cache_overload_threshold: float = 0.95
    max_body_size: int = 64 * 1024**2


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (no overrides).

    Existing environment variables always take precedence (same as python-dotenv
    with override=False). Missing file or malformed lines are silently ignored.
    """
    candidate = path if path is not None else Path.cwd() / ".env"
    if not candidate.is_file():
        return
    try:
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


def reload_from_files(
    config: Config,
    settings_path: Path | None = None,
    models_path: Path | None = None,
    config_dir: Path | None = None,
) -> None:
    if config_dir is not None:
        if settings_path is None:
            settings_path = config_dir / "einstellung.json"
        if models_path is None:
            models_path = config_dir / "modelle.json"
    if settings_path is not None and settings_path.is_file():
        try:
            data = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        _SETTINGS_KEYS = {
            "poll_interval": (float, "poll_interval"),
            "timeout": (float, "timeout"),
            "listen_port": (int, "listen_port"),
            "queue_time_weight": (float, "queue_time_weight"),
            "max_retries": (int, "max_retries"),
            "retry_backoff": (float, "retry_backoff"),
            "overload_threshold": (int, "overload_threshold"),
            "kv_cache_overload_threshold": (float, "kv_cache_overload_threshold"),
            "max_body_size": (int, "max_body_size"),
        }
        for json_key, (typ, attr) in _SETTINGS_KEYS.items():
            if json_key in data:
                try:
                    setattr(config, attr, typ(data[json_key]))
                except (ValueError, TypeError):
                    pass

    if models_path is not None and models_path.is_file():
        try:
            data = json.loads(models_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        urls = data.get("vllm_urls")
        if isinstance(urls, list) and all(isinstance(u, str) for u in urls):
            config.vllm_urls = urls


def load_config(config_dir: Path | None = None) -> Config:
    config = Config(
        vllm_urls=[
            os.getenv("VLLM_1_URL", "http://vllm1:8000"),
            os.getenv("VLLM_2_URL", "http://vllm2:8000"),
        ],
        poll_interval=float(os.getenv("POLL_INTERVAL", "2.0")),
        timeout=float(os.getenv("TIMEOUT", "300.0")),
        listen_port=int(os.getenv("LISTEN_PORT", "8000")),
        queue_time_weight=float(os.getenv("QUEUE_TIME_WEIGHT", "1.0")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
        retry_backoff=float(os.getenv("RETRY_BACKOFF", "0.2")),
        overload_threshold=int(os.getenv("OVERLOAD_THRESHOLD", "10")),
        kv_cache_overload_threshold=float(os.getenv("KV_CACHE_OVERLOAD_THRESHOLD", "0.95")),
        max_body_size=int(os.getenv("MAX_BODY_SIZE", str(64 * 1024**2))),
    )
    if config_dir is not None:
        reload_from_files(config, config_dir=config_dir)
    return config
