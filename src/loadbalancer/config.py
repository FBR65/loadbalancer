# src/loadbalancer/config.py
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("loadbalancer.config")


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
    auth_token: str = ""


# Accepted settings: json key -> (attribute, converter, inclusive min, inclusive max).
# A value outside the range (or non-finite, or a bool) is rejected and the current
# value is kept, so a typo can never disable the upstream timeout or the retry budget.
_SETTINGS: dict[str, tuple[str, Any, float | None, float | None]] = {
    "poll_interval": ("poll_interval", float, 0.1, None),
    "timeout": ("timeout", float, 0.1, None),
    "listen_port": ("listen_port", int, 1, 65535),
    "queue_time_weight": ("queue_time_weight", float, 0.0, None),
    "max_retries": ("max_retries", int, 0, None),
    "retry_backoff": ("retry_backoff", float, 0.0, None),
    "overload_threshold": ("overload_threshold", int, 0, None),
    "kv_cache_overload_threshold": ("kv_cache_overload_threshold", float, 0.0, 1.0),
    "max_body_size": ("max_body_size", int, 1, None),
    "auth_token": ("auth_token", str, None, None),
}


def _coerce(attr: str, converter: Any, raw: object, lo: float | None, hi: float | None) -> Any:
    """Convert `raw` and enforce the range, or return None when unacceptable.

    Bools are rejected outright: `int(True) == 1` would silently rewrite a
    numeric setting. Non-finite floats (`NaN`/`Infinity`, which `json.loads`
    accepts) never pass the bound checks.
    """
    if converter is str:
        if not isinstance(raw, str):
            logger.warning("ignoring %s: must be a string", attr)
            return None
        return raw
    if isinstance(raw, bool):
        logger.warning("ignoring %s: bool is not a valid number", attr)
        return None
    try:
        value = converter(raw)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(value):
        return None
    if lo is not None and value < lo:
        return None
    if hi is not None and value > hi:
        return None
    return value


def _apply_setting(config: Config, key: str, raw: object) -> None:
    spec = _SETTINGS.get(key)
    if spec is None:
        return
    attr, converter, lo, hi = spec
    value = _coerce(attr, converter, raw, lo, hi)
    if value is None:
        logger.warning("ignoring invalid setting %s=%r", key, raw)
        return
    setattr(config, attr, value)


def _env_text(name: str, default: str) -> str:
    """A non-numeric setting, read verbatim (an auth token must not be coerced)."""
    return os.getenv(name, default) or ""


def _env_number(
    name: str, default: float, converter: Any, lo: float | None, hi: float | None
) -> Any:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = _coerce(name, converter, raw, lo, hi)
    if value is None:
        logger.warning("ignoring invalid env %s=%r; using %r", name, raw, default)
        return default
    return value


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


def _read_json(path: Path) -> dict[str, Any] | None:
    """Parse a JSON object from `path`; None on any error (missing/broken/not an object)."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("cannot read %s: %s", path.name, exc)
        return None
    return data if isinstance(data, dict) else None


def reload_from_files(
    config: Config,
    settings_path: Path | None = None,
    models_path: Path | None = None,
    config_dir: Path | None = None,
) -> None:
    """Apply operator-edited JSON files onto `config`.

    Each file is handled independently: a broken settings file must not stop a
    valid models file (and vice versa) from being applied.
    """
    if config_dir is not None:
        if settings_path is None:
            settings_path = config_dir / "einstellung.json"
        if models_path is None:
            models_path = config_dir / "modelle.json"

    if settings_path is not None:
        data = _read_json(settings_path)
        if data is not None:
            for key in _SETTINGS:
                if key in data:
                    _apply_setting(config, key, data[key])

    if models_path is not None:
        data = _read_json(models_path)
        if data is not None:
            urls = data.get("vllm_urls")
            if isinstance(urls, list) and all(isinstance(u, str) for u in urls):
                config.vllm_urls = urls
            else:
                logger.warning("ignoring %s: vllm_urls must be a list of strings", models_path.name)


def load_config(config_dir: Path | None = None) -> Config:
    config = Config(
        vllm_urls=[
            os.getenv("VLLM_1_URL", "http://vllm1:8000"),
            os.getenv("VLLM_2_URL", "http://vllm2:8000"),
        ],
    )
    config.auth_token = _env_text("AUTH_TOKEN", config.auth_token)
    for key, (attr, converter, lo, hi) in _SETTINGS.items():
        if converter is str:
            continue  # already read above; the numeric reader cannot handle text
        default = getattr(config, attr)
        env_name = key.upper()
        value = _env_number(env_name, default, converter, lo, hi)
        setattr(config, attr, value)
    if config_dir is not None:
        reload_from_files(config, config_dir=config_dir)
    return config
