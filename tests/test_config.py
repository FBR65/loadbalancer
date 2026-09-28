"""Configuration loading and file-based reload must reject hostile/garbage values.

A config file is operator-edited JSON, but it is still an untrusted input
surface: a typo or a copy-paste of a JSON `NaN` must not silently disable the
upstream timeout or shrink the retry budget. These tests pin the hardening.
"""

import json
import math
from pathlib import Path

from loadbalancer.config import Config, load_config, load_dotenv, reload_from_files


def _write(path: Path, name: str, payload: object) -> None:
    (path / name).write_text(json.dumps(payload) if not isinstance(payload, str) else payload)


def test_negative_max_retries_never_produces_a_zero_attempt_loop() -> None:
    """A negative budget must be rejected so the retry loop always runs >=1 attempt.

    We keep the previous/default value (rather than clamping to 0) because a
    negative number is a typo, not a request for "zero retries".
    """
    cfg = Config()
    reload_from_files(cfg, settings_path=_tmp_settings({"max_retries": -1}))
    assert cfg.max_retries >= 0
    assert cfg.max_retries == 2, "out-of-range value must be rejected, default kept"


def test_max_retries_from_env_is_rejected_when_negative(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MAX_RETRIES", "-5")
    assert load_config().max_retries == 2


def test_nan_and_inf_are_rejected_and_old_value_kept() -> None:
    """`json.loads` accepts NaN/Infinity; a non-finite timeout is unusable."""
    cfg = Config()
    before = cfg.timeout
    reload_from_files(cfg, settings_path=_tmp_settings({"timeout": float("nan")}))
    assert math.isfinite(cfg.timeout)
    assert cfg.timeout == before
    reload_from_files(cfg, settings_path=_tmp_settings({"poll_interval": float("inf")}))
    assert math.isfinite(cfg.poll_interval)


def test_bool_is_rejected_for_numeric_settings() -> None:
    """`int(True) == 1` would silently turn on retries / change the port."""
    cfg = Config()
    reload_from_files(cfg, settings_path=_tmp_settings({"max_retries": True, "listen_port": False}))
    assert cfg.max_retries == 2, "bool must not be coerced to 1"
    assert cfg.listen_port == 8000, "bool must not be coerced to 0"


def test_out_of_range_values_are_rejected() -> None:
    cfg = Config()
    reload_from_files(
        cfg,
        settings_path=_tmp_settings(
            {"poll_interval": -1.0, "listen_port": 70000, "max_body_size": -5, "timeout": -1.0}
        ),
    )
    assert cfg.poll_interval > 0
    assert 0 < cfg.listen_port < 65536
    assert cfg.max_body_size > 0
    assert cfg.timeout > 0


def test_malformed_settings_do_not_block_a_valid_models_file(tmp_path: Path) -> None:
    """One broken file must not prevent the other from loading (per-file isolation)."""
    _write(tmp_path, "einstellung.json", "{ not json")
    _write(tmp_path, "modelle.json", {"vllm_urls": ["http://a:1", "http://b:2"]})
    cfg = load_config(config_dir=tmp_path)
    assert cfg.vllm_urls == ["http://a:1", "http://b:2"]


def test_models_file_with_non_string_urls_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "modelle.json", {"vllm_urls": ["http://a:1", 42, None]})
    cfg = load_config(config_dir=tmp_path)
    assert cfg.vllm_urls != ["http://a:1", 42, None]
    assert all(isinstance(u, str) for u in cfg.vllm_urls)


def test_ca_bundle_helpers_are_gone() -> None:
    """The ITZBund CA bundle is no longer used; nothing may reference it."""
    import loadbalancer.config as cfgmod

    assert not hasattr(cfgmod, "upstream_ssl_context")
    assert not hasattr(cfgmod, "SSL_CERT_FILE")
    assert not hasattr(cfgmod, "_DEFAULT_CA_BUNDLE")


def test_load_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    env = tmp_path / ".env"
    env.write_text('FOO=from_file\nBAR="quoted"\n# comment\nNOEQUALS\n')
    monkeypatch.setenv("FOO", "from_env")
    monkeypatch.delenv("BAR", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["FOO"] == "from_env"
    assert os.environ["BAR"] == "quoted"
    assert "NOEQUALS" not in os.environ


def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "does-not-exist")  # must not raise


def _tmp_settings(payload: object) -> Path:
    import tempfile

    d = Path(tempfile.mkdtemp())
    _write(d, "einstellung.json", payload)
    return d / "einstellung.json"
