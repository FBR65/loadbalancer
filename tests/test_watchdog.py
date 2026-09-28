"""ConfigWatcher: hot reload of einstellung.json / modelle.json."""

import asyncio
from collections.abc import Callable
from pathlib import Path

from loadbalancer.config import Config
from loadbalancer.watchdog import ConfigWatcher, _ConfigFileHandler


def _settings(path: Path, payload: object) -> None:
    import json

    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    await asyncio.wait_for(_loop_until(predicate), timeout)


async def _loop_until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0.02)


def test_is_running_false_before_start(tmp_path: Path) -> None:
    w = ConfigWatcher(Config(), tmp_path)
    assert not w.is_running()


def test_start_and_stop_lifecycle(tmp_path: Path) -> None:
    w = ConfigWatcher(Config(), tmp_path)
    w.start()
    try:
        assert w.is_running()
        w.start()  # idempotent
        assert w.is_running()
    finally:
        w.stop()
    assert not w.is_running()


def test_stop_without_start_is_noop(tmp_path: Path) -> None:
    ConfigWatcher(Config(), tmp_path).stop()  # must not raise


def test_check_and_reload_applies_settings_and_models(tmp_path: Path) -> None:
    cfg = Config()
    w = ConfigWatcher(cfg, tmp_path)
    _settings(tmp_path / "einstellung.json", {"max_retries": 7, "poll_interval": 3.5})
    _settings(tmp_path / "modelle.json", {"vllm_urls": ["http://x:1"]})
    w.check_and_reload()
    assert cfg.max_retries == 7
    assert cfg.poll_interval == 3.5
    assert cfg.vllm_urls == ["http://x:1"]


def test_handler_events_trigger_reload(tmp_path: Path) -> None:
    cfg = Config()
    w = ConfigWatcher(cfg, tmp_path)
    _settings(tmp_path / "einstellung.json", {"max_retries": 5})
    handler = _ConfigFileHandler(w)
    handler.on_modified(object())
    assert cfg.max_retries == 5
    _settings(tmp_path / "einstellung.json", {"max_retries": 6})
    handler.on_created(object())
    assert cfg.max_retries == 6


async def test_file_change_is_picked_up_while_running(tmp_path: Path) -> None:
    """End to end: touching einstellung.json updates the live Config."""
    cfg = Config()
    w = ConfigWatcher(cfg, tmp_path)
    w.start()
    try:
        _settings(tmp_path / "einstellung.json", {"queue_time_weight": 4.25})
        await _wait_until(lambda: cfg.queue_time_weight == 4.25)
    finally:
        w.stop()
    assert cfg.queue_time_weight == 4.25


async def test_model_url_change_is_picked_up_while_running(tmp_path: Path) -> None:
    cfg = Config(vllm_urls=["http://old:1"])
    w = ConfigWatcher(cfg, tmp_path)
    w.start()
    try:
        _settings(tmp_path / "modelle.json", {"vllm_urls": ["http://new:1"]})
        await _wait_until(lambda: cfg.vllm_urls == ["http://new:1"])
    finally:
        w.stop()
    assert cfg.vllm_urls == ["http://new:1"]


def test_stop_is_idempotent(tmp_path: Path) -> None:
    w = ConfigWatcher(Config(), tmp_path)
    w.start()
    w.stop()
    w.stop()  # second stop must not raise


def test_custom_file_names(tmp_path: Path) -> None:
    w = ConfigWatcher(Config(), tmp_path, settings_file="s.json", models_file="m.json")
    assert w.settings_path.name == "s.json"
    assert w.models_path.name == "m.json"


async def test_watcher_does_not_leak_thread(tmp_path: Path) -> None:
    w = ConfigWatcher(Config(), tmp_path)
    w.start()
    w.stop()
    await asyncio.sleep(0)
