# src/loadbalancer/watchdog.py
import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from loadbalancer.config import Config, reload_from_files

logger = logging.getLogger("loadbalancer.watchdog")


class _ConfigFileHandler(FileSystemEventHandler):
    def __init__(self, watcher: "ConfigWatcher") -> None:
        self._watcher = watcher

    def on_modified(self, event: object) -> None:
        self._watcher.check_and_reload()

    def on_created(self, event: object) -> None:
        self._watcher.check_and_reload()


class ConfigWatcher:
    def __init__(
        self,
        config: Config,
        config_dir: Path,
        settings_file: str = "einstellung.json",
        models_file: str = "modelle.json",
    ) -> None:
        self.config = config
        self.config_dir = config_dir
        self.settings_path = config_dir / settings_file
        self.models_path = config_dir / models_file
        self._observer: BaseObserver | None = None

    def check_and_reload(self) -> None:
        reload_from_files(
            self.config,
            settings_path=self.settings_path,
            models_path=self.models_path,
        )

    def start(self) -> None:
        if self._observer is not None:
            return
        self._observer = Observer()
        self._observer.schedule(
            _ConfigFileHandler(self),
            str(self.config_dir),
            recursive=False,
        )
        self._observer.daemon = True
        self._observer.start()
        logger.info("config watchdog started on %s", self.config_dir)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("config watchdog stopped")

    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()
