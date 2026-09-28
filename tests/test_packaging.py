"""SC-5: the distribution actually builds and exposes a working entry point."""

import inspect
from importlib.metadata import entry_points

import vllm_lb


def test_package_exports_its_api() -> None:
    assert "build_app" in vllm_lb.__all__
    assert callable(vllm_lb.build_app)
    assert "vllm-lb!" not in (vllm_lb.__doc__ or "")


def test_console_script_resolves_to_a_callable() -> None:
    matches = [e for e in entry_points(group="console_scripts") if e.name == "loadbalancer"]
    assert len(matches) == 1, f"console script not installed: {matches}"
    assert callable(matches[0].load())


def test_console_script_entry_point_is_synchronous() -> None:
    """A console script that returns a coroutine starts nothing at all."""
    (script,) = [e for e in entry_points(group="console_scripts") if e.name == "loadbalancer"]
    target = script.load()

    assert not inspect.iscoroutinefunction(target), (
        "entry point must be a sync function that drives the event loop"
    )


def test_importing_the_package_does_not_start_anything() -> None:
    import importlib

    module = importlib.import_module("vllm_lb")
    assert module is vllm_lb
