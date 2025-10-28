from __future__ import annotations

# Lightweight pytest helpers for running asyncio tests without extra plugins.

import asyncio
import inspect

import pytest


def _should_handle_asyncio(pyfuncitem: pytest.Function) -> bool:
    """Return ``True`` if the test should run inside an asyncio loop."""

    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return False
    return pyfuncitem.get_closest_marker("asyncio") is not None


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool:
    """Execute ``@pytest.mark.asyncio`` tests using ``asyncio.run``."""

    if not _should_handle_asyncio(pyfuncitem):
        return False

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(pyfuncitem.obj(**pyfuncitem.funcargs))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    return True


def pytest_configure(config: pytest.Config) -> None:
    """Register the custom asyncio marker to silence warnings."""

    config.addinivalue_line("markers", "asyncio: mark test to run in an asyncio event loop")
