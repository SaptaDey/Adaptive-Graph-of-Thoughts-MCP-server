"""Pytest configuration for the test suite."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Hook for configuring pytest during initialization."""
    # No-op configuration hook. Originally this exited early to disable tests
    # when the repository was used without its optional dependencies. The test
    # suite now runs by default, so we simply return to allow normal execution.
    return
