"""Pytest configuration for the test suite."""

import pytest

# Pre-load the real adaptive_graph_of_thoughts.config module so that stub modules
# registered via sys.modules.setdefault in individual test files (e.g. test_health.py)
# do not shadow the real module for tests collected later in the session.
try:
    import adaptive_graph_of_thoughts.config  # noqa: F401
except Exception:
    pass


def pytest_configure(config: pytest.Config) -> None:
    """Hook for configuring pytest during initialization."""
    # No-op configuration hook. Originally this exited early to disable tests
    # when the repository was used without its optional dependencies. The test
    # suite now runs by default, so we simply return to allow normal execution.
    return
