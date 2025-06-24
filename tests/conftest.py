import warnings

import pytest


# Register custom markers
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark tests as slow")


# Ignore PendingDeprecationWarning from Starlette multipart import
warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
)
# Filter out pydantic v2 deprecation warnings for old validator usage
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"adaptive_graph_of_thoughts\.api\.schemas",
)


@pytest.fixture(name="_sample_hypothesis_data")
def _sample_hypothesis_data_fixture(sample_hypothesis_data):
    """Alias for sample_hypothesis_data used by some tests."""
    return sample_hypothesis_data


@pytest.fixture(name="_monkeypatch")
def _monkeypatch_fixture(monkeypatch):
    """Alias for the standard monkeypatch fixture."""
    return monkeypatch
