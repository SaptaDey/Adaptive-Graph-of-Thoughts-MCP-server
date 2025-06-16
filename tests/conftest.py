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
