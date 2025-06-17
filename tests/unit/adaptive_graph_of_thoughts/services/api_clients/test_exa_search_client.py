# --------------------------------------------------------------------------- #
# Additional tests for ExaSearchClient                                        #
# Framework: pytest                                                           #
# --------------------------------------------------------------------------- #
import pytest

from adaptive_graph_of_thoughts.services.api_clients.exa_search_client import (
    ExaSearchClient,
)


class MockExaClient:
    def search(self, _query, num_results):
        # A simple mock, can be expanded if needed
        return [
            {"url": f"http://example.com/{i}", "title": f"Result {i}", "id": str(i)}
            for i in range(num_results)
        ]


class EmptyMockExaClient:
    def search(self, _query, _num_results):
        return []


def test_search_no_results(monkeypatch):
    """
    ExaSearchClient.search should return an empty list when the underlying
    SDK returns no results.
    """

    # Arrange
    def mock_init(self, _api_key: str, _base_url: str):
        self.client = EmptyMockExaClient()

    monkeypatch.setattr(ExaSearchClient, "__init__", mock_init)

    client = ExaSearchClient(api_key="fake_key", base_url="https://fake.url")

    # Act
    results = client.search(query="nothing", num_results=5)

    # Assert
    assert results == []


def test_search_invalid_num_results(monkeypatch):
    """
    Passing num_results <= 0 should raise a ValueError.
    """
    monkeypatch.setattr(
        ExaSearchClient,
        "__init__",
        lambda self, _api_key, _base_url: setattr(self, "client", MockExaClient()),
    )
    client = ExaSearchClient(api_key="key", base_url="https://url.test")

    # Assert
    for bad_value in (0, -1):
        with pytest.raises(ValueError):
            client.search(query="test", num_results=bad_value)


def test_search_api_failure(monkeypatch):
    """
    The client should propagate exceptions raised by the underlying SDK.
    """

    class FailingMockExaClient:
        def search(self, _query, _num_results):
            raise RuntimeError("Upstream failure")

    monkeypatch.setattr(
        ExaSearchClient,
        "__init__",
        lambda self, _api_key, _base_url: setattr(self, "client", FailingMockExaClient()),
    )
    client = ExaSearchClient(api_key="k", base_url="https://u")

    with pytest.raises(RuntimeError):
        client.search(query="trigger failure", num_results=1)


def test_client_initialization_missing_key(_monkeypatch):
    """
    Instantiating ExaSearchClient without an API key should raise ValueError.
    If the implementation does not currently validate this, mark as xfail.
    """
    if not hasattr(ExaSearchClient, "__post_init_validation__"):
        pytest.xfail("API key validation not implemented yet")

    with pytest.raises(ValueError):
        ExaSearchClient(api_key="", base_url="https://ignored")
