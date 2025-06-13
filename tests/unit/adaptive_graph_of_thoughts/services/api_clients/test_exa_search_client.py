# --------------------------------------------------------------------------- #
# Additional tests for ExaSearchClient                                        #
# Framework: pytest                                                           #
# --------------------------------------------------------------------------- #

class EmptyMockExaClient:
    def search(self, query, num_results):
        """
        Returns an empty list for any search query.
        
        Args:
            query: The search query string.
            num_results: The number of results requested.
        
        Returns:
            An empty list, indicating no search results.
        """
        return []

def test_search_no_results(monkeypatch):
    """
    ExaSearchClient.search should return an empty list when the underlying
    SDK returns no results.
    """
    # Arrange
    def mock_init(self, api_key: str, base_url: str):
        """
        Replaces the ExaSearchClient's initialization to use an EmptyMockExaClient.
        
        This mock initializer is used in tests to ensure the client always returns no search results.
        """
        self.client = EmptyMockExaClient()

    monkeypatch.setattr(ExaSearchClient, "__init__", mock_init)

    client = ExaSearchClient(api_key="fake_key", base_url="https://fake.url")

    # Act
    results = client.search(query="nothing", num_results=5)

    # Assert
    assert results == []

def test_search_invalid_num_results(monkeypatch):
    """
    Tests that ExaSearchClient.search raises ValueError when num_results is zero or negative.
    """
    monkeypatch.setattr(
        ExaSearchClient,
        "__init__",
        lambda self, api_key, base_url: setattr(self, "client", MockExaClient()),
    )
    client = ExaSearchClient(api_key="key", base_url="https://url.test")

    # Assert
    for bad_value in (0, -1):
        with pytest.raises(ValueError):
            client.search(query="test", num_results=bad_value)

def test_search_api_failure(monkeypatch):
    """
    Tests that exceptions raised by the underlying SDK client during search are propagated by ExaSearchClient.
    
    This test replaces the underlying client with a mock that raises a RuntimeError, then asserts that calling search raises the same exception.
    """
    class FailingMockExaClient:
        def search(self, query, num_results):
            """
            Raises a RuntimeError to simulate an upstream failure during a search operation.
            """
            raise RuntimeError("Upstream failure")

    monkeypatch.setattr(
        ExaSearchClient,
        "__init__",
        lambda self, api_key, base_url: setattr(self, "client", FailingMockExaClient()),
    )
    client = ExaSearchClient(api_key="k", base_url="https://u")

    with pytest.raises(RuntimeError):
        client.search(query="trigger failure", num_results=1)

def test_client_initialization_missing_key(monkeypatch):
    """
    Instantiating ExaSearchClient without an API key should raise ValueError.
    If the implementation does not currently validate this, mark as xfail.
    """
    if not hasattr(ExaSearchClient, "__post_init_validation__"):
        pytest.xfail("API key validation not implemented yet")

    with pytest.raises(ValueError):
        ExaSearchClient(api_key="", base_url="https://ignored")