# --------------------------------------------------------------------------- #
# Additional tests for ExaSearchClient                                        #
# Framework: pytest                                                           #
# --------------------------------------------------------------------------- #

class EmptyMockExaClient:
    def search(self, query, num_results):
        """
        Returns an empty list to simulate no search results for the given query.
        
        Args:
            query: The search query string.
            num_results: The maximum number of results to return.
        
        Returns:
            An empty list, indicating no results found.
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
        Initializes the instance with an EmptyMockExaClient as its client.
        
        Replaces the standard initialization to inject a mock client that always returns no results, typically for testing purposes.
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
    Tests that ExaSearchClient.search raises a ValueError when num_results is zero or negative.
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
    Tests that ExaSearchClient.search propagates exceptions from the underlying client.
    
    This test replaces the internal client with a mock that raises a RuntimeError, and verifies that the exception is not caught or altered by ExaSearchClient.
    """
    class FailingMockExaClient:
        def search(self, query, num_results):
            """
            Simulates a search operation that always fails.
            
            Raises:
                RuntimeError: Always raised to simulate an upstream failure.
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
    Tests that creating an ExaSearchClient without an API key raises a ValueError.
    
    If API key validation is not implemented, the test is marked as expected to fail.
    """
    if not hasattr(ExaSearchClient, "__post_init_validation__"):
        pytest.xfail("API key validation not implemented yet")

    with pytest.raises(ValueError):
        ExaSearchClient(api_key="", base_url="https://ignored")