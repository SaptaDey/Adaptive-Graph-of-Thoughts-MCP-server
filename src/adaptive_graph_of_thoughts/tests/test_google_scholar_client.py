# Fixtures and helpers for mocking HTTP responses and loading sample HTML
import os
import requests
import pytest
from adaptive_graph_of_thoughts.google_scholar_client import (
    GoogleScholarClient,
    GoogleScholarError,
)

def load_fixture(filename):
    """
    Loads and returns the contents of a fixture file from the 'fixtures' directory.
    
    Args:
        filename: Name of the fixture file to load.
    
    Returns:
        The contents of the specified fixture file as a string.
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", filename
    )
    with open(fixture_path, encoding="utf-8") as f:
        return f.read()

@pytest.fixture
def mock_response(monkeypatch):
    """
    Pytest fixture that mocks `requests.get` to return a successful HTTP response with HTML content from a fixture file.
    
    Args:
        monkeypatch: Pytest's monkeypatch fixture for patching `requests.get`.
    
    Returns:
        A function that, when called with a fixture filename, patches `requests.get` to return a mock response containing the file's HTML content.
    """
    def _mock(html_file):
        """
        Mocks the `requests.get` method to return a response with HTML content from a fixture file.
        
        The mocked response has a status code of 200 and its `text` attribute contains the contents of the specified HTML fixture file. The `raise_for_status` method is overridden to do nothing.
        """
        html = load_fixture(html_file)
        class MockResponse:
            status_code = 200
            text = html
            def raise_for_status(self):
                """A no-op method that simulates successful HTTP status checks in mocked responses."""
                pass
        monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())
    return _mock

@pytest.fixture
def mock_error_response(monkeypatch):
    """
    Pytest fixture that mocks `requests.get` to return an HTTP response with a specified error status code.
    
    Returns:
        A function that, when called with a status code, sets up the mock to raise a `requests.HTTPError` with that status code on `raise_for_status()`.
    """
    def _mock(status_code):
        """
        Creates a mock HTTP response with a specified status code that raises an HTTPError when checked.
        
        Intended for use in tests to simulate error responses from HTTP requests.
        """
        class MockResponse:
            status_code = status_code
            text = ""
            def raise_for_status(self):
                raise requests.HTTPError(f"{status_code} Error")
        monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())
    return _mock

@pytest.mark.parametrize(
    "query,max_results,expected_count",
    [
        ("machine learning", 5, 5),
        ("deep learning", 1, 1),
        ("NLP", 3, 3),
    ],
)
def test_search_happy_path(mock_response, query, max_results, expected_count):
    """Happy path returns list of dicts with required fields."""
    mock_response("google_scholar_search_sample.html")
    client = GoogleScholarClient()
    results = client.search(query, max_results=max_results)
    assert isinstance(results, list)
    assert len(results) == expected_count
    for article in results:
        assert isinstance(article["title"], str) and article["title"]
        assert isinstance(article["url"], str) and article["url"]
        assert isinstance(article["year"], int)
        assert isinstance(article["citations"], int)

@pytest.mark.parametrize(
    "query",
    ["", "a" * 1001, "!@#$%^&*()"]
)
def test_search_invalid_query_raises(query):
    """Invalid queries should raise ValueError."""
    client = GoogleScholarClient()
    with pytest.raises(ValueError):
        client.search(query)

@pytest.mark.parametrize("status_code", [500, 502, 503])
def test_search_http_error_raises(mock_error_response, status_code):
    """HTTP errors result in GoogleScholarError."""
    mock_error_response(status_code)
    client = GoogleScholarClient()
    with pytest.raises(GoogleScholarError):
        client.search("test")

def test_search_timeout_raises(monkeypatch):
    """
    Tests that a network timeout during a search raises a GoogleScholarError.
    """
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout),
    )
    client = GoogleScholarClient()
    with pytest.raises(GoogleScholarError):
        client.search("timeout")

def test_search_malformed_response_raises(monkeypatch):
    """Malformed HTML leads to parsing errors as GoogleScholarError."""
    class MockResp:
        status_code = 200
        text = "<html><body>no expected markers</body></html>"
        def raise_for_status(self):
            pass
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResp())
    client = GoogleScholarClient()
    with pytest.raises(GoogleScholarError):
        client.search("malformed")

def test_search_pagination_respects_max_results(mock_response):
    """
    Tests that the GoogleScholarClient returns no more than the specified max_results across paginated search results.
    """
    mock_response("google_scholar_search_sample.html")
    client = GoogleScholarClient()
    results = client.search("paging test", max_results=2)
    assert len(results) == 2