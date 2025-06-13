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
    Pytest fixture that mocks a successful HTTP GET request with sample HTML content.
    
    Returns:
        A function that, when called with a fixture filename, patches `requests.get`
        to return a mock response containing the HTML from the specified file.
    """
    def _mock(html_file):
        """
        Mocks the HTTP GET request to return the contents of a specified HTML fixture file.
        
        Args:
            html_file: The filename of the HTML fixture to use for the mock response.
        
        This function replaces `requests.get` with a lambda that returns a mock response object containing the loaded HTML content and a status code of 200.
        """
        html = load_fixture(html_file)
        class MockResponse:
            status_code = 200
            text = html
            def raise_for_status(self):
                """
                Does nothing; placeholder for the raise_for_status method in mock responses.
                """
                pass
        monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())
    return _mock

@pytest.fixture
def mock_error_response(monkeypatch):
    """
    Pytest fixture that mocks HTTP GET requests to simulate error responses.
    
    Returns a function that, when called with a status code, patches `requests.get`
    to return a mock response object that raises an HTTPError with the given status code.
    """
    def _mock(status_code):
        Creates a mock HTTP response with a specified status code that raises an HTTPError when `raise_for_status` is called.
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
    """
    Tests that invalid search queries raise a ValueError.
    
    Args:
    	query: The invalid query string to test.
    """
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
    """Network timeouts should raise GoogleScholarError."""
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout),
    )
    client = GoogleScholarClient()
    with pytest.raises(GoogleScholarError):
        client.search("timeout")

def test_search_malformed_response_raises(monkeypatch):
    """
    Tests that a malformed HTML response without expected markers causes GoogleScholarClient.search to raise a GoogleScholarError.
    """
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