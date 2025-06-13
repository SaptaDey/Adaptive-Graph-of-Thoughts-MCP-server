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
        Mocks the HTTP GET request to return a predefined HTML response from a fixture file.
        
        Args:
            html_file: The filename of the HTML fixture to load and use as the mock response body.
        """
        html = load_fixture(html_file)
        class MockResponse:
            status_code = 200
            text = html
            def raise_for_status(self):
                """
                Does nothing; placeholder for the HTTP response's raise_for_status method in mocks.
                """
                pass
        monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())
    return _mock

@pytest.fixture
def mock_error_response(monkeypatch):
    """
    Pytest fixture that mocks HTTP GET requests to return a specified error status code.
    
    Returns:
        A function that, when called with a status code, patches `requests.get` to return a mock response object whose `raise_for_status` method raises a `requests.HTTPError` with the given status code.
    """
    def _mock(status_code):
        Creates and applies a mock for `requests.get` that returns a response with the specified HTTP status code and raises an HTTPError when `raise_for_status` is called.
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
    """
    Tests that the search method returns the expected number of result dictionaries with valid fields for a typical query.
    
    Args:
        mock_response: Fixture to mock the HTTP response with sample HTML content.
        query: The search query string.
        max_results: The maximum number of results to return.
        expected_count: The expected number of results in the response.
    """
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
        query: The invalid search query to test.
    """
    client = GoogleScholarClient()
    with pytest.raises(ValueError):
        client.search(query)

@pytest.mark.parametrize("status_code", [500, 502, 503])
def test_search_http_error_raises(mock_error_response, status_code):
    """
    Tests that HTTP error responses cause the search method to raise GoogleScholarError.
    
    Args:
        status_code: The HTTP status code to simulate in the mocked response.
    """
    mock_error_response(status_code)
    client = GoogleScholarClient()
    with pytest.raises(GoogleScholarError):
        client.search("test")

def test_search_timeout_raises(monkeypatch):
    """
    Tests that a network timeout during a search request raises a GoogleScholarError.
    """
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout),
    )
    client = GoogleScholarClient()
    with pytest.raises(GoogleScholarError):
        client.search("timeout")

def test_search_malformed_response_raises(monkeypatch):
    """
    Tests that a malformed HTML response from Google Scholar causes the search method to raise a GoogleScholarError.
    """
    class MockResp:
        status_code = 200
        text = "<html><body>no expected markers</body></html>"
        def raise_for_status(self):
            """
            Does nothing; placeholder for the HTTP response's status check method in mocks.
            """
            pass
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResp())
    client = GoogleScholarClient()
    with pytest.raises(GoogleScholarError):
        client.search("malformed")

def test_search_pagination_respects_max_results(mock_response):
    """Client should respect max_results across pages."""
    mock_response("google_scholar_search_sample.html")
    client = GoogleScholarClient()
    results = client.search("paging test", max_results=2)
    assert len(results) == 2