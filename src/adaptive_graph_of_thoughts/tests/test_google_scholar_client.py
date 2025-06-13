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
    
    Returns:
        A function that, when called with a fixture filename, sets up the mock response for subsequent HTTP requests.
    """
    def _mock(html_file):
        """
        Mocks the `requests.get` method to return a successful HTTP response with HTML content from a fixture file.
        
        Args:
            html_file: The filename of the HTML fixture to load and use as the response body.
        """
        html = load_fixture(html_file)
        class MockResponse:
            status_code = 200
            text = html
            def raise_for_status(self):
                """
                Does nothing when called; placeholder for the HTTP response's raise_for_status method in mocks.
                """
                pass
        monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())
    return _mock

@pytest.fixture
def mock_error_response(monkeypatch):
    """
    Pytest fixture that mocks `requests.get` to return an HTTP response with a specified error status code.
    
    Returns:
        A function that, when called with a status code, sets up the mock to return a response whose `raise_for_status` method raises a `requests.HTTPError` with the corresponding status code.
    """
    def _mock(status_code):
        """
        Creates a mock HTTP response with a specified status code that raises an HTTPError when checked.
        
        The mock response's `raise_for_status` method raises a `requests.HTTPError` with a message based on the provided status code.
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
    """
    Tests that the search method returns a list of articles with required fields for valid queries.
    
    Verifies that each result contains non-empty title and URL strings, and integer year and citations fields, and that the number of results matches the expected count.
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
    """Invalid queries should raise ValueError."""
    client = GoogleScholarClient()
    with pytest.raises(ValueError):
        client.search(query)

@pytest.mark.parametrize("status_code", [500, 502, 503])
def test_search_http_error_raises(mock_error_response, status_code):
    """
    Tests that HTTP error responses from the search method raise GoogleScholarError.
    
    Args:
        status_code: The HTTP status code to simulate in the mock response.
    """
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
    """
    Tests that a malformed HTML response from Google Scholar causes the search method to raise a GoogleScholarError.
    """
    class MockResp:
        status_code = 200
        text = "<html><body>no expected markers</body></html>"
        def raise_for_status(self):
            """
            Does nothing when called; used to mock the behavior of a successful HTTP response's status check.
            """
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