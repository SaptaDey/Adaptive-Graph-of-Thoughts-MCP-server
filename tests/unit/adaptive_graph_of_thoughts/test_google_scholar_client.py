# Fixtures and helpers for mocking HTTP responses and loading sample HTML
import os
import requests
import pytest
from src.adaptive_graph_of_thoughts.google_scholar_client import (
    GoogleScholarClient,
    GoogleScholarError,
)

def load_fixture(filename):
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", filename
    )
    with open(fixture_path, encoding="utf-8") as f:
        return f.read()

@pytest.fixture
def mock_response(monkeypatch):
    """Mock successful HTTP response with sample HTML."""
    def _mock(html_file):
        html = load_fixture(html_file)
        class MockResponse:
            status_code = 200
            text = html
            def raise_for_status(self):
                pass
        monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())
    return _mock

@pytest.fixture
def mock_error_response(monkeypatch):
    """Mock HTTP error response."""
    def _mock(status_code):
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
    """Network timeouts should raise GoogleScholarError."""
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
    """Client should respect max_results across pages."""
    mock_response("google_scholar_search_sample.html")
    client = GoogleScholarClient()
    results = client.search("paging test", max_results=2)
    assert len(results) == 2