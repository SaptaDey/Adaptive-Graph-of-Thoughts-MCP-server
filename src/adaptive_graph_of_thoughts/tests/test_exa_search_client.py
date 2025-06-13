# The repository uses pytest as the test framework (contains import pytest and test_ functions).
import os
import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError

from src.adaptive_graph_of_thoughts.exa_search_client import (
    ExaSearchClient,
    ExaResult,
    ExaAuthenticationError,
)

@pytest.fixture
def mock_exa_response():
    """Representative JSON payload returned by the ExaSearch API."""
    return {
        "results": [
            {"url": "http://example.com/1", "title": "Title 1", "score": 0.9},
            {"url": "http://example.com/2", "title": "Title 2", "score": 0.8},
        ],
        "next_page_token": None,
    }

def test_happy_path_successful_query(mock_exa_response, monkeypatch):
    """Test that a successful query returns a list of ExaResult objects."""
    monkeypatch.setenv("EXA_API_KEY", "dummy_key")
    client = ExaSearchClient()
    with patch.object(ExaSearchClient, "_send_request", return_value=mock_exa_response) as mock_send:
        results = client.query("test query", num_results=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for idx, res in enumerate(results):
            assert isinstance(res, ExaResult)
            assert res.url == mock_exa_response["results"][idx]["url"]
            assert res.title == mock_exa_response["results"][idx]["title"]
            assert res.score == mock_exa_response["results"][idx]["score"]
        mock_send.assert_called_once()

def test_pagination_multiple_pages(monkeypatch):
    """Test that pagination aggregates results across multiple pages."""
    monkeypatch.setenv("EXA_API_KEY", "dummy_key")
    page1 = {
        "results": [{"url": "http://example.com/a", "title": "A", "score": 1.0}],
        "next_page_token": "token123",
    }
    page2 = {
        "results": [{"url": "http://example.com/b", "title": "B", "score": 0.5}],
        "next_page_token": None,
    }
    client = ExaSearchClient()
    with patch.object(ExaSearchClient, "_send_request", side_effect=[page1, page2]) as mock_send:
        results = client.query("pagination test")
        assert len(results) == 2
        assert results[0].url == "http://example.com/a"
        assert results[1].url == "http://example.com/b"
        assert mock_send.call_count == 2

def test_query_no_results(monkeypatch):
    """Test that an empty results array returns an empty list."""
    monkeypatch.setenv("EXA_API_KEY", "dummy_key")
    client = ExaSearchClient()
    with patch.object(ExaSearchClient, "_send_request", return_value={"results": [], "next_page_token": None}) as mock_send:
        results = client.query("no results")
        assert results == []
        mock_send.assert_called_once()

def test_invalid_api_key_raises(monkeypatch):
    """Test that an invalid API key raises ExaAuthenticationError."""
    monkeypatch.setenv("EXA_API_KEY", "invalid_key")
    client = ExaSearchClient()
    def _raise_401(*args, **kwargs):
        err = HTTPError("401 Client Error: Unauthorized")
        err.response = MagicMock(status_code=401)
        raise err
    with patch.object(ExaSearchClient, "_send_request", side_effect=_raise_401):
        with pytest.raises(ExaAuthenticationError):
            client.query("auth failure")

def test_invalid_parameter_type(monkeypatch):
    """Test that invalid parameter types raise ValueError before any HTTP call."""
    monkeypatch.setenv("EXA_API_KEY", "dummy_key")
    client = ExaSearchClient()
    with patch.object(ExaSearchClient, "_send_request") as mock_send:
        with pytest.raises(ValueError):
            client.query("bad params", num_results="ten")
        mock_send.assert_not_called()