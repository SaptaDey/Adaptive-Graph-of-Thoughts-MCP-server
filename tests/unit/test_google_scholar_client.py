"""Unit tests for GoogleScholarClient using pytest, pytest_asyncio, and pytest_httpx.

This module tests:
- search() happy path with valid organic_results
- search() with no organic_results
- search() handling SerpAPI error payload
- search() propagating transport errors as GoogleScholarClientError
- search() handling malformed JSON responses
- internal parsing logic normalization for various author and cited_by_count formats

HTTP interactions are mocked via pytest_httpx's HTTPXMock fixture. Monkeypatching is used to simulate lower-level client errors.
"""

import pytest, asyncio, json
from types import SimpleNamespace
from loguru import logger
import httpx
from pytest_httpx import HTTPXMock
from adaptive_graph_of_thoughts.config import Settings, GoogleScholarConfig
from adaptive_graph_of_thoughts.clients.google_scholar_client import (
    GoogleScholarClient,
    GoogleScholarClientError,
    GoogleScholarArticle,
)
from adaptive_graph_of_thoughts.clients.http_client import APIHTTPError

@pytest.fixture
def settings_with_dummy_key():
    """Provide Settings with a dummy Google Scholar API key and base URL."""
    return Settings(
        google_scholar=GoogleScholarConfig(
            api_key="DUMMY_KEY",
            base_url="https://serpapi.com"
        )
    )

@pytest.fixture
async def gs_client(settings_with_dummy_key):
    """Instantiate a GoogleScholarClient and ensure it is closed after tests."""
    client = GoogleScholarClient(settings_with_dummy_key.google_scholar)
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_search_happy_path(gs_client, httpx_mock: HTTPXMock):
    """When organic_results contains one valid entry, search() returns one article with correct fields."""
    good_payload = {
        "organic_results": [
            {
                "title": "Some title",
                "authors": [{"name": "Alice"}, {"name": "Bob"}],
                "publication_info": {},
                "cited_by": {"value": 42},
                "link": "https://example.com/article"
            }
        ]
    }
    httpx_mock.add_response(json=good_payload)
    results = await gs_client.search("quantum", num_results=5)
    assert len(results) == 1
    art = results[0]
    assert isinstance(art, GoogleScholarArticle)
    assert art.title == "Some title"
    assert art.source == "Google Scholar"
    assert art.authors == ["Alice", "Bob"]
    assert isinstance(art.cited_by_count, int) and art.cited_by_count == 42

@pytest.mark.asyncio
async def test_search_no_organic_results(gs_client, httpx_mock: HTTPXMock):
    """An empty organic_results array yields an empty list."""
    httpx_mock.add_response(json={"organic_results": []})
    results = await gs_client.search("nohits")
    assert results == []

@pytest.mark.asyncio
async def test_search_serpapi_error_field(gs_client, httpx_mock: HTTPXMock):
    """If SerpAPI returns an error field, search() swallows it and returns an empty list."""
    httpx_mock.add_response(json={
        "error": "Engine unavailable",
        "search_parameters": {"q": "foo", "engine": "google_scholar"}
    })
    results = await gs_client.search("foo")
    assert results == []

@pytest.mark.asyncio
async def test_search_http_error(gs_client, monkeypatch):
    """Transport-level HTTP errors (APIHTTPError) are propagated as GoogleScholarClientError."""
    async def _raise(*args, **kwargs):
        raise APIHTTPError("boom")
    monkeypatch.setattr(gs_client.http_client, "get", _raise)
    with pytest.raises(GoogleScholarClientError):
        await gs_client.search("fail")

@pytest.mark.asyncio
async def test_search_malformed_json_error(gs_client, monkeypatch):
    """A malformed JSON payload (ValueError) raises GoogleScholarClientError."""
    async def _fake(*args, **kwargs):
        return SimpleNamespace(
            status_code=200,
            json=lambda: (_ for _ in ()).throw(ValueError("bad"))
        )
    monkeypatch.setattr(gs_client.http_client, "get", _fake)
    with pytest.raises(GoogleScholarClientError):
        await gs_client.search("malformed")

@pytest.mark.parametrize(
    "raw_result, expected_authors, expected_cited_by",
    [
        # authors as comma-separated string, cited_by_count as string
        (
            {"title": "T1", "authors": "Alice, Bob", "cited_by_count": "5"},
            ["Alice", "Bob"],
            5
        ),
        # authors as list of strings, cited_by_count as int
        (
            {"title": "T2", "authors": ["Carol", "Dave"], "cited_by_count": 10},
            ["Carol", "Dave"],
            10
        ),
        # missing both fields defaults to empty list and zero
        (
            {"title": "T3"},
            [],
            0
        ),
    ]
)
def test_parse_serpapi_response_variants(raw_result, expected_authors, expected_cited_by):
    """Internal parsing handles string/list authors and string/int cited_by_count formats."""
    data = {"organic_results": [raw_result]}
    articles = GoogleScholarClient._parse_serpapi_response(data)
    assert len(articles) == 1
    art = articles[0]
    assert art.authors == expected_authors
    assert art.cited_by_count == expected_cited_by