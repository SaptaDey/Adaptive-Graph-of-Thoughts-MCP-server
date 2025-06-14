import pytest
from pytest_httpx import HTTPXMock
import json # For converting string to dict for httpx_mock.add_response json parameter
import httpx # For specific exceptions like httpx.ConnectError

from adaptive_graph_of_thoughts.config import Settings, GoogleScholarConfig
from adaptive_graph_of_thoughts.services.api_clients.google_scholar_client import (
    GoogleScholarClient,
    GoogleScholarArticle,
    GoogleScholarClientError,
)
from adaptive_graph_of_thoughts.services.api_clients.base_client import (
    APIHTTPError,
    APIRequestError,
)

# --- Sample Data ---
SAMPLE_GS_SEARCH_SUCCESS_JSON_STR = """{
    "search_metadata": {"status": "Success"},
    "search_parameters": {"q": "sample query", "engine": "google_scholar"},
    "organic_results": [
        {
            "title": "Sample Paper 1", "link": "http://example.com/paper1", "snippet": "Snippet for paper 1.",
            "publication_info": {"summary": "Journal A, 2020", "authors": [{"name": "Author X"}, {"name": "Author Y"}]},
            "inline_links": {
                "cited_by": {"total": 10, "link": "http://example.com/citedby1"},
                "versions": {"link": "http://example.com/versions1"},
                "serpapi_cite_link": "http://example.com/cite1"
            }
        },
        {
            "title": "Sample Paper 2", "link": "http://example.com/paper2", "snippet": "Snippet for paper 2.",
            "publication_info": {"summary": "Conference B, 2021", "authors": "Author Z, Author W"},
            "inline_links": {"cited_by": {"total": 5}}
        }
    ]
}"""

SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR = """{
    "search_metadata": {"status": "Success"},
    "search_parameters": {"q": "empty query", "engine": "google_scholar"},
    "organic_results": []
}"""

SAMPLE_GS_SEARCH_MISSING_ORGANIC_RESULTS_KEY_JSON_STR = """{
    "search_metadata": {"status": "Success"},
    "search_parameters": {"q": "missing key query", "engine": "google_scholar"},
    "some_other_key": []
}"""

SAMPLE_GS_SEARCH_INVALID_JSON_STR = "This is not valid JSON."

SAMPLE_GS_AUTHORS_VARIATIONS_JSON_STR = """{
    "organic_results": [
        {"title": "Paper A", "publication_info": {"authors": [{"name": "First A."}, {"name": "Second B."}]}},
        {"title": "Paper B", "publication_info": {"authors": "Single Author String"}},
        {"title": "Paper C", "publication_info": {}},
        {"title": "Paper D"}
    ]
}"""

SAMPLE_GS_CITED_BY_VARIATIONS_JSON_STR = """{
    "organic_results": [
        {"title": "Cited Paper Valid", "inline_links": {"cited_by": {"total": 123}}},
        {"title": "Cited Paper Invalid", "inline_links": {"cited_by": {"total": "onetwothree"}}},
        {"title": "Cited Paper Missing Total", "inline_links": {"cited_by": {}}},
        {"title": "Cited Paper No Links"}
    ]
}"""


# --- Fixtures ---
@pytest.fixture
def mock_gs_settings() -> Settings:
    """
    Creates a Settings instance with a minimal GoogleScholarConfig for testing purposes.
    
    Returns:
        A Settings object configured with a base URL and API key for Google Scholar.
    """
    return Settings(
        google_scholar=GoogleScholarConfig(
            base_url="https://serpapi.com/search",  # Example base URL
            api_key="test_gs_api_key"
        )
    )

@pytest.fixture
async def gs_client_fixture(mock_gs_settings: Settings) -> GoogleScholarClient:
    """
    Provides an asynchronous fixture that yields a GoogleScholarClient instance configured with mock settings for use in tests.
    """
    async with GoogleScholarClient(settings=mock_gs_settings) as client:
        yield client

# --- Test Cases ---

def test_gs_client_initialization(mock_gs_settings: Settings):
    """Test successful initialization of GoogleScholarClient."""
    client = GoogleScholarClient(settings=mock_gs_settings)
    assert client is not None
    assert client.config == mock_gs_settings.google_scholar
    assert client.api_key == "test_gs_api_key"

def test_gs_client_initialization_missing_config():
    """
    Verifies that GoogleScholarClient raises GoogleScholarClientError when required configuration (API key or base URL) is missing.
    """
    with pytest.raises(GoogleScholarClientError, match="Google Scholar configuration is not properly set"):
        GoogleScholarClient(settings=Settings(google_scholar=None))

    with pytest.raises(GoogleScholarClientError, match="Google Scholar configuration is not properly set"):
        GoogleScholarClient(settings=Settings(google_scholar=GoogleScholarConfig(api_key=None, base_url="https://base.url"))) # type: ignore

    with pytest.raises(GoogleScholarClientError, match="Google Scholar configuration is not properly set"):
        GoogleScholarClient(settings=Settings(google_scholar=GoogleScholarConfig(api_key="key", base_url=None))) # type: ignore


async def test_search_success(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock):
    """
    Tests that the GoogleScholarClient.search method returns correctly parsed articles on a successful API response.
    
    Verifies that all article fields are extracted as expected and that the API request includes the correct parameters.
    """
    client = gs_client_fixture
    httpx_mock.add_response(
        url=client.config.base_url, # As base_url itself is the full endpoint for SerpApi usually
        method="GET",
        json=json.loads(SAMPLE_GS_SEARCH_SUCCESS_JSON_STR)
    )

    articles = await client.search("sample query")

    assert len(articles) == 2
    article1 = articles[0]
    assert article1.title == "Sample Paper 1"
    assert article1.link == "http://example.com/paper1"
    assert article1.snippet == "Snippet for paper 1."
    assert article1.authors == "Author X, Author Y"
    assert article1.publication_info == "Journal A, 2020"
    assert article1.cited_by_count == 10
    assert article1.citation_link == "http://example.com/cite1"

    # Verify API call parameters
    request = httpx_mock.get_requests()[0]
    assert request.url.params["api_key"] == client.api_key
    assert request.url.params["engine"] == "google_scholar"
    assert request.url.params["q"] == "sample query"

async def test_search_empty_results(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock):
    """
    Tests that the search method returns an empty list when the API response contains no articles.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR))

    articles = await client.search("query yields empty")
    assert len(articles) == 0

async def test_search_missing_organic_results_key(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock, caplog):
    """
    Tests that the search method returns an empty list and logs a warning when the API response lacks the 'organic_results' key.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_MISSING_ORGANIC_RESULTS_KEY_JSON_STR))

    articles = await client.search("query missing key")
    assert len(articles) == 0
    assert "No 'organic_results' in SerpApi response for query 'missing key query' on engine 'google_scholar'" in caplog.text


async def test_search_http_error(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock):
    """
    Tests that the search method raises GoogleScholarClientError when an HTTP error occurs.
    
    Asserts that the underlying cause of the exception is an APIHTTPError.
    """
    client = gs_client_fixture
    httpx_mock.add_response(status_code=500, text="Server Error")

    with pytest.raises(GoogleScholarClientError) as exc_info:
        await client.search("query http error")
    assert isinstance(exc_info.value.__cause__, APIHTTPError)

async def test_search_request_error(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock):
    """
    Tests that a connection error during a search raises GoogleScholarClientError with APIRequestError as the cause.
    """
    client = gs_client_fixture
    httpx_mock.add_exception(httpx.ConnectError("Simulated connection error"))

    with pytest.raises(GoogleScholarClientError) as exc_info:
        await client.search("query request error")
    assert isinstance(exc_info.value.__cause__, APIRequestError)


async def test_search_invalid_json_response(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock):
    """
    Tests that the search method raises a GoogleScholarClientError when the API returns invalid JSON.
    """
    client = gs_client_fixture
    httpx_mock.add_response(text=SAMPLE_GS_SEARCH_INVALID_JSON_STR)

    with pytest.raises(GoogleScholarClientError, match="Google Scholar API JSON decode error"):
        await client.search("query invalid json")

async def test_correct_api_key_usage(httpx_mock: HTTPXMock):
    """
    Verifies that the GoogleScholarClient uses the correct API key from its configuration when making a search request.
    """
    specific_api_key = "specific_test_api_key_12345"
    settings = Settings(
        google_scholar=GoogleScholarConfig(
            base_url="https://serpapi.com/search", api_key=specific_api_key
        )
    )
    async with GoogleScholarClient(settings=settings) as client:
        httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR)) # Return empty to simplify
        await client.search("test_api_key_query")

    request = httpx_mock.get_requests()[0]
    assert request.url.params["api_key"] == specific_api_key

async def test_parsing_various_author_formats(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock):
    """
    Tests that the GoogleScholarClient correctly parses author information from various formats in the API response, including lists of dictionaries, strings, and missing values.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_AUTHORS_VARIATIONS_JSON_STR))
    articles = await client.search("author format query")

    assert articles[0].authors == "First A., Second B."
    assert articles[1].authors == "Single Author String"
    assert articles[2].authors is None
    assert articles[3].authors is None

async def test_parsing_cited_by_count(gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock, caplog):
    """
    Tests that the client correctly parses the cited-by count from various formats in the API response.
    
    Verifies that valid integer values are parsed, invalid or missing values result in None, and appropriate log messages are generated for parsing failures.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_CITED_BY_VARIATIONS_JSON_STR))
    articles = await client.search("cited_by format query")

    assert articles[0].cited_by_count == 123
    assert articles[1].cited_by_count is None # Invalid string "onetwothree"
    assert "Could not parse cited_by_count 'onetwothree' as int" in caplog.text
    assert articles[2].cited_by_count is None # Missing 'total' key
    assert articles[3].cited_by_count is None # No 'cited_by' or 'inline_links'

# Fixture for timeout exception
@pytest.fixture
def timeout_exc() -> httpx.ReadTimeout:
    """
    Creates an httpx.ReadTimeout exception instance for simulating read timeouts in tests.
    
    Returns:
        An httpx.ReadTimeout exception with a preset message.
    """
    return httpx.ReadTimeout("read timeout")

async def test_search_pagination(gs_client_fixture, httpx_mock):
    """
    Tests that the search method handles pagination by making multiple requests when the number of requested results exceeds a single page, and aggregates articles from all pages.
    """
    client = gs_client_fixture
    page1 = json.loads(SAMPLE_GS_SEARCH_SUCCESS_JSON_STR)
    page2 = json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR)
    httpx_mock.add_response(json=page1)
    httpx_mock.add_response(json=page2)
    articles = await client.search("sample query", num=20)
    # Two pages should be fetched
    assert len(httpx_mock.get_requests()) == 2
    assert len(articles) >= 2

async def test_search_timeout(gs_client_fixture, httpx_mock, timeout_exc):
    """
    Tests that a read timeout during a search request raises a GoogleScholarClientError caused by APIRequestError.
    """
    client = gs_client_fixture
    httpx_mock.add_exception(timeout_exc)
    with pytest.raises(GoogleScholarClientError) as exc:
        await client.search("timeout query")
    assert isinstance(exc.value.__cause__, APIRequestError)

async def test_search_ignores_unexpected_keys(gs_client_fixture, httpx_mock):
    """
    Tests that the search method ignores unexpected keys in the API response and parses articles correctly.
    """
    client = gs_client_fixture
    page = json.loads(SAMPLE_GS_SEARCH_SUCCESS_JSON_STR)
    page["unexpected_key"] = {"foo": "bar"}
    httpx_mock.add_response(json=page)
    articles = await client.search("query with extra keys")
    # Unexpected keys should be ignored and parsing succeed
    expected_count = len(json.loads(SAMPLE_GS_SEARCH_SUCCESS_JSON_STR)["organic_results"])
    assert len(articles) == expected_count

async def test_search_parameter_propagation(gs_client_fixture, httpx_mock):
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR))
    await client.search("param query", num=50, hl="es")
    request = httpx_mock.get_requests()[-1]
    assert request.url.params["num"] == "50"
    assert request.url.params["hl"] == "es"

def test_article_repr():
    article = GoogleScholarArticle(
        title="T", link="L", snippet="S", authors="A",
        publication_info="P", cited_by_count=1, citation_link="C"
    )
    r = repr(article)
    assert "GoogleScholarArticle" in r and "T" in r
    # __str__ should reflect the representation as well
    s = str(article)
    assert "T" in s