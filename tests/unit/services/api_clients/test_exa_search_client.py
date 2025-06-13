import pytest
from pytest_httpx import HTTPXMock
import json # For converting string to dict for httpx_mock.add_response json parameter
import httpx # For specific exceptions like httpx.ConnectError

from adaptive_graph_of_thoughts.config import Settings, ExaSearchConfig
from adaptive_graph_of_thoughts.services.api_clients.exa_search_client import (
    ExaSearchClient,
    ExaArticleResult,
    ExaSearchClientError,
)
from adaptive_graph_of_thoughts.services.api_clients.base_client import (
    APIHTTPError,
    APIRequestError,
)

# --- Sample Data ---
SAMPLE_EXA_SEARCH_SUCCESS_JSON_STR = """{
    "results": [
        {"id": "exa_res_1", "url": "http://example.com/exa1", "title": "Exa Result 1", "score": 0.9, "publishedDate": "2023-01-01", "author": "Exa Author 1", "highlights": ["Highlight 1 for result 1."]},
        {"id": "exa_res_2", "url": "http://example.com/exa2", "title": "Exa Result 2", "score": 0.85, "publishedDate": "2023-02-01", "author": "Exa Author 2", "highlights": ["Highlight A for result 2.", "Highlight B for result 2."]}
    ]
}"""

SAMPLE_EXA_FIND_SIMILAR_SUCCESS_JSON_STR = """{
    "results": [
        {"id": "exa_sim_1", "url": "http://example.com/similar1", "title": "Similar Exa Result 1", "score": 0.92, "publishedDate": "2023-03-01", "author": "Similar Author 1", "highlights": ["Highlight for similar 1."]}
    ]
}"""

SAMPLE_EXA_EMPTY_RESULTS_JSON_STR = """{
    "results": []
}"""

SAMPLE_EXA_MISSING_RESULTS_KEY_JSON_STR = """{
    "some_other_key": "some_value"
}"""

# --- Fixtures ---
@pytest.fixture
def mock_exa_settings() -> Settings:
    """
    Creates a Settings object with a minimal ExaSearchConfig for testing purposes.
    
    Returns:
        A Settings instance containing an ExaSearchConfig with a test base URL and API key.
    """
    return Settings(
        exa_search=ExaSearchConfig(
            base_url="https://api.exa.ai",
            api_key="test_exa_api_key"
        )
    )

@pytest.fixture
async def exa_client_fixture(mock_exa_settings: Settings) -> ExaSearchClient:
    """
    Provides an async fixture that yields an ExaSearchClient instance configured with mock settings.
    
    This fixture manages the client's lifecycle using an async context manager, ensuring proper resource cleanup after tests.
    """
    async with ExaSearchClient(settings=mock_exa_settings) as client:
        yield client

# --- Test Cases ---

def test_exa_client_initialization(mock_exa_settings: Settings):
    """Test successful initialization of ExaSearchClient."""
    client = ExaSearchClient(settings=mock_exa_settings)
    assert client is not None
    assert client.config == mock_exa_settings.exa_search
    assert client.api_key == "test_exa_api_key"
    # Check if default headers are set in the underlying HTTPX client
    assert client.http_client.client.headers["x-api-key"] == "test_exa_api_key"
    assert client.http_client.client.headers["Content-Type"] == "application/json"
    assert client.http_client.client.headers["Accept"] == "application/json"

def test_exa_client_initialization_missing_config():
    """
    Tests that ExaSearchClientError is raised if the Exa Search configuration is missing, or if the API key or base URL is not provided.
    """
    with pytest.raises(ExaSearchClientError, match="Exa Search configuration is not properly set"):
        ExaSearchClient(settings=Settings(exa_search=None))

    with pytest.raises(ExaSearchClientError, match="Exa Search configuration is not properly set"):
        ExaSearchClient(settings=Settings(exa_search=ExaSearchConfig(api_key=None, base_url="https://api.exa.ai"))) # type: ignore

    with pytest.raises(ExaSearchClientError, match="Exa Search configuration is not properly set"):
        ExaSearchClient(settings=Settings(exa_search=ExaSearchConfig(api_key="key", base_url=None))) # type: ignore

async def test_search_success(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that the `search` method returns a list of articles on a successful API response.
    
    Verifies that the returned articles have expected fields and that the outgoing request contains the correct payload and method.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_SEARCH_SUCCESS_JSON_STR)
    )

    articles = await client.search("sample query")

    assert len(articles) == 2
    article1 = articles[0]
    assert article1.id == "exa_res_1"
    assert article1.url == "http://example.com/exa1"
    assert article1.title == "Exa Result 1"
    assert article1.score == 0.9
    assert article1.published_date == "2023-01-01"
    assert article1.author == "Exa Author 1"
    assert article1.highlights == ["Highlight 1 for result 1."]

    # Verify API call
    request = httpx_mock.get_requests()[0]
    assert request.method == "POST"
    assert str(request.url) == f"{client.config.base_url}/search"
    payload = json.loads(request.content)
    assert payload["query"] == "sample query"
    assert payload["num_results"] == 10 # Default
    assert payload["type"] == "neural" # Default

async def test_find_similar_success(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that find_similar returns a list of ExaArticleResult objects for a valid source URL.
    
    Verifies that the API call is made with the correct payload and that the returned articles match expected fields from the mocked response.
    """
    client = exa_client_fixture
    source_url = "http://example.com/source_url_for_similar"
    httpx_mock.add_response(
        url=f"{client.config.base_url}/find_similar",
        method="POST",
        json=json.loads(SAMPLE_EXA_FIND_SIMILAR_SUCCESS_JSON_STR)
    )

    articles = await client.find_similar(source_url)

    assert len(articles) == 1
    article1 = articles[0]
    assert article1.id == "exa_sim_1"
    assert article1.url == "http://example.com/similar1"
    assert article1.title == "Similar Exa Result 1"
    assert article1.score == 0.92

    # Verify API call
    request = httpx_mock.get_requests()[0]
    assert request.method == "POST"
    payload = json.loads(request.content)
    assert payload["url"] == source_url

async def test_search_empty_results(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that the search method returns an empty list when the API responds with no results.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR)
    )
    articles = await client.search("query yields empty")
    assert len(articles) == 0

async def test_search_missing_results_key(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock, caplog):
    """
    Tests that the search method returns an empty list and logs a warning when the API response is missing the 'results' key.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_MISSING_RESULTS_KEY_JSON_STR)
    )
    articles = await client.search("query missing key")
    assert len(articles) == 0
    assert "No 'results' key found in Exa API response." in caplog.text

async def test_search_http_error(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that the search method raises ExaSearchClientError with an APIHTTPError cause when the API responds with an HTTP error status.
    """
    client = exa_client_fixture
    httpx_mock.add_response(url=f"{client.config.base_url}/search", method="POST", status_code=401, text="Unauthorized")

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.search("query http error")
    assert isinstance(exc_info.value.__cause__, APIHTTPError)
    assert "Exa API search request failed" in str(exc_info.value)

async def test_search_request_error(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that a connection error during a search request raises ExaSearchClientError with APIRequestError as the cause.
    """
    client = exa_client_fixture
    httpx_mock.add_exception(
        httpx.ConnectError("Simulated connection error"),
        url=f"{client.config.base_url}/search",
        method="POST"
    )

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.search("query request error")
    assert isinstance(exc_info.value.__cause__, APIRequestError)

async def test_search_with_all_parameters(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that the search method correctly includes all optional parameters in the request payload and uses the expected key formats.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_SEARCH_SUCCESS_JSON_STR) # Content doesn't matter, just need success
    )

    await client.search(
        query="test query",
        num_results=5,
        type="keyword",
        use_autoprompt=True,
        category="article",
        start_published_date="2023-01-01",
        end_published_date="2023-12-31"
    )

    request = httpx_mock.get_requests()[0]
    payload = json.loads(request.content)
    assert payload["query"] == "test query"
    assert payload["num_results"] == 5
    assert payload["type"] == "keyword"
    assert payload["use_autoprompt"] is True
    assert payload["category"] == "article"
    assert payload["startPublishedDate"] == "2023-01-01" # Exa API expects camelCase for dates
    assert payload["endPublishedDate"] == "2023-12-31"

async def test_correct_headers_usage(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR)
    )
    await client.search("test headers")

    request = httpx_mock.get_requests()[0]
    assert request.headers["x-api-key"] == client.api_key
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Accept"] == "application/json"
    assert "AdaptiveGraphOfThoughtsClient/1.0 (ExaSearchClient)" in request.headers["User-Agent"]

async def test_find_similar_with_date_params(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that find_similar sends date range parameters in the request payload.
    
    Verifies that the start and end published dates are correctly included in the payload when calling find_similar with date parameters.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/find_similar",
        method="POST",
        json=json.loads(SAMPLE_EXA_FIND_SIMILAR_SUCCESS_JSON_STR)
    )
    await client.find_similar(
        url="http://example.com/test",
        num_results=3,
        start_published_date="2022-01-01",
        end_published_date="2022-12-31"
    )
    request = httpx_mock.get_requests()[0]
    payload = json.loads(request.content)
    assert payload["url"] == "http://example.com/test"
    assert payload["num_results"] == 3
    assert payload["startPublishedDate"] == "2022-01-01"
    assert payload["endPublishedDate"] == "2022-12-31"
# --- Additional Edge-Case & Failure Tests ---

@pytest.mark.asyncio
async def test_search_invalid_json_response(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Exa returns 200 OK with malformed JSON – client should raise ExaSearchClientError
    whose __cause__ is a json.JSONDecodeError.
    """
    client = exa_client_fixture
    # note: httpx_mock will return bytes that cannot be parsed
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        text="{'invalid': 'json'}"  # invalid single quotes
    )

    with pytest.raises(ExaSearchClientError) as exc:
        await client.search("bad json")

    # Ensure json error is chained
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)
    assert "Invalid JSON received from Exa API" in str(exc.value)

@pytest.mark.asyncio
async def test_rate_limit_handling(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that a 429 Too Many Requests response from the Exa API causes ExaSearchClientError to be raised with an APIHTTPError as the cause and an appropriate rate limit message.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        status_code=429,
        text="Rate limit exceeded"
    )

    with pytest.raises(ExaSearchClientError) as exc:
        await client.search("rate limited")

    cause = exc.value.__cause__
    assert isinstance(cause, APIHTTPError)
    assert "429" in str(cause)
    assert "rate limit" in str(exc.value).lower()

@pytest.mark.asyncio
async def test_search_large_num_results(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that the client correctly handles requests and responses when a large number of results is specified, ensuring all returned articles are parsed and the payload includes the requested count.
    """
    client = exa_client_fixture
    # fabricate 50 result objects
    large_results = {"results": [
        {"id": f"rid_{i}", "url": f"http://ex.com/{i}", "title": f"Title {i}",
         "score": 0.5, "publishedDate": "2023-01-01", "author": "A", "highlights": []}
        for i in range(50)
    ]}
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=large_results
    )

    articles = await client.search("lots", num_results=50)
    assert len(articles) == 50
    assert articles[49].id == "rid_49"

    # payload verification
    payload = json.loads(httpx_mock.get_requests()[0].content)
    assert payload["num_results"] == 50

@pytest.mark.asyncio
async def test_find_similar_invalid_param_combo(exa_client_fixture: ExaSearchClient):
    """
    Passing both url and text should raise ExaSearchClientError (client-side validation).
    """
    client = exa_client_fixture
    with pytest.raises(ExaSearchClientError):
        await client.find_similar(url="http://e.com", text="should conflict")

@pytest.mark.asyncio
async def test_client_context_manager_closes_session(mock_exa_settings: Settings):
    """
    Ensure underlying httpx.AsyncClient is closed after exiting async context manager.
    """
    async with ExaSearchClient(settings=mock_exa_settings) as client:
        assert not client.http_client.client.is_closed
    # after context exit
    assert client.http_client.client.is_closed
