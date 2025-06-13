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
    Creates a Settings instance with a minimal ExaSearchConfig for testing purposes.
    
    Returns:
        A Settings object containing an ExaSearchConfig with a test base URL and API key.
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
    Asynchronously yields an ExaSearchClient instance configured with mock settings.
    
    Intended for use as a pytest fixture to provide a ready-to-use ExaSearchClient within test cases.
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
    """Test ExaSearchClientError if config (API key or base URL) is missing."""
    with pytest.raises(ExaSearchClientError, match="Exa Search configuration is not properly set"):
        ExaSearchClient(settings=Settings(exa_search=None))

    with pytest.raises(ExaSearchClientError, match="Exa Search configuration is not properly set"):
        ExaSearchClient(settings=Settings(exa_search=ExaSearchConfig(api_key=None, base_url="https://api.exa.ai"))) # type: ignore

    with pytest.raises(ExaSearchClientError, match="Exa Search configuration is not properly set"):
        ExaSearchClient(settings=Settings(exa_search=ExaSearchConfig(api_key="key", base_url=None))) # type: ignore

async def test_search_success(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that the search method returns a list of ExaArticleResult objects on success.
    
    Simulates a successful POST request to the /search endpoint and verifies that the returned articles have the expected fields and values. Also checks that the API request uses the correct method, URL, and default payload parameters.
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
    Tests that the find_similar method returns a list of ExaArticleResult objects when the API responds successfully.
    
    Verifies that the correct POST request is made to the /find_similar endpoint with the expected payload and that the returned article fields match the mocked response.
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
    Tests that the search method returns an empty list when the API response contains no results.
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
    Tests that the search method returns an empty list and logs a message when the API response lacks the 'results' key.
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
    Tests that an HTTP error during a search request raises ExaSearchClientError with APIHTTPError as the cause.
    """
    client = exa_client_fixture
    httpx_mock.add_response(url=f"{client.config.base_url}/search", method="POST", status_code=401, text="Unauthorized")

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.search("query http error")
    assert isinstance(exc_info.value.__cause__, APIHTTPError)
    assert "Exa API search request failed" in str(exc_info.value)

async def test_search_request_error(exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock):
    """
    Tests that ExaSearchClient raises ExaSearchClientError with an APIRequestError cause when a request-level connection error occurs during a search API call.
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
    Tests that the search API call includes all optional parameters in the request payload with correct keys and values.
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
    Tests that the find_similar API call includes date parameters and num_results in the request payload.
    
    Verifies that the payload sent to the /find_similar endpoint contains the correct URL, number of results, and properly formatted start and end published dates.
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
```
