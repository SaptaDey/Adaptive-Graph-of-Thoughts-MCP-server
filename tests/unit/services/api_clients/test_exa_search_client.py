import json  # For converting string to dict for httpx_mock.add_response json parameter

import httpx  # For specific exceptions like httpx.ConnectError
import pytest
from pytest_httpx import HTTPXMock

from adaptive_graph_of_thoughts.config import ExaSearchConfig, Settings
from adaptive_graph_of_thoughts.infrastructure.api_clients.base_client import (
    APIHTTPError,
    APIRequestError,
)
from adaptive_graph_of_thoughts.infrastructure.api_clients.exa_search_client import (
    ExaSearchClient,
    ExaSearchClientError,
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
        A Settings object configured with a base URL and API key for the Exa Search API.
    """
    return Settings(
        exa_search=ExaSearchConfig(
            base_url="https://api.exa.ai", api_key="test_exa_api_key"
        )
    )


@pytest.fixture
async def exa_client_fixture(mock_exa_settings: Settings) -> ExaSearchClient:
    """
    Yields an `ExaSearchClient` instance configured with mock settings for use in asynchronous tests.
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
    with pytest.raises(
        ExaSearchClientError, match="Exa Search configuration is not properly set"
    ):
        ExaSearchClient(settings=Settings(exa_search=None))

    with pytest.raises(
        ExaSearchClientError, match="Exa Search configuration is not properly set"
    ):
        ExaSearchClient(
            settings=Settings(
                exa_search=ExaSearchConfig(api_key=None, base_url="https://api.exa.ai")
            )
        )  # type: ignore

    with pytest.raises(
        ExaSearchClientError, match="Exa Search configuration is not properly set"
    ):
        ExaSearchClient(
            settings=Settings(exa_search=ExaSearchConfig(api_key="key", base_url=None))
        )  # type: ignore


async def test_search_success(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the ExaSearchClient.search method returns parsed article results on a successful API response.

    Mocks a successful /search API call and verifies that the returned list of ExaArticleResult objects matches the expected data. Also checks that the API request uses the correct method, URL, and default payload parameters.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_SEARCH_SUCCESS_JSON_STR),
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
    assert payload["num_results"] == 10  # Default
    assert payload["type"] == "neural"  # Default


async def test_find_similar_success(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the find_similar method returns the expected article results for a valid source URL.

    Mocks a successful /find_similar API response and verifies that the returned articles match the mocked data. Also checks that the API call uses the correct HTTP method and payload.
    """
    client = exa_client_fixture
    source_url = "http://example.com/source_url_for_similar"
    httpx_mock.add_response(
        url=f"{client.config.base_url}/find_similar",
        method="POST",
        json=json.loads(SAMPLE_EXA_FIND_SIMILAR_SUCCESS_JSON_STR),
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


async def test_search_empty_results(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the search method returns an empty list when the API responds with no results.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )
    articles = await client.search("query yields empty")
    assert len(articles) == 0


async def test_search_missing_results_key(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock, caplog
):
    """
    Tests that the search method returns an empty list and logs a message when the API response is missing the 'results' key.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_MISSING_RESULTS_KEY_JSON_STR),
    )
    articles = await client.search("query missing key")
    assert len(articles) == 0
    assert "No 'results' key found in Exa API response." in caplog.text


async def test_search_http_error(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that an HTTP error during a search request raises ExaSearchClientError with the correct cause and error message.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        status_code=401,
        text="Unauthorized",
    )

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.search("query http error")
    assert isinstance(exc_info.value.__cause__, APIHTTPError)
    assert "Exa API search request failed" in str(exc_info.value)


async def test_search_request_error(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that a connection error during a search request raises ExaSearchClientError.

    Simulates a connection failure when calling the search method and verifies that the resulting exception is an ExaSearchClientError with an underlying cause of APIRequestError.
    """
    client = exa_client_fixture
    httpx_mock.add_exception(
        httpx.ConnectError("Simulated connection error"),
        url=f"{client.config.base_url}/search",
        method="POST",
    )

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.search("query request error")
    assert isinstance(exc_info.value.__cause__, APIRequestError)


async def test_search_with_all_parameters(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the search method correctly includes all optional parameters in the request payload when provided.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(
            SAMPLE_EXA_SEARCH_SUCCESS_JSON_STR
        ),  # Content doesn't matter, just need success
    )

    await client.search(
        query="test query",
        num_results=5,
        type="keyword",
        use_autoprompt=True,
        category="article",
        start_published_date="2023-01-01",
        end_published_date="2023-12-31",
    )

    request = httpx_mock.get_requests()[0]
    payload = json.loads(request.content)
    assert payload["query"] == "test query"
    assert payload["num_results"] == 5
    assert payload["type"] == "keyword"
    assert payload["use_autoprompt"] is True
    assert payload["category"] == "article"
    assert (
        payload["startPublishedDate"] == "2023-01-01"
    )  # Exa API expects camelCase for dates
    assert payload["endPublishedDate"] == "2023-12-31"


async def test_correct_headers_usage(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )
    await client.search("test headers")

    request = httpx_mock.get_requests()[0]
    assert request.headers["x-api-key"] == client.api_key
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Accept"] == "application/json"
    assert (
        "AdaptiveGraphOfThoughtsClient/1.0 (ExaSearchClient)"
        in request.headers["User-Agent"]
    )


async def test_find_similar_with_date_params(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the find_similar method includes date range parameters in the request payload.

    Verifies that start and end published dates are correctly serialized and sent when provided.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/find_similar",
        method="POST",
        json=json.loads(SAMPLE_EXA_FIND_SIMILAR_SUCCESS_JSON_STR),
    )
    await client.find_similar(
        url="http://example.com/test",
        num_results=3,
        start_published_date="2022-01-01",
        end_published_date="2022-12-31",
    )
    request = httpx_mock.get_requests()[0]
    payload = json.loads(request.content)
    assert payload["url"] == "http://example.com/test"
    assert payload["num_results"] == 3
    assert payload["startPublishedDate"] == "2022-01-01"
    assert payload["endPublishedDate"] == "2022-12-31"


# --- Additional Edge-Case & Failure Tests ---


@pytest.mark.asyncio
async def test_search_invalid_json_response(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the client raises ExaSearchClientError with a JSONDecodeError cause when the Exa API returns malformed JSON in a 200 OK response.
    """
    client = exa_client_fixture
    # note: httpx_mock will return bytes that cannot be parsed
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        text="{'invalid': 'json'}",  # invalid single quotes
    )

    with pytest.raises(ExaSearchClientError) as exc:
        await client.search("bad json")

    # Ensure json error is chained
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)
    assert "Invalid JSON received from Exa API" in str(exc.value)


@pytest.mark.asyncio
async def test_rate_limit_handling(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Exa returns 429 Too Many Requests - ensure specific message and APIHTTPError chaining.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        status_code=429,
        text="Rate limit exceeded",
    )

    with pytest.raises(ExaSearchClientError) as exc:
        await client.search("rate limited")

    cause = exc.value.__cause__
    assert isinstance(cause, APIHTTPError)
    assert "429" in str(cause)
    assert "rate limit" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_search_large_num_results(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the client correctly handles requests and responses when a large number of results is requested, ensuring all returned articles are parsed and the payload includes the correct `num_results` value.
    """
    client = exa_client_fixture
    # fabricate 50 result objects
    large_results = {
        "results": [
            {
                "id": f"rid_{i}",
                "url": f"http://ex.com/{i}",
                "title": f"Title {i}",
                "score": 0.5,
                "publishedDate": "2023-01-01",
                "author": "A",
                "highlights": [],
            }
            for i in range(50)
        ]
    }
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search", method="POST", json=large_results
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


# --- Additional Comprehensive Edge Case Tests ---


@pytest.mark.asyncio
async def test_search_malformed_article_data(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock, caplog
):
    """
    Tests that the client gracefully handles malformed article data in API responses.
    """
    client = exa_client_fixture
    malformed_response = {
        "results": [
            {
                "id": "valid_article",
                "url": "http://example.com/valid",
                "title": "Valid Article",
                "score": 0.9,
                "publishedDate": "2023-01-01",
                "author": "Author",
                "highlights": ["Highlight"]
            },
            {
                # Missing required fields
                "id": "malformed_article",
                "score": "invalid_score_type"  # Wrong type
            },
            {
                "id": "partial_article",
                "url": "http://example.com/partial",
                "title": None,  # Null title
                "score": 0.8
            }
        ]
    }
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=malformed_response,
    )

    articles = await client.search("malformed data test")
    
    # Should return only valid articles, skipping malformed ones
    assert len(articles) >= 1  # At least the valid one should be parsed
    valid_article = next((a for a in articles if a.id == "valid_article"), None)
    assert valid_article is not None
    assert "Error creating ExaArticleResult" in caplog.text


@pytest.mark.asyncio
async def test_search_with_special_characters(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the client properly handles queries with special characters and Unicode.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    special_queries = [
        "query with spaces and symbols !@#$%^&*()",
        "unicode characters: café naïve résumé",
        'quotes "double" and \'single\'',
        "numbers 12345 and symbols <>&",
        "newlines\nand\ttabs",
        "   whitespace query   ",  # Test query stripping
    ]

    for query in special_queries:
        articles = await client.search(query)
        assert isinstance(articles, list)
        
        request = httpx_mock.get_requests()[-1]
        payload = json.loads(request.content)
        # Query should be stripped
        assert payload["query"] == query.strip()


@pytest.mark.asyncio
async def test_search_with_boundary_parameters(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests boundary conditions for search parameters.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    # Test minimum and maximum num_results
    await client.search("boundary test", num_results=1)
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["num_results"] == 1

    await client.search("boundary test", num_results=1000)
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["num_results"] == 1000


@pytest.mark.asyncio
async def test_search_timeout_handling(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that timeout errors are properly handled and wrapped in ExaSearchClientError.
    """
    client = exa_client_fixture
    httpx_mock.add_exception(
        httpx.TimeoutException("Request timed out"),
        url=f"{client.config.base_url}/search",
        method="POST",
    )

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.search("timeout test")
    
    assert isinstance(exc_info.value.__cause__, APIRequestError)
    assert "timeout" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_api_response_with_additional_fields(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the client handles API responses with additional unknown fields gracefully.
    """
    client = exa_client_fixture
    response_with_extras = {
        "results": [
            {
                "id": "extra_fields_article",
                "url": "http://example.com/extra",
                "title": "Article with Extra Fields",
                "score": 0.95,
                "publishedDate": "2023-01-01",
                "author": "Author",
                "highlights": ["Highlight"],
                "extra_field_1": "should be ignored",
                "nested_extra": {"key": "value"},
                "future_feature": True
            }
        ],
        "extra_response_field": "ignored",
        "metadata": {"version": "v2"}
    }
    
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=response_with_extras,
    )

    articles = await client.search("extra fields test")
    
    assert len(articles) == 1
    article = articles[0]
    assert article.id == "extra_fields_article"
    assert article.title == "Article with Extra Fields"


@pytest.mark.asyncio
async def test_search_with_date_edge_cases(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests search with various date format edge cases.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    # Test with same start and end date
    await client.search(
        "date edge test",
        start_published_date="2023-06-15",
        end_published_date="2023-06-15"
    )
    
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["startPublishedDate"] == "2023-06-15"
    assert payload["endPublishedDate"] == "2023-06-15"


@pytest.mark.asyncio
async def test_search_with_all_search_types(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that all supported search types work correctly.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    search_types = ["neural", "keyword"]
    
    for search_type in search_types:
        await client.search("test query", type=search_type)
        
        request = httpx_mock.get_requests()[-1]
        payload = json.loads(request.content)
        assert payload["type"] == search_type


@pytest.mark.asyncio
async def test_search_with_all_categories(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests search with different category values.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    categories = ["article", "news", "research", "tweet", "company"]
    
    for category in categories:
        await client.search("category test", category=category)
        
        request = httpx_mock.get_requests()[-1]
        payload = json.loads(request.content)
        assert payload["category"] == category


@pytest.mark.asyncio
async def test_autoprompt_parameter_variations(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests the use_autoprompt parameter with different boolean values.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    # Test with True
    await client.search("autoprompt test", use_autoprompt=True)
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["use_autoprompt"] is True

    # Test with False
    await client.search("autoprompt test", use_autoprompt=False)
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["use_autoprompt"] is False


@pytest.mark.asyncio
async def test_response_with_missing_optional_fields(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests handling of articles with missing optional fields like author, highlights, etc.
    """
    client = exa_client_fixture
    minimal_response = {
        "results": [
            {
                "id": "minimal_article",
                "url": "http://example.com/minimal",
                "title": "Minimal Article",
                "score": 0.8,
                # Missing: publishedDate, author, highlights
            }
        ]
    }
    
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=minimal_response,
    )

    articles = await client.search("minimal fields test")
    
    assert len(articles) == 1
    article = articles[0]
    assert article.id == "minimal_article"
    assert article.url == "http://example.com/minimal"
    assert article.title == "Minimal Article"
    assert article.score == 0.8
    assert article.published_date == ""  # Default from model
    assert article.author == ""  # Default from model
    assert article.highlights == []  # Default from model


@pytest.mark.asyncio
async def test_very_long_query_handling(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that very long queries are handled correctly.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    # Create a very long query (>1000 characters)
    long_query = "AI artificial intelligence machine learning " * 50
    
    articles = await client.search(long_query)
    assert isinstance(articles, list)
    
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["query"] == long_query.strip()
    assert len(payload["query"]) > 1000


@pytest.mark.asyncio
async def test_concurrent_requests_handling(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that concurrent requests to the client work correctly.
    """
    import asyncio
    
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    # Make multiple concurrent requests
    tasks = [
        client.search(f"concurrent query {i}")
        for i in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    for result in results:
        assert isinstance(result, list)
    
    # Verify all requests were made
    assert len(httpx_mock.get_requests()) == 5


@pytest.mark.asyncio
async def test_client_reuse_after_error(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that the client can be reused after encountering an error.
    """
    client = exa_client_fixture
    
    # First request - error
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        status_code=500,
        text="Internal Server Error",
    )
    
    with pytest.raises(ExaSearchClientError):
        await client.search("error query")
    
    # Second request - success
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_SEARCH_SUCCESS_JSON_STR),
    )
    
    articles = await client.search("success query")
    assert len(articles) == 2


@pytest.mark.asyncio
async def test_api_key_in_headers_not_logged(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock, caplog
):
    """
    Tests that API keys are not inadvertently logged in error messages or debug output.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        status_code=401,
        text="Unauthorized - Invalid API Key",
    )
    
    with pytest.raises(ExaSearchClientError):
        await client.search("api key test")
    
    # Ensure API key is not in any log messages
    for record in caplog.records:
        assert "test_exa_api_key" not in record.getMessage()


@pytest.mark.asyncio
async def test_response_with_null_values(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests handling of API responses containing null values in various fields.
    """
    client = exa_client_fixture
    response_with_nulls = {
        "results": [
            {
                "id": "null_values_article",
                "url": "http://example.com/null",
                "title": "Article with Nulls",
                "score": 0.7,
                "publishedDate": None,
                "author": None,
                "highlights": None
            }
        ]
    }
    
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=response_with_nulls,
    )

    articles = await client.search("null values test")
    
    assert len(articles) == 1
    article = articles[0]
    assert article.id == "null_values_article"
    # Pydantic should handle None values according to field defaults
    assert article.published_date == "" or article.published_date is None
    assert article.author == "" or article.author is None


def test_client_config_property_access(mock_exa_settings: Settings):
    """
    Tests that client configuration properties are accessible and correct.
    """
    client = ExaSearchClient(settings=mock_exa_settings)
    
    assert client.config.base_url == "https://api.exa.ai"
    assert client.config.api_key == "test_exa_api_key"
    assert client.api_key == "test_exa_api_key"


@pytest.mark.asyncio
async def test_find_similar_http_error_handling(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that HTTP errors during find_similar requests are properly handled.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/find_similar",
        method="POST",
        status_code=404,
        text="Not Found",
    )

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.find_similar("http://nonexistent.com")
    
    assert isinstance(exc_info.value.__cause__, APIHTTPError)
    assert "Exa API find_similar request failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_with_empty_string_parameters(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests behavior when optional string parameters are empty strings.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    await client.search(
        "empty strings test",
        category="",
        start_published_date="",
        end_published_date=""
    )
    
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    
    # Empty strings should still be included in payload
    assert payload.get("category") == ""
    assert payload.get("startPublishedDate") == ""
    assert payload.get("endPublishedDate") == ""


@pytest.mark.asyncio
async def test_response_parsing_with_unicode_content(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests that Unicode content in API responses is properly parsed and handled.
    """
    client = exa_client_fixture
    unicode_response = {
        "results": [
            {
                "id": "unicode_article",
                "url": "http://example.com/unicode",
                "title": "Article with Unicode: café naïve résumé 中文 한국어 العربية",
                "score": 0.9,
                "publishedDate": "2023-01-01",
                "author": "François García 李明",
                "highlights": ["Highlights with émojis 🚀 and symbols ±∞"]
            }
        ]
    }
    
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=unicode_response,
    )

    articles = await client.search("unicode test")
    
    assert len(articles) == 1
    article = articles[0]
    assert "café naïve résumé 中文 한국어 العربية" in article.title
    assert "François García 李明" == article.author
    assert "🚀" in article.highlights[0]


@pytest.mark.asyncio
async def test_empty_query_handling(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests behavior with empty query strings.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    # Test empty query
    articles = await client.search("")
    assert isinstance(articles, list)
    
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["query"] == ""  # Should be stripped empty string


@pytest.mark.asyncio
async def test_search_response_with_error_field(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock, caplog
):
    """
    Tests handling when API response contains an error field.
    """
    client = exa_client_fixture
    error_response = {
        "error": "Rate limit exceeded",
        "message": "Please try again later"
    }
    
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=error_response,
    )

    articles = await client.search("error field test")
    
    assert len(articles) == 0
    assert "Exa API returned an error: Rate limit exceeded" in caplog.text


@pytest.mark.asyncio
async def test_find_similar_empty_results(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests find_similar with empty results.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/find_similar",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    articles = await client.find_similar("http://example.com/test")
    assert len(articles) == 0


@pytest.mark.asyncio
async def test_search_with_zero_num_results(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests search with zero num_results parameter.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=json.loads(SAMPLE_EXA_EMPTY_RESULTS_JSON_STR),
    )

    articles = await client.search("zero results test", num_results=0)
    assert isinstance(articles, list)
    
    request = httpx_mock.get_requests()[-1]
    payload = json.loads(request.content)
    assert payload["num_results"] == 0


@pytest.mark.asyncio
async def test_response_with_empty_highlights_array(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests handling of results with explicitly empty highlights arrays.
    """
    client = exa_client_fixture
    response = {
        "results": [
            {
                "id": "no_highlights",
                "url": "http://example.com/test",
                "title": "Article without highlights",
                "score": 0.8,
                "highlights": []  # Explicitly empty
            }
        ]
    }
    
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=response,
    )

    articles = await client.search("empty highlights test")
    
    assert len(articles) == 1
    assert articles[0].highlights == []


@pytest.mark.asyncio
async def test_search_invalid_response_structure(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock
):
    """
    Tests handling of completely invalid response structures.
    """
    client = exa_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        text="Not JSON at all",
        headers={"content-type": "text/plain"}
    )

    with pytest.raises(ExaSearchClientError) as exc_info:
        await client.search("invalid response test")
    
    assert "JSON decode error" in str(exc_info.value) or "Invalid JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_article_with_missing_id_skipped(
    exa_client_fixture: ExaSearchClient, httpx_mock: HTTPXMock, caplog
):
    """
    Tests that articles without IDs are skipped during parsing.
    """
    client = exa_client_fixture
    response = {
        "results": [
            {
                "url": "http://example.com/no-id",
                "title": "Article without ID",
                "score": 0.8
                # Missing "id" field
            },
            {
                "id": "valid_article",
                "url": "http://example.com/valid",
                "title": "Valid Article",
                "score": 0.9
            }
        ]
    }
    
    httpx_mock.add_response(
        url=f"{client.config.base_url}/search",
        method="POST",
        json=response,
    )

    articles = await client.search("missing id test")
    
    assert len(articles) == 1  # Only the valid article
    assert articles[0].id == "valid_article"
    assert "Skipping Exa result due to missing ID" in caplog.text
