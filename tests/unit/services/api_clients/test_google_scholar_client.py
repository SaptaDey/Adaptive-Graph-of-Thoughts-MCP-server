import json  # For converting string to dict for httpx_mock.add_response json parameter

import httpx  # For specific exceptions like httpx.ConnectError
import pytest
from pytest_httpx import HTTPXMock

from adaptive_graph_of_thoughts.config import GoogleScholarConfig, Settings
from adaptive_graph_of_thoughts.infrastructure.api_clients.base_client import (
    APIHTTPError,
    APIRequestError,
)
from adaptive_graph_of_thoughts.infrastructure.api_clients.google_scholar_client import (
    GoogleScholarArticle,
    GoogleScholarClient,
    GoogleScholarClientError,
    UnexpectedResponseStructureError,
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
            api_key="test_gs_api_key",
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
    with pytest.raises(
        GoogleScholarClientError,
        match="Google Scholar configuration is not properly set",
    ):
        GoogleScholarClient(settings=Settings(google_scholar=None))

    with pytest.raises(
        GoogleScholarClientError,
        match="Google Scholar configuration is not properly set",
    ):
        GoogleScholarClient(
            settings=Settings(
                google_scholar=GoogleScholarConfig(
                    api_key=None, base_url="https://base.url"
                )
            )
        )  # type: ignore

    with pytest.raises(
        GoogleScholarClientError,
        match="Google Scholar configuration is not properly set",
    ):
        GoogleScholarClient(
            settings=Settings(
                google_scholar=GoogleScholarConfig(api_key="key", base_url=None)
            )
        )  # type: ignore


async def test_search_success(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock
):
    """
    Tests that the GoogleScholarClient.search method returns correctly parsed articles on a successful API response.

    Verifies that all article fields are extracted as expected and that the API request includes the correct parameters.
    """
    client = gs_client_fixture
    httpx_mock.add_response(
        url=client.config.base_url,  # As base_url itself is the full endpoint for SerpApi usually
        method="GET",
        json=json.loads(SAMPLE_GS_SEARCH_SUCCESS_JSON_STR),
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


async def test_search_empty_results(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock
):
    """
    Tests that the search method returns an empty list when the API response contains no articles.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR))

    articles = await client.search("query yields empty")
    assert len(articles) == 0


async def test_search_missing_organic_results_key(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock, caplog
):
    """
    Tests that the search method returns an empty list and logs a warning when the API response lacks the 'organic_results' key.
    """
    client = gs_client_fixture
    httpx_mock.add_response(
        json=json.loads(SAMPLE_GS_SEARCH_MISSING_ORGANIC_RESULTS_KEY_JSON_STR)
    )

    with pytest.raises(UnexpectedResponseStructureError):
        await client.search("query missing key")
    assert (
        "No 'organic_results' in SerpApi response for query 'missing key query' on engine 'google_scholar'"
        in caplog.text
    )


async def test_search_http_error(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock
):
    """
    Tests that the search method raises GoogleScholarClientError when an HTTP error occurs.

    Asserts that the underlying cause of the exception is an APIHTTPError.
    """
    client = gs_client_fixture
    httpx_mock.add_response(status_code=500, text="Server Error")

    with pytest.raises(GoogleScholarClientError) as exc_info:
        await client.search("query http error")
    assert isinstance(exc_info.value.__cause__, APIHTTPError)


async def test_search_request_error(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock
):
    """
    Tests that a connection error during a search raises GoogleScholarClientError with APIRequestError as the cause.
    """
    client = gs_client_fixture
    httpx_mock.add_exception(httpx.ConnectError("Simulated connection error"))

    with pytest.raises(GoogleScholarClientError) as exc_info:
        await client.search("query request error")
    assert isinstance(exc_info.value.__cause__, APIRequestError)


async def test_search_invalid_json_response(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock
):
    """
    Tests that the search method raises a GoogleScholarClientError when the API returns invalid JSON.
    """
    client = gs_client_fixture
    httpx_mock.add_response(text=SAMPLE_GS_SEARCH_INVALID_JSON_STR)

    with pytest.raises(
        GoogleScholarClientError, match="Google Scholar API JSON decode error"
    ):
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
        httpx_mock.add_response(
            json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR)
        )  # Return empty to simplify
        await client.search("test_api_key_query")

    request = httpx_mock.get_requests()[0]
    assert request.url.params["api_key"] == specific_api_key


async def test_parsing_various_author_formats(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock
):
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


async def test_parsing_cited_by_count(
    gs_client_fixture: GoogleScholarClient, httpx_mock: HTTPXMock, caplog
):
    """
    Tests that the client correctly parses the cited-by count from various formats in the API response.

    Verifies that valid integer values are parsed, invalid or missing values result in None, and appropriate log messages are generated for parsing failures.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_CITED_BY_VARIATIONS_JSON_STR))
    articles = await client.search("cited_by format query")

    assert articles[0].cited_by_count == 123
    assert articles[1].cited_by_count is None  # Invalid string "onetwothree"
    assert "Could not parse cited_by_count 'onetwothree' as int" in caplog.text
    assert articles[2].cited_by_count is None  # Missing 'total' key
    assert articles[3].cited_by_count is None  # No 'cited_by' or 'inline_links'


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
    expected_count = len(
        json.loads(SAMPLE_GS_SEARCH_SUCCESS_JSON_STR)["organic_results"]
    )
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
        title="T",
        link="L",
        snippet="S",
        authors="A",
        publication_info="P",
        cited_by_count=1,
        citation_link="C",
    )
    r = repr(article)
    assert "GoogleScholarArticle" in r and "T" in r
    # __str__ should reflect the representation as well
    s = str(article)
    assert "T" in s


# --- Additional Comprehensive Edge Cases and Error Handling Tests ---


async def test_search_with_special_characters_and_encoding(gs_client_fixture, httpx_mock):
    """
    Tests that the search method properly handles queries with special characters, unicode, and encoding issues.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR))
    
    special_queries = [
        "machine learning & AI",
        "résumé parsing algorithms",
        "量子计算 quantum computing",
        "search with \"quotes\" and 'apostrophes'",
        "query with %20 encoding issues",
        "newline\ncharacter\ttab",
        "",  # Empty string
        "   ",  # Only whitespace
        "🤖 AI with emojis 🔬",
        "symbols ©™® and math ∑∫∂",
    ]
    
    for query in special_queries:
        articles = await client.search(query)
        assert isinstance(articles, list)
        
    # Verify all requests were made
    assert len(httpx_mock.get_requests()) == len(special_queries)


async def test_search_with_all_optional_parameters(gs_client_fixture, httpx_mock):
    """
    Tests that the search method correctly passes all optional parameters to the API.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR))
    
    await client.search(
        query="comprehensive query",
        num_results=25,
        lang="fr",
        region="ca"
    )
    
    request = httpx_mock.get_requests()[0]
    assert request.url.params["num"] == "25"
    assert request.url.params["hl"] == "fr"
    assert request.url.params["gl"] == "ca"
    assert request.url.params["q"] == "comprehensive query"
    assert request.url.params["engine"] == "google_scholar"


async def test_search_malformed_publication_info_structures(gs_client_fixture, httpx_mock):
    """
    Tests that the client handles malformed or missing publication_info gracefully.
    """
    malformed_response = {
        "organic_results": [
            {"title": "Paper 1", "publication_info": None},
            {"title": "Paper 2", "publication_info": {"summary": None}},
            {"title": "Paper 3", "publication_info": {"authors": []}},
            {"title": "Paper 4", "publication_info": {"summary": "", "authors": []}},
            {"title": "Paper 5"},  # Missing publication_info entirely
            {"title": "Paper 6", "publication_info": {"authors": [{"invalid": "structure"}]}},
            {"title": "Paper 7", "publication_info": {"authors": "string instead of list"}},
        ]
    }
    
    client = gs_client_fixture
    httpx_mock.add_response(json=malformed_response)
    
    articles = await client.search("malformed publication info")
    
    assert len(articles) == 7
    for article in articles:
        assert article.title is not None
        # All should handle missing/malformed publication_info gracefully
        assert article.publication_info is None or isinstance(article.publication_info, str)
        assert article.authors is None or isinstance(article.authors, str)


async def test_search_malformed_inline_links_data(gs_client_fixture, httpx_mock):
    """
    Tests that the client handles malformed or missing inline_links data structures robustly.
    """
    malformed_response = {
        "organic_results": [
            {"title": "Paper 1", "inline_links": None},
            {"title": "Paper 2", "inline_links": {}},
            {"title": "Paper 3", "inline_links": {"cited_by": None}},
            {"title": "Paper 4", "inline_links": {"cited_by": {"total": None}}},
            {"title": "Paper 5", "inline_links": {"serpapi_cite_link": None}},
            {"title": "Paper 6", "inline_links": {"versions": {"link": ""}, "cited_by": {"total": -1}}},
            {"title": "Paper 7", "inline_links": {"cited_by": {"total": "invalid_number"}}},
            {"title": "Paper 8", "inline_links": {"cited_by": {}}},  # Missing total key
        ]
    }
    
    client = gs_client_fixture
    httpx_mock.add_response(json=malformed_response)
    
    articles = await client.search("malformed inline links")
    
    assert len(articles) == 8
    for article in articles:
        assert article.title is not None
        # Should handle all malformed cases gracefully
        assert article.cited_by_count is None or isinstance(article.cited_by_count, int)
        assert article.citation_link is None or isinstance(article.citation_link, str)
        assert article.versions_link is None or isinstance(article.versions_link, str)


async def test_search_response_with_missing_titles(gs_client_fixture, httpx_mock, caplog):
    """
    Tests that the client skips articles with missing titles and logs warnings appropriately.
    """
    response_with_missing_titles = {
        "organic_results": [
            {"title": "Valid Paper 1", "link": "http://example1.com"},
            {"title": None, "link": "http://example2.com"},  # Should be skipped
            {"title": "", "link": "http://example3.com"},  # Empty title
            {"link": "http://example4.com"},  # Missing title entirely
            {"title": "Valid Paper 2", "link": "http://example5.com"},
        ]
    }
    
    client = gs_client_fixture
    httpx_mock.add_response(json=response_with_missing_titles)
    
    articles = await client.search("missing titles query")
    
    # Should only include articles with valid titles
    assert len(articles) <= 3  # At most 3 valid articles
    
    # Check that warning was logged for skipped articles
    assert "Skipping a Google Scholar result due to missing title" in caplog.text
    
    # Verify all returned articles have titles
    for article in articles:
        assert article.title is not None
        assert len(article.title.strip()) > 0


async def test_client_context_manager_lifecycle(mock_gs_settings):
    """
    Tests the complete lifecycle of the client context manager including error scenarios.
    """
    client = GoogleScholarClient(settings=mock_gs_settings)
    
    # Test normal context manager usage
    async with client as c:
        assert c is not None
        assert hasattr(c, 'search')
        assert c is client  # Should return self
    
    # Test that client can be reused after context exit
    async with client as c:
        assert c is not None
    
    # Test context manager with exception
    try:
        async with client as c:
            raise ValueError("Test exception")
    except ValueError:
        pass  # Expected
    
    # Client should still work after exception
    async with client:
        pass


async def test_search_with_various_response_encodings(gs_client_fixture, httpx_mock):
    """
    Tests that the client handles various text encoding issues in API responses.
    """
    response_with_encoding = {
        "organic_results": [
            {
                "title": "Résumé Analysis with Machine Lëarning",
                "link": "https://example.com/résumé",
                "snippet": "This paper discusses naïve approaches to résumé parsing using émojis 🤖 and symbols ©™®",
                "publication_info": {"summary": "Journál of Artificiál Intelligence, 2023"},
            },
            {
                "title": "中文标题 Chinese Title with 数学 Math",
                "snippet": "Abstract with unicode: α, β, γ mathematical symbols",
                "publication_info": {"summary": "会议论文集, 2023"},
            }
        ]
    }
    
    client = gs_client_fixture
    httpx_mock.add_response(json=response_with_encoding)
    
    articles = await client.search("encoding test")
    
    assert len(articles) == 2
    
    # Test first article with accented characters and emojis
    article1 = articles[0]
    assert "Résumé" in article1.title
    assert "🤖" in article1.snippet
    assert "©™®" in article1.snippet
    
    # Test second article with Chinese characters
    article2 = articles[1]
    assert "中文标题" in article2.title
    assert "α, β, γ" in article2.snippet


async def test_search_concurrent_requests_stress(gs_client_fixture, httpx_mock):
    """
    Tests that the client can handle multiple concurrent search requests without issues.
    """
    import asyncio
    
    client = gs_client_fixture
    
    # Add multiple responses for concurrent requests
    for i in range(10):
        httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_SUCCESS_JSON_STR))
    
    # Make concurrent requests
    tasks = [
        client.search(f"concurrent query {i}", num_results=5)
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All requests should succeed
    assert len(results) == 10
    for result in results:
        assert not isinstance(result, Exception)
        assert isinstance(result, list)
        assert len(result) == 2  # Based on SAMPLE_GS_SEARCH_SUCCESS_JSON_STR


async def test_search_with_none_and_empty_values_in_response(gs_client_fixture, httpx_mock):
    """
    Tests that the client handles None values and empty strings in various fields.
    """
    response_with_nones = {
        "organic_results": [
            {
                "title": "",  # Empty string
                "link": None,
                "snippet": None,
                "publication_info": {"summary": None, "authors": None},
                "inline_links": {"cited_by": {"total": None}, "serpapi_cite_link": None}
            },
            {
                "title": "Valid Title",
                "link": "",  # Empty string
                "snippet": "Valid snippet",
                "publication_info": None,
                "inline_links": None
            }
        ]
    }
    
    client = gs_client_fixture
    httpx_mock.add_response(json=response_with_nones)
    
    articles = await client.search("none values query")
    
    # Should handle empty/None values gracefully
    assert len(articles) >= 1  # At least one article should be valid
    
    for article in articles:
        # Check that None values are handled properly
        if article.title is not None:
            assert isinstance(article.title, str)
        if article.link is not None:
            assert isinstance(article.link, str)
        if article.snippet is not None:
            assert isinstance(article.snippet, str)


async def test_search_very_large_response_handling(gs_client_fixture, httpx_mock):
    """
    Tests that the client can handle very large API responses with many articles.
    """
    # Create a large response with many articles
    large_response = {
        "organic_results": [
            {
                "title": f"Paper {i}",
                "link": f"http://example.com/paper{i}",
                "snippet": f"This is snippet for paper {i}" * 10,  # Long snippets
                "publication_info": {"summary": f"Journal {i % 5}, 202{i % 4}"},
                "inline_links": {"cited_by": {"total": i * 10}}
            }
            for i in range(100)  # 100 articles
        ]
    }
    
    client = gs_client_fixture
    httpx_mock.add_response(json=large_response)
    
    articles = await client.search("large response query")
    
    assert len(articles) == 100
    
    # Verify that all articles are properly parsed
    for i, article in enumerate(articles):
        assert article.title == f"Paper {i}"
        assert article.link == f"http://example.com/paper{i}"
        assert article.cited_by_count == i * 10


async def test_search_with_zero_and_negative_parameters(gs_client_fixture, httpx_mock):
    """
    Tests that the client handles edge cases for numeric parameters.
    """
    client = gs_client_fixture
    
    # Add responses for each test case
    for _ in range(3):
        httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR))
    
    # Test with zero num_results
    articles1 = await client.search("zero num query", num_results=0)
    assert isinstance(articles1, list)
    
    # Test with negative num_results
    articles2 = await client.search("negative num query", num_results=-5)
    assert isinstance(articles2, list)
    
    # Test with very large num_results
    articles3 = await client.search("large num query", num_results=1000)
    assert isinstance(articles3, list)
    
    # Verify API calls were made with the parameters
    requests = httpx_mock.get_requests()
    assert len(requests) == 3
    assert requests[0].url.params["num"] == "0"
    assert requests[1].url.params["num"] == "-5"
    assert requests[2].url.params["num"] == "1000"


async def test_search_very_long_query_handling(gs_client_fixture, httpx_mock):
    """
    Tests that the client handles very long search queries appropriately.
    """
    client = gs_client_fixture
    httpx_mock.add_response(json=json.loads(SAMPLE_GS_SEARCH_EMPTY_RESULTS_JSON_STR))
    
    # Create a very long query (over 1000 characters)
    long_query = "machine learning artificial intelligence deep learning neural networks " * 20
    
    articles = await client.search(long_query)
    
    assert isinstance(articles, list)
    
    request = httpx_mock.get_requests()[0]
    assert request.url.params["q"] == long_query


def test_article_model_comprehensive_field_validation():
    """
    Tests comprehensive field validation and behavior of the GoogleScholarArticle model.
    """
    # Test with all fields provided
    article_full = GoogleScholarArticle(
        title="Test Title",
        link="http://test.com",
        snippet="Test snippet",
        authors="Test Authors",
        publication_info="Test Publication, 2023",
        cited_by_count=42,
        citation_link="http://cite.test.com",
        source="Google Scholar",
        related_articles_link="http://related.com",
        versions_link="http://versions.com",
        cited_by_link="http://citedby.com",
        raw_result="raw_data"
    )
    
    assert article_full.title == "Test Title"
    assert article_full.cited_by_count == 42
    assert article_full.source == "Google Scholar"
    
    # Test with minimal fields (using defaults)
    article_minimal = GoogleScholarArticle(title="Minimal Title")
    
    assert article_minimal.title == "Minimal Title"
    assert article_minimal.link == ""  # Default empty string
    assert article_minimal.cited_by_count == 0  # Default zero
    assert article_minimal.source == "Google Scholar"  # Default value
    
    # Test string representation
    repr_str = repr(article_full)
    assert "GoogleScholarArticle" in repr_str
    assert "Test Title" in repr_str


async def test_search_response_with_deeply_nested_structures(gs_client_fixture, httpx_mock):
    """
    Tests that the client handles API responses with complex nested structures gracefully.
    """
    complex_nested_response = {
        "organic_results": [
            {
                "title": "Complex Nested Paper",
                "link": "http://example.com/complex",
                "snippet": "Complex test snippet",
                "publication_info": {
                    "summary": "Advanced Journal of Complexity",
                    "authors": [
                        {
                            "name": "Author One",
                            "affiliations": [
                                {"institution": "University A", "department": "CS"},
                                {"institution": "University B", "department": "AI"}
                            ],
                            "author_id": "12345",
                            "verified": True
                        },
                        {
                            "name": "Author Two",
                            "extra_data": {
                                "h_index": 25,
                                "publications": 150,
                                "citations": {"total": 5000, "recent": 500}
                            }
                        }
                    ],
                    "venue": {
                        "name": "ICML 2023",
                        "type": "conference",
                        "impact_factor": 4.5,
                        "location": {"city": "Honolulu", "country": "USA"}
                    },
                    "publication_date": {"year": 2023, "month": 7, "day": 15}
                },
                "inline_links": {
                    "cited_by": {
                        "total": 25,
                        "link": "http://cited.com",
                        "breakdown": {
                            "by_year": {"2023": 10, "2024": 15},
                            "by_field": {"ML": 20, "AI": 5}
                        }
                    },
                    "versions": {
                        "cluster_id": "abcd1234",
                        "link": "http://versions.com",
                        "available_versions": [
                            {"type": "preprint", "source": "arXiv", "date": "2023-01-15"},
                            {"type": "published", "source": "journal", "date": "2023-07-15"}
                        ]
                    },
                    "serpapi_cite_link": "http://cite.com/complex",
                    "pdf_link": "http://pdf.example.com/paper.pdf",
                    "supplementary": {
                        "code": "http://github.com/author/code",
                        "data": "http://dataset.com/data",
                        "slides": "http://slides.com/presentation"
                    }
                },
                "metrics": {
                    "downloads": 1500,
                    "views": 5000,
                    "social_media": {
                        "twitter_mentions": 25,
                        "reddit_discussions": 5
                    }
                }
            }
        ],
        "search_metadata": {
            "id": "search_123",
            "status": "Success",
            "processed_at": "2024-01-15T10:30:00Z",
            "total_time_taken": 0.45
        },
        "search_information": {
            "query_displayed": "Complex Nested Paper",
            "total_results": "About 1,250,000 results",
            "results_state": "Fully displayed"
        }
    }
    
    client = gs_client_fixture
    httpx_mock.add_response(json=complex_nested_response)
    
    articles = await client.search("complex nested response")
    
    assert len(articles) == 1
    article = articles[0]
    
    # Verify that complex nested data is parsed correctly
    assert article.title == "Complex Nested Paper"
    assert article.authors == "Author One, Author Two"  # Should extract names from complex structure
    assert article.publication_info == "Advanced Journal of Complexity"
    assert article.cited_by_count == 25
    assert article.citation_link == "http://cite.com/complex"
    
    # Verify the raw result contains the full complex structure
    assert isinstance(article.raw_result, dict) or isinstance(article.raw_result, str)


