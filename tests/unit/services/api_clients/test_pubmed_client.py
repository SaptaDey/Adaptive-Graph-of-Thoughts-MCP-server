import pytest
from pytest_httpx import HTTPXMock
import json # For loading JSON strings if needed, though direct dicts are often easier for responses

from adaptive_graph_of_thoughts.config import Settings, PubMedConfig
from adaptive_graph_of_thoughts.services.api_clients.pubmed_client import (
    PubMedClient,
    PubMedArticle,
    PubMedClientError,
)
from adaptive_graph_of_thoughts.services.api_clients.base_client import (
    APIHTTPError,
    APIRequestError,
)

# --- Sample Data ---
SAMPLE_ESEARCH_RESPONSE_SUCCESS_JSON = {
    "header": {"type": "esearch", "version": "2.0"},
    "esearchresult": {
        "count": "2", "retmax": "2", "retstart": "0",
        "idlist": ["123456", "789012"],
        "translationset": [],
        "querytranslation": "sample query[all fields]"
    }
}

SAMPLE_ESEARCH_SINGLE_ID_JSON = {
    "header": {"type": "esearch", "version": "2.0"},
    "esearchresult": {
        "count": "1", "retmax": "1", "retstart": "0",
        "idlist": ["123456"],
        "translationset": [],
        "querytranslation": "sample query[all fields]"
    }
}


SAMPLE_ESUMMARY_RESPONSE_XML_STR = """<eSummaryResult>
    <DocSum>
        <Id>123456</Id>
        <Item Name="Title" Type="String">Sample Article Title 1</Item>
        <Item Name="AuthorList" Type="List">
            <Item Name="Author" Type="String">Author A</Item>
            <Item Name="Author" Type="String">Author B</Item>
        </Item>
        <Item Name="Source" Type="String">Journal of Samples</Item>
        <Item Name="PubDate" Type="String">2023 Jan</Item>
        <Item Name="DOI" Type="String">10.1000/sample.doi.1</Item>
    </DocSum>
    <DocSum>
        <Id>789012</Id>
        <Item Name="Title" Type="String">Sample Article Title 2</Item>
        <Item Name="AuthorList" Type="List">
            <Item Name="Author" Type="String">Author C</Item>
        </Item>
        <Item Name="Source" Type="String">Another Journal</Item>
        <Item Name="PubDate" Type="String">2022 Dec</Item>
        <Item Name="DOI" Type="String">10.1000/sample.doi.2</Item>
    </DocSum>
</eSummaryResult>"""

SAMPLE_ESUMMARY_SINGLE_ARTICLE_XML_STR = """<eSummaryResult>
    <DocSum>
        <Id>123456</Id>
        <Item Name="Title" Type="String">Sample Article Title 1</Item>
        <Item Name="AuthorList" Type="List">
            <Item Name="Author" Type="String">Author A</Item>
            <Item Name="Author" Type="String">Author B</Item>
        </Item>
        <Item Name="Source" Type="String">Journal of Samples</Item>
        <Item Name="PubDate" Type="String">2023 Jan</Item>
        <Item Name="DOI" Type="String">10.1000/sample.doi.1</Item>
    </DocSum>
</eSummaryResult>"""


SAMPLE_EFETCH_ABSTRACT_XML_STR_123456 = """<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">123456</PMID>
            <Article PubModel="Print">
                <Abstract>
                    <AbstractText>This is abstract for PMID 123456.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>"""

SAMPLE_EFETCH_ABSTRACT_XML_STR_789012 = """<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">789012</PMID>
            <Article PubModel="Print">
                <Abstract>
                    <AbstractText>This is abstract for PMID 789012. It is longer.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>"""


SAMPLE_EFETCH_NO_ABSTRACT_XML_STR = """<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">999999</PMID>
            <Article PubModel="Print">
                <!-- No Abstract Node -->
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>"""

SAMPLE_EMPTY_ESEARCH_RESPONSE_JSON = {
    "header": {"type": "esearch", "version": "2.0"},
    "esearchresult": {"count": "0", "retmax": "0", "retstart": "0", "idlist": []}
}

# --- Fixtures ---
@pytest.fixture
def mock_settings() -> Settings:
    """
    Provides a minimal Settings instance configured for PubMed API testing.
    
    Returns:
        A Settings object with PubMedConfig containing a base URL and email.
    """
    return Settings(
        pubmed=PubMedConfig(
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            email="test@example.com" # Recommended by NCBI
        )
    )

@pytest.fixture
async def pubmed_client_fixture(mock_settings: Settings) -> PubMedClient:
    """
    Async fixture that provides a PubMedClient instance for use in tests.
    
    Yields:
        An initialized PubMedClient within an async context manager.
    """
    async with PubMedClient(settings=mock_settings) as client:
        yield client

# --- Test Cases ---

def test_pubmed_client_initialization(mock_settings: Settings):
    """Test successful initialization of PubMedClient."""
    client = PubMedClient(settings=mock_settings)
    assert client is not None
    assert client.config == mock_settings.pubmed
    assert client.email == "test@example.com"

def test_pubmed_client_initialization_missing_config():
    """Test PubMedClientError is raised if PubMed config is missing or incomplete."""
    with pytest.raises(PubMedClientError, match="PubMed configuration (base_url) is not set"):
        PubMedClient(settings=Settings(pubmed=None))

    with pytest.raises(PubMedClientError, match="PubMed configuration (base_url) is not set"):
        PubMedClient(settings=Settings(pubmed=PubMedConfig(base_url=None))) # type: ignore

async def test_search_articles_success(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that `search_articles` successfully retrieves, parses, and returns article metadata and abstracts for a valid PubMed query.
    
    Simulates the full workflow by mocking PubMed API responses for `esearch`, `esummary`, and `efetch` endpoints, and verifies that the resulting articles contain correct metadata and abstract text. Also checks that the correct API requests are made with expected parameters.
    """
    client = pubmed_client_fixture

    # Mock esearch
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        method="GET",
        json=SAMPLE_ESEARCH_RESPONSE_SUCCESS_JSON
    )
    # Mock esummary
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esummary.fcgi",
        method="GET",
        text=SAMPLE_ESUMMARY_RESPONSE_XML_STR
    )
    # Mock efetch for PMID 123456
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/efetch.fcgi?db=pubmed&id=123456&rettype=abstract&retmode=xml&email=test%40example.com",
        method="GET",
        text=SAMPLE_EFETCH_ABSTRACT_XML_STR_123456
    )
    # Mock efetch for PMID 789012
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/efetch.fcgi?db=pubmed&id=789012&rettype=abstract&retmode=xml&email=test%40example.com",
        method="GET",
        text=SAMPLE_EFETCH_ABSTRACT_XML_STR_789012
    )

    articles = await client.search_articles("sample query", max_results=2)

    assert len(articles) == 2

    article1 = next(a for a in articles if a.pmid == "123456")
    assert article1.title == "Sample Article Title 1"
    assert article1.authors == ["Author A", "Author B"]
    assert article1.journal == "Journal of Samples"
    assert article1.publication_date == "2023 Jan"
    assert article1.doi == "10.1000/sample.doi.1"
    assert article1.url == "https://pubmed.ncbi.nlm.nih.gov/123456/"
    assert article1.abstract == "This is abstract for PMID 123456."

    article2 = next(a for a in articles if a.pmid == "789012")
    assert article2.title == "Sample Article Title 2"
    assert article2.abstract == "This is abstract for PMID 789012. It is longer."

    # Verify calls
    esearch_req = httpx_mock.get_request(url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi")
    assert esearch_req is not None
    assert "term=sample+query" in str(esearch_req.url)
    assert "retmax=2" in str(esearch_req.url)

    esummary_req = httpx_mock.get_request(url=f"{client.config.base_url.rstrip('/')}/esummary.fcgi")
    assert esummary_req is not None
    assert "id=123456%2C789012" in str(esummary_req.url) # comma is %2C

    efetch_reqs = httpx_mock.get_requests(url__regex=r".*efetch\.fcgi.*")
    assert len(efetch_reqs) == 2


async def test_search_articles_no_pmids(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Verifies that `search_articles` returns an empty list when the eSearch endpoint yields no PMIDs and does not call eSummary or eFetch.
    """
    client = pubmed_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        json=SAMPLE_EMPTY_ESEARCH_RESPONSE_JSON
    )

    articles = await client.search_articles("query_with_no_results")
    assert len(articles) == 0

    # Ensure esummary and efetch were not called
    assert len(httpx_mock.get_requests(url__regex=r".*esummary\.fcgi.*")) == 0
    assert len(httpx_mock.get_requests(url__regex=r".*efetch\.fcgi.*")) == 0

async def test_search_articles_esearch_http_error(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that a PubMedClientError is raised when the eSearch endpoint returns an HTTP error.
    """
    client = pubmed_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        method="GET",
        status_code=500,
        text="Internal Server Error"
    )
    with pytest.raises(PubMedClientError, match="PubMed eSearch failed"):
        await client.search_articles("test query")

async def test_search_articles_esummary_http_error(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that a PubMedClientError is raised when the esummary endpoint returns an HTTP error during article search.
    """
    client = pubmed_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        json=SAMPLE_ESEARCH_SINGLE_ID_JSON # Success for esearch
    )
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esummary.fcgi",
        method="GET",
        status_code=400,
        text="Bad Request"
    )
    with pytest.raises(PubMedClientError, match="PubMed eSummary failed"):
        await client.search_articles("test query")

async def test_search_articles_efetch_http_error(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that PubMedClientError is raised when an HTTP error occurs during the efetch step of search_articles.
    """
    client = pubmed_client_fixture
    httpx_mock.add_response(url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi", json=SAMPLE_ESEARCH_SINGLE_ID_JSON)
    httpx_mock.add_response(url=f"{client.config.base_url.rstrip('/')}/esummary.fcgi", text=SAMPLE_ESUMMARY_SINGLE_ARTICLE_XML_STR)
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/efetch.fcgi", # Matches any efetch here
        method="GET",
        status_code=503,
        text="Service Unavailable"
    )

    # The error could be PubMedClientError wrapping APIHTTPError or a more specific message
    # Depending on how fetch_abstract propagates errors if one abstract fails among many.
    # The current implementation of search_articles might raise if any efetch fails.
    with pytest.raises(PubMedClientError, match=r"PubMed eFetch failed \(PMID 123456\)|Could not fetch abstract for PMID 123456"):
        await client.search_articles("test query")


async def test_fetch_abstract_direct_success(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """Test direct call to fetch_abstract successfully retrieves an abstract."""
    client = pubmed_client_fixture
    pmid = "123456"
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=xml&email=test%40example.com",
        text=SAMPLE_EFETCH_ABSTRACT_XML_STR_123456
    )
    abstract = await client.fetch_abstract(pmid)
    assert abstract == "This is abstract for PMID 123456."

async def test_fetch_abstract_direct_not_found(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """Test direct call to fetch_abstract when no abstract is found."""
    client = pubmed_client_fixture
    pmid = "999999"
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=xml&email=test%40example.com",
        text=SAMPLE_EFETCH_NO_ABSTRACT_XML_STR
    )
    abstract = await client.fetch_abstract(pmid)
    assert abstract is None

async def test_api_key_and_email_usage(httpx_mock: HTTPXMock):
    """
    Verifies that the PubMedClient includes the configured api_key and email in all PubMed API request parameters during article search.
    
    Ensures that the api_key and email are present in the parameters for esearch, esummary, and efetch endpoint requests when provided in the client settings.
    """
    settings_with_key = Settings(
        pubmed=PubMedConfig(
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            email="user_with_key@example.com",
            api_key="testapikey123"
        )
    )
    async with PubMedClient(settings=settings_with_key) as client:
        # Mock esearch, esummary, efetch to allow search_articles to complete
        httpx_mock.add_response(
            url__regex=r".*esearch\.fcgi.*", # Regex to catch any esearch call
            json=SAMPLE_ESEARCH_SINGLE_ID_JSON
        )
        httpx_mock.add_response(
            url__regex=r".*esummary\.fcgi.*",
            text=SAMPLE_ESUMMARY_SINGLE_ARTICLE_XML_STR
        )
        httpx_mock.add_response(
            url__regex=r".*efetch\.fcgi.*",
            text=SAMPLE_EFETCH_ABSTRACT_XML_STR_123456
        )

        await client.search_articles("test query for api key", max_results=1)

    # Check esearch call
    esearch_request = httpx_mock.get_request(url__regex=r".*esearch\.fcgi.*")
    assert esearch_request is not None
    request_params_esearch = esearch_request.url.params
    assert request_params_esearch.get("email") == "user_with_key@example.com"
    assert request_params_esearch.get("api_key") == "testapikey123"

    # Check esummary call
    esummary_request = httpx_mock.get_request(url__regex=r".*esummary\.fcgi.*")
    assert esummary_request is not None
    request_params_esummary = esummary_request.url.params
    assert request_params_esummary.get("email") == "user_with_key@example.com"
    assert request_params_esummary.get("api_key") == "testapikey123"

    # Check efetch call
    efetch_request = httpx_mock.get_request(url__regex=r".*efetch\.fcgi.*")
    assert efetch_request is not None
    request_params_efetch = efetch_request.url.params
    assert request_params_efetch.get("email") == "user_with_key@example.com"
    assert request_params_efetch.get("api_key") == "testapikey123"

async def test_search_articles_esearch_request_error(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that a network error during the eSearch API call results in a PubMedClientError.
    """
    client = pubmed_client_fixture
    httpx_mock.add_exception(
        pytest.raises(APIRequestError), # This is not how httpx_mock expects exceptions. It expects an exception instance.
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        method="GET"
    )
    # Correct way to mock an exception with httpx_mock:
    httpx_mock.add_exception(
        Exception("Simulated network error"), # General exception for simplicity
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        method="GET"
    )

    with pytest.raises(PubMedClientError, match="PubMed eSearch failed"):
        await client.search_articles("test query")

# Consider adding tests for invalid XML parsing in _parse_esummary_response
# and _parse_abstract_from_efetch_xml if robust error handling for malformed XML is critical.
# For instance, if ET.fromstring fails.
# The current client implementation catches ET.ParseError and re-raises as PubMedClientError.

async def test_parse_esummary_malformed_xml(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """Test handling of malformed XML from eSummary."""
    client = pubmed_client_fixture
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        json=SAMPLE_ESEARCH_SINGLE_ID_JSON
    )
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esummary.fcgi",
        text="<malformed_xml>" # Not well-formed XML
    )
    with pytest.raises(PubMedClientError, match="XML parsing error for eSummary"):
        await client.search_articles("test query")

async def test_fetch_abstract_malformed_xml(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that fetch_abstract raises PubMedClientError when eFetch returns malformed XML.
    """
    client = pubmed_client_fixture
    pmid = "123456"
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=xml&email=test%40example.com",
        text="<unclosedTag"
    )
    with pytest.raises(PubMedClientError, match=f"XML parsing error for eFetch \(PMID {pmid}\)"):
        await client.fetch_abstract(pmid)
await client.fetch_abstract(pmid)

# --- Additional Edge-Case Tests ---
import httpx

def test_pubmed_client_invalid_base_url():
    """Test initialization fails with invalid base_url."""
    bad_settings = Settings(pubmed=PubMedConfig(base_url="htt://bad url", email="x@y.com"))
    with pytest.raises(PubMedClientError, match="base_url"):
        PubMedClient(settings=bad_settings)

@pytest.mark.asyncio
async def test_search_articles_timeout_handling(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that a timeout during the eSearch API call raises a PubMedClientError.
    """
    client = pubmed_client_fixture
    esearch_url = f"{client.config.base_url.rstrip('/')}/esearch.fcgi"
    httpx_mock.add_exception(httpx.TimeoutException("timeout"), url=esearch_url)
    with pytest.raises(PubMedClientError, match="eSearch failed"):
        await client.search_articles("timeout query")

@pytest.mark.asyncio
async def test_search_articles_large_retmax(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """
    Tests that search_articles correctly passes a large max_results value as the retmax parameter and returns all expected articles when multiple PMIDs are found.
    """
    client = pubmed_client_fixture
    # Create esearch response with three IDs
    large_esearch = {
        "header": {"type": "esearch", "version": "2.0"},
        "esearchresult": {
            "count": "3", "retmax": "3", "retstart": "0",
            "idlist": ["1", "2", "3"],
            "translationset": [],
            "querytranslation": "q"
        }
    }
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esearch.fcgi",
        method="GET",
        json=large_esearch
    )
    # Mock esummary with minimal DocSum entries
    esummary_xml = "<eSummaryResult>" + "".join(f"<DocSum><Id>{pmid}</Id></DocSum>" for pmid in ["1", "2", "3"]) + "</eSummaryResult>"
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/esummary.fcgi",
        method="GET",
        text=esummary_xml
    )
    # Mock efetch for each PMID
    for pmid in ["1", "2", "3"]:
        efetch_xml = (
            "<PubmedArticleSet>"
            f"<PubmedArticle><MedlineCitation><Article><Abstract><AbstractText>Abstract {pmid}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticleSet>"
        )
        httpx_mock.add_response(
            url__regex=r".*efetch\.fcgi.*",
            method="GET",
            text=efetch_xml
        )
    articles = await client.search_articles("q", max_results=1000)
    assert len(articles) == 3
    esearch_req = httpx_mock.get_request(url__regex=r".*esearch\.fcgi.*")
    assert esearch_req is not None
    params = esearch_req.url.params
    assert params.get("retmax") == "1000"

@pytest.mark.parametrize(
    "xml_str, expected_titles, expected_authors",
    [
        (
            "<eSummaryResult><DocSum><Id>1</Id></DocSum></eSummaryResult>",
            [""],
            [[]],
        ),
        (
            "<eSummaryResult><DocSum><Id>1</Id><Item Name=\"Title\" Type=\"String\">Title</Item></DocSum></eSummaryResult>",
            ["Title"],
            [[]],
        ),
        (
            "<eSummaryResult><DocSum><Id>1</Id><Item Name=\"AuthorList\" Type=\"List\"></Item></DocSum></eSummaryResult>",
            [""],
            [[]],
        ),
    ],
)
def test_parse_esummary_response_edge_cases(xml_str, expected_titles, expected_authors):
    """
    Tests parsing of eSummary XML responses with edge cases such as missing or minimal fields.
    
    Verifies that the parsed article titles and author lists match the expected values for each edge case.
    """
    client = PubMedClient(settings=Settings(pubmed=PubMedConfig(base_url="http://url", email="e@e.com")))
    articles = client._parse_esummary_response(xml_str)
    titles = [a.title for a in articles]
    authors = [a.authors for a in articles]
    assert titles == expected_titles
    assert authors == expected_authors

@pytest.mark.asyncio
async def test_fetch_abstract_multiple_articles_in_set(pubmed_client_fixture: PubMedClient, httpx_mock: HTTPXMock):
    """Test fetch_abstract returns abstract of the first article when multiple are present."""
    client = pubmed_client_fixture
    combined_xml = (
        SAMPLE_EFETCH_ABSTRACT_XML_STR_123456.replace("</PubmedArticleSet>", "")
        + SAMPLE_EFETCH_ABSTRACT_XML_STR_789012.replace("<PubmedArticleSet>", "")
    )
    pmid = "123456"
    httpx_mock.add_response(
        url=f"{client.config.base_url.rstrip('/')}/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=xml&email=test%40example.com",
        method="GET",
        text=combined_xml
    )
    abstract = await client.fetch_abstract(pmid)
    # Based on implementation, should return the first AbstractText encountered
    assert abstract == "This is abstract for PMID 123456."
