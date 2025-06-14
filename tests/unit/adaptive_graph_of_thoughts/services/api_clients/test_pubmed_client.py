import pytest
import re # Import re for regex
from httpx import Request, Response, TimeoutException # Correct import for httpx
from unittest.mock import MagicMock

from adaptive_graph_of_thoughts.services.api_clients.pubmed_client import (
    PubMedClient,
    PubMedClientError,
    PubMedArticle,
)
from adaptive_graph_of_thoughts.config import Config, PubMedConfig

# Sample fixtures for search and fetch responses
SAMPLE_SEARCH_RESPONSE = {
    "esearchresult": {
        "idlist": ["12345"],
        "count": "1"
    }
}

SAMPLE_SEARCH_EMPTY_RESPONSE = {
    "esearchresult": {
        "idlist": [],
        "count": "0"
    }
}

SAMPLE_ESUMMARY_RESPONSE_XML = """
<eSummaryResult>
    <DocSum>
        <Id>12345</Id>
        <Item Name="Title" Type="String">Test Article</Item>
        <Item Name="AuthorList" Type="List">
            <Item Name="Author" Type="String">Author One</Item>
            <Item Name="Author" Type="String">Author Two</Item>
        </Item>
        <Item Name="DOI" Type="String">10.1234/test.doi</Item>
    </DocSum>
</eSummaryResult>
"""

SAMPLE_ABSTRACT_XML_RESPONSE = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Test Article with Abstract</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">This is the background.</AbstractText>
          <AbstractText>This is an unstructured part of the abstract.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Test for _build_query was removed as the method seems to be internal / refactored away.

@pytest.mark.asyncio
async def test_search_happy_path(httpx_mock):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com"))
    client = PubMedClient(settings=mock_settings)

    escaped_base_url = re.escape(client.config.base_url)
    esearch_url_pattern = re.compile(f"{escaped_base_url}/esearch.fcgi\\?.*term=cancer.*")
    esummary_url_pattern = re.compile(f"{escaped_base_url}/esummary.fcgi\\?.*id=12345.*")
    efetch_abstract_url_pattern = re.compile(f"{escaped_base_url}/efetch.fcgi\\?.*id=12345.*rettype=abstract.*")

    httpx_mock.add_response(url=esearch_url_pattern, json=SAMPLE_SEARCH_RESPONSE, status_code=200, method="GET")
    httpx_mock.add_response(url=esummary_url_pattern, text=SAMPLE_ESUMMARY_RESPONSE_XML, status_code=200, method="GET")

    minimal_abstract_xml_for_search = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <Abstract>
              <AbstractText>This is a test abstract.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    httpx_mock.add_response(url=efetch_abstract_url_pattern, text=minimal_abstract_xml_for_search, status_code=200, method="GET")

    articles = await client.search_articles(query="cancer")
    assert len(articles) == 1
    article = articles[0]
    assert article.pmid == "12345"
    assert article.title == "Test Article"
    assert article.authors == ["Author One", "Author Two"]
    assert article.doi == "10.1234/test.doi"
    assert article.abstract == "This is a test abstract."


@pytest.mark.asyncio
async def test_search_empty_result(httpx_mock):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com"))
    client = PubMedClient(settings=mock_settings)
    url_pattern = re.compile(f"{re.escape(client.config.base_url)}/esearch.fcgi\\?.*term=nope.*")
    httpx_mock.add_response(url=url_pattern, json=SAMPLE_SEARCH_EMPTY_RESPONSE, status_code=200, method="GET")
    articles = await client.search_articles(query="nope")
    assert articles == []

@pytest.mark.asyncio
async def test_search_non_200_status(httpx_mock):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com"))
    client = PubMedClient(settings=mock_settings)
    url_pattern = re.compile(f"{re.escape(client.config.base_url)}/esearch.fcgi\\?.*term=error.*")
    httpx_mock.add_response(url=url_pattern, status_code=500, json={}, method="GET")
    with pytest.raises(PubMedClientError):
        await client.search_articles(query="error")

@pytest.mark.asyncio
async def test_search_malformed_json(httpx_mock):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com"))
    client = PubMedClient(settings=mock_settings)
    url_pattern = re.compile(f"{re.escape(client.config.base_url)}/esearch.fcgi\\?.*term=badjson.*")
    httpx_mock.add_response(url=url_pattern, text="not a json", status_code=200, method="GET")
    with pytest.raises(PubMedClientError):
        await client.search_articles(query="badjson")

@pytest.mark.asyncio
async def test_search_timeout(httpx_mock, monkeypatch):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com", timeout_seconds=0.1))
    client = PubMedClient(settings=mock_settings)

    async def raise_timeout_async(*args, **kwargs):
        req = kwargs.get('request')
        if req is None:
            req = Request("GET", "http://dummy.url/timeout")
        raise TimeoutException("Simulated timeout", request=req)

    monkeypatch.setattr(client.http_client, "get", raise_timeout_async)

    with pytest.raises(PubMedClientError):
        await client.search_articles(query="timeout")


@pytest.mark.asyncio
async def test_fetch_article_happy_path(httpx_mock):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com"))
    client = PubMedClient(settings=mock_settings)
    pmid = "12345"
    url_pattern = re.compile(f"{re.escape(client.config.base_url)}/efetch.fcgi\\?.*id={pmid}.*rettype=abstract.*")
    httpx_mock.add_response(
        url=url_pattern,
        text=SAMPLE_ABSTRACT_XML_RESPONSE,
        status_code=200,
        method="GET"
    )
    article_abstract = await client.fetch_abstract(pmid)
    assert article_abstract is not None
    assert "BACKGROUND: This is the background." in article_abstract
    assert "This is an unstructured part of the abstract." in article_abstract
    assert "Test Article with Abstract" not in article_abstract

@pytest.mark.asyncio
async def test_fetch_article_404(httpx_mock):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com"))
    client = PubMedClient(settings=mock_settings)
    pmid = "99999"
    url_pattern = re.compile(f"{re.escape(client.config.base_url)}/efetch.fcgi\\?.*id={pmid}.*")
    httpx_mock.add_response(
        url=url_pattern,
        status_code=404,
        method="GET"
    )
    with pytest.raises(PubMedClientError):
        await client.fetch_abstract(pmid)

@pytest.mark.asyncio
async def test_fetch_article_malformed_xml(httpx_mock):
    mock_settings = Config(pubmed=PubMedConfig(base_url="http://dummy.url", email="test@example.com"))
    client = PubMedClient(settings=mock_settings)
    pmid = "12345"
    url_pattern = re.compile(f"{re.escape(client.config.base_url)}/efetch.fcgi\\?.*id={pmid}.*")
    httpx_mock.add_response(
        url=url_pattern,
        text="<invalid><xml>",
        status_code=200,
        method="GET"
    )
    with pytest.raises(PubMedClientError):
        await client.fetch_abstract(pmid)

def test___all___exports():
    import adaptive_graph_of_thoughts.services.api_clients.pubmed_client as module
    assert "PubMedClient" in getattr(module, "__all__", [])
    assert "PubMedArticle" in getattr(module, "__all__", [])
    assert "PubMedClientError" in getattr(module, "__all__", [])

# Note: The original SAMPLE_EFETCH_RESPONSE was removed by commenting out as it was structurally more like an eSummary result.
# The tests now use SAMPLE_ESUMMARY_RESPONSE_XML for esummary calls and
# SAMPLE_ABSTRACT_XML_RESPONSE (or minimal_abstract_xml_for_search for internal abstract fetching) for efetch (abstract) calls.
# """
# SAMPLE_EFETCH_RESPONSE = """
# <PubmedArticleSet>
#   <PubmedArticle>
#     <MedlineCitation>
#       <PMID>12345</PMID>
#       <Article>
#         <ArticleTitle>Test Article</ArticleTitle>
#       </Article>
#     </MedlineCitation>
#   </PubmedArticle>
# </PubmedArticleSet>
# """
