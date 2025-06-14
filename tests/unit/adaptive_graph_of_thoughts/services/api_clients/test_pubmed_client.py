import pytest
import requests
from adaptive_graph_of_thoughts.services.api_clients.pubmed_client import PubMedClient, PublicationAPIError

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

SAMPLE_EFETCH_RESPONSE = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Test Article</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

@pytest.mark.parametrize("query, expected", [
    ("cancer", {"term": "cancer", "retmax": PubMedClient.DEFAULT_RETMAX}),
    ("", {"term": "", "retmax": PubMedClient.DEFAULT_RETMAX}),
])
def test_build_query_params(query, expected):
    client = PubMedClient()
    params = client._build_query(query)
    assert params["term"] == expected["term"]
    assert int(params["retmax"]) == expected["retmax"]

def test_search_happy_path(requests_mock):
    client = PubMedClient()
    url = f"{client.BASE_URL}/esearch.fcgi"
    requests_mock.get(url, json=SAMPLE_SEARCH_RESPONSE, status_code=200)
    result = client.search("cancer")
    assert result == ["12345"]

def test_search_empty_result(requests_mock):
    client = PubMedClient()
    url = f"{client.BASE_URL}/esearch.fcgi"
    requests_mock.get(url, json=SAMPLE_SEARCH_EMPTY_RESPONSE, status_code=200)
    result = client.search("nope")
    assert result == []

def test_search_non_200_status(requests_mock):
    client = PubMedClient()
    url = f"{client.BASE_URL}/esearch.fcgi"
    requests_mock.get(url, status_code=500, json={})
    with pytest.raises(PublicationAPIError):
        client.search("error")

def test_search_malformed_json(requests_mock):
    client = PubMedClient()
    url = f"{client.BASE_URL}/esearch.fcgi"
    requests_mock.get(url, text="not a json", status_code=200)
    with pytest.raises(PublicationAPIError):
        client.search("badjson")

def test_search_timeout(monkeypatch):
    client = PubMedClient()
    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout
    monkeypatch.setattr(client._session, "get", raise_timeout)
    with pytest.raises(PublicationAPIError):
        client.search("timeout")

def test_fetch_article_happy_path(requests_mock):
    client = PubMedClient()
    url = f"{client.BASE_URL}/efetch.fcgi"
    requests_mock.get(url, text=SAMPLE_EFETCH_RESPONSE, status_code=200)
    article = client.fetch_article("12345")
    assert article.id == "12345"
    assert article.title == "Test Article"

def test_fetch_article_404(requests_mock):
    client = PubMedClient()
    url = f"{client.BASE_URL}/efetch.fcgi"
    requests_mock.get(url, status_code=404)
    with pytest.raises(PublicationAPIError):
        client.fetch_article("99999")

def test_fetch_article_malformed_xml(requests_mock):
    client = PubMedClient()
    url = f"{client.BASE_URL}/efetch.fcgi"
    requests_mock.get(url, text="<invalid><xml>", status_code=200)
    with pytest.raises(PublicationAPIError):
        client.fetch_article("12345")

def test___all___exports():
    import adaptive_graph_of_thoughts.services.api_clients.pubmed_client as module
    assert "PubMedClient" in getattr(module, "__all__", [])