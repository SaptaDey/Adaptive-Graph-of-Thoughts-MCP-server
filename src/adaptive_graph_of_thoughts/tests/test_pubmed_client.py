import os
import pytest
from unittest.mock import MagicMock
import xml.etree.ElementTree as ET

from src.adaptive_graph_of_thoughts.pubmed_client import PubMedClient, PubMedClientError

@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Disable actual sleeping during retry backoff."""
    monkeypatch.setattr("time.sleep", lambda _duration: None)

def _mock_response(mocker, status=200, json_data=None, text=""):
    """Helper to create a mock HTTP response."""
    resp = mocker.Mock()
    resp.status_code = status
    resp.ok = status == 200
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp

@pytest.fixture
def client(mocker):
    """Create a PubMedClient with a mocked HTTP session."""
    session = mocker.Mock()
    return PubMedClient(session=session)

def test_search_success_single_page(client, mocker):
    """Given a small result set, search returns all IDs in one call."""
    fake_json = {"esearchresult": {"idlist": ["1", "2"], "count": "2"}}
    client.session.get.return_value = _mock_response(mocker, json_data=fake_json)

    ids = client.search("cancer", max_results=20)

    assert ids == ["1", "2"]
    expected_url = f"{client.BASE_URL}/esearch.fcgi"
    client.session.get.assert_called_once_with(
        expected_url, params=mocker.ANY, timeout=client.TIMEOUT
    )

def test_search_pagination_handles_multiple_pages(client, mocker):
    """When result count exceeds page size, search pages through results."""
    first = {"esearchresult": {"idlist": ["1", "2", "3"], "count": "5"}}
    second = {"esearchresult": {"idlist": ["4", "5"], "count": "5"}}
    client.session.get.side_effect = [
        _mock_response(mocker, json_data=first),
        _mock_response(mocker, json_data=second),
    ]

    ids = client.search("cancer", max_results=5, page_size=3)

    assert ids == ["1", "2", "3", "4", "5"]
    assert client.session.get.call_count == 2

def test_search_uses_api_key_if_present(client, monkeypatch, mocker):
    """Search should include the API key from environment if set."""
    monkeypatch.setenv("PUBMED_API_KEY", "SECRET")
    fake_json = {"esearchresult": {"idlist": ["1"], "count": "1"}}
    client.session.get.return_value = _mock_response(mocker, json_data=fake_json)

    client.search("diabetes", max_results=1)

    params = client.session.get.call_args.kwargs["params"]
    assert params.get("api_key") == "SECRET"

def test_retry_logic_on_transient_error(client, mocker):
    """Search should retry on 503 errors before succeeding."""
    client.session.get.side_effect = [
        _mock_response(mocker, status=503),
        _mock_response(mocker, status=503),
        _mock_response(mocker, json_data={"esearchresult": {"idlist": ["x"], "count": "1"}}),
    ]

    ids = client.search("retry-test", max_results=1)

    assert ids == ["x"]
    assert client.session.get.call_count == 3

@pytest.mark.parametrize("query", ["", "   "])
def test_search_empty_query_raises_value_error(client, query):
    """Providing an empty or whitespace-only query raises ValueError."""
    with pytest.raises(ValueError):
        client.search(query, max_results=1)

def test_search_malformed_json_raises_pubmed_client_error(client, mocker):
    """Invalid JSON structure in search response should raise PubMedClientError."""
    client.session.get.return_value = _mock_response(mocker, json_data={"foo": {}})

    with pytest.raises(PubMedClientError):
        client.search("cancer", max_results=1)

def test_get_article_parses_valid_xml(client, mocker):
    """get_article should return parsed XML Element for valid response."""
    xml = "<PubmedArticleSet><Article><Title>Test</Title></Article></PubmedArticleSet>"
    client.session.get.return_value = _mock_response(mocker, text=xml)

    article = client.get_article("1234")

    assert isinstance(article, ET.Element)
    assert article.find(".//Title").text == "Test"

def test_get_article_raises_on_http_error(client, mocker):
    """
    Tests that get_article raises PubMedClientError when the HTTP response status is not 200.
    """
    client.session.get.return_value = _mock_response(mocker, status=404)

    with pytest.raises(PubMedClientError):
        client.get_article("9999")
def test_search_timeout_raises_pubmed_client_error(client, mocker):
    """
    Tests that PubMedClient raises PubMedClientError when a search request times out.
    """
    client.session.get.side_effect = TimeoutError("timeout")
    with pytest.raises(PubMedClientError):
        client.search("timeout-case", max_results=1)

def test_get_article_malformed_xml_raises_pubmed_client_error(client, mocker):
    """
    Tests that get_article raises PubMedClientError when the XML response is malformed.
    """
    bad_xml = "<bad><xml>"  # unclosed tag
    client.session.get.return_value = _mock_response(mocker, text=bad_xml)
    with pytest.raises(PubMedClientError):
        client.get_article("123")

def test_search_respects_max_results(client, mocker):
    """
    Tests that the search method returns no more than max_results IDs, even if the API provides more.
    """
    many_ids = [str(i) for i in range(50)]
    json_payload = {"esearchresult": {"idlist": many_ids, "count": "50"}}
    client.session.get.return_value = _mock_response(mocker, json_data=json_payload)
    ids = client.search("query", max_results=10)
    assert len(ids) == 10
    assert ids == many_ids[:10]

@pytest.mark.parametrize("page_size", [0, -1, 101])
def test_search_invalid_page_size_raises_value_error(client, page_size):
    """
    Tests that providing an invalid page_size to the search method raises a ValueError.
    """
    with pytest.raises(ValueError):
        client.search("cancer", max_results=5, page_size=page_size)

def test_search_sets_custom_headers(client, mocker):
    """search should include custom headers set on the client."""
    client.HEADERS = {"User-Agent": "AOT-Tester"}
    fake_json = {"esearchresult": {"idlist": ["1"], "count": "1"}}
    client.session.get.return_value = _mock_response(mocker, json_data=fake_json)
    client.search("cancer", max_results=1)
    called_headers = client.session.get.call_args.kwargs.get("headers")
    assert called_headers == client.HEADERS