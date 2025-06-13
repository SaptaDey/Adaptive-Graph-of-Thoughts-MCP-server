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
    """get_article should raise PubMedClientError on non-200 status."""
    client.session.get.return_value = _mock_response(mocker, status=404)

    with pytest.raises(PubMedClientError):
        client.get_article("9999")
import requests.exceptions

def test_search_timeout_retries_success(client, mocker):
    """Search should retry on Timeout errors and succeed on retry."""
    # First call raises Timeout, second returns a valid response
    client.session.get.side_effect = [
        requests.exceptions.Timeout(),
        _mock_response(mocker, json_data={"esearchresult": {"idlist": ["T1"], "count": "1"}})
    ]

    ids = client.search("timeout-test", max_results=1)

    assert ids == ["T1"]
    assert client.session.get.call_count == 2

def test_search_truncates_when_max_results_exceeded(client, mocker):
    """Search should truncate the results list when total count exceeds max_results."""
    # API reports more IDs than requested
    fake_json = {"esearchresult": {"idlist": ["1", "2", "3", "4"], "count": "10"}}
    client.session.get.return_value = _mock_response(mocker, json_data=fake_json)

    ids = client.search("truncate-test", max_results=2)

    assert len(ids) == 2
    assert ids == ["1", "2"]

def test_get_article_invalid_xml_raises_error(client, mocker):
    """get_article should raise PubMedClientError on malformed XML."""
    # Return HTTP 200 but with invalid XML
    client.session.get.return_value = _mock_response(mocker, status=200, text="<invalid><xml>")

    with pytest.raises(PubMedClientError):
        client.get_article("badxml")

def test_search_without_api_key_param(client, monkeypatch, mocker):
    """Search should omit the api_key param when the environment variable is unset."""
    monkeypatch.delenv("PUBMED_API_KEY", raising=False)
    fake_json = {"esearchresult": {"idlist": ["X"], "count": "1"}}
    client.session.get.return_value = _mock_response(mocker, json_data=fake_json)

    client.search("no-key", max_results=1)
    params = client.session.get.call_args.kwargs["params"]
    assert "api_key" not in params

@pytest.mark.parametrize(
    "max_results,page_size,expected_calls,side_effect_lists",
    [
        (3, 1, 3, [["A"], ["B"], ["C"]]),
        (3, 3, 1, [["A", "B", "C"]]),
        (3, 5, 1, [["A", "B", "C"]]),
    ],
)
def test_search_pagination_edge_cases(client, mocker, max_results, page_size, expected_calls, side_effect_lists):
    """Parametric tests for various page_size vs. max_results combinations."""
    side_effect = []
    total_count = str(sum(len(lst) for lst in side_effect_lists))
    for lst in side_effect_lists:
        side_effect.append(
            _mock_response(
                mocker,
                json_data={"esearchresult": {"idlist": lst, "count": total_count}}
            )
        )
    client.session.get.side_effect = side_effect

    ids = client.search("edge-case", max_results=max_results, page_size=page_size)

    # Flatten side_effect_lists for expected ids
    expected_ids = [item for sublist in side_effect_lists for item in sublist]
    assert ids == expected_ids
    assert client.session.get.call_count == expected_calls