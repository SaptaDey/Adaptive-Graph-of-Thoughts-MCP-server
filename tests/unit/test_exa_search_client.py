import pytest
import requests
from exa_search_client import ExaSearchClient, ExaSearchError, DocumentNotFoundError

@pytest.fixture
def client():
    return ExaSearchClient(base_url="http://example.com", api_key="testkey")

def test_search_returns_expected_ids(monkeypatch, client):
    dummy = {"results": [{"id": 1}, {"id": 2}]}
    class Resp:
        status_code = 200
        def json(self): return dummy
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Resp())
    results = client.search("hello")
    assert [item["id"] for item in results] == [1, 2]

@pytest.mark.parametrize("query,expected", [
    ("", []),
    ("notfound", []),
])
def test_search_empty_or_no_hits(monkeypatch, client, query, expected):
    class Resp:
        status_code = 200
        def json(self): return {"results": []}
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Resp())
    assert client.search(query) == expected

def test_search_raises_on_http_error(monkeypatch, client):
    class Resp:
        status_code = 500
        text = "server error"
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Resp())
    with pytest.raises(ExaSearchError):
        client.search("failcase")

def test_get_document_success(monkeypatch, client):
    payload = {"id": 42, "content": "foo"}
    class Resp:
        status_code = 200
        def json(self): return payload
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Resp())
    assert client.get_document(42) == payload

def test_get_document_404_raises(monkeypatch, client):
    class Resp:
        status_code = 404
        text = "not found"
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Resp())
    with pytest.raises(DocumentNotFoundError):
        client.get_document(123)

def test_delete_document_success(monkeypatch, client):
    class Resp:
        status_code = 204
    monkeypatch.setattr(requests, "delete", lambda *args, **kwargs: Resp())
    assert client.delete_document(5) is True

def test_delete_document_404_raises(monkeypatch, client):
    class Resp:
        status_code = 404
        text = "missing"
    monkeypatch.setattr(requests, "delete", lambda *args, **kwargs: Resp())
    with pytest.raises(DocumentNotFoundError):
        client.delete_document(5)

@pytest.mark.parametrize("settings", [
    {"relevance": "high"},
    {}
])
def test_update_settings_happy(monkeypatch, client, settings):
    class Resp:
        status_code = 200
    monkeypatch.setattr(requests, "put", lambda *args, **kwargs: Resp())
    assert client.update_settings(settings) is True

def test_update_settings_failure(monkeypatch, client):
    class Resp:
        status_code = 400
        text = "bad request"
    monkeypatch.setattr(requests, "put", lambda *args, **kwargs: Resp())
    with pytest.raises(ExaSearchError):
        client.update_settings({"invalid": True})