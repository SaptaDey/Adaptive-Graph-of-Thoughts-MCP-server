import pytest
import requests
from pubmed_client import PubmedClient

class TestPubmedClient:

    @pytest.fixture(autouse=True)
    def client(self):
        # Instantiate with a dummy API key for header tests
        return PubmedClient(api_key="test_key")

    def make_response(self, status_code: int, text: str) -> requests.Response:
        resp = requests.Response()
        resp.status_code = status_code
        resp._content = text.encode('utf-8')
        return resp

    def test_search_success_returns_pmids(self, monkeypatch):
        sample_xml = (
            "<PubmedArticleSet>"
              "<IdList><Id>123</Id><Id>456</Id></IdList>"
            "</PubmedArticleSet>"
        )
        # Mock Session.get to return our sample XML
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, headers=None, params=None: self.make_response(200, sample_xml)
        )
        result = self.client.search("cancer")
        assert result == ["123", "456"]

    @pytest.mark.parametrize("sample_xml,expected", [
        ("<PubmedArticleSet><IdList></IdList></PubmedArticleSet>", []),
        ("<PubmedArticleSet></PubmedArticleSet>", []),
    ])
    def test_search_no_results_returns_empty_list(self, monkeypatch, sample_xml, expected):
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, headers=None, params=None: self.make_response(200, sample_xml)
        )
        assert self.client.search("unknown") == expected

    def test_search_network_error_raises_exception(self, monkeypatch):
        def raise_conn(self, url, headers=None, params=None):
            raise requests.ConnectionError("Network failure")
        monkeypatch.setattr(requests.Session, "get", raise_conn)
        with pytest.raises(requests.ConnectionError):
            self.client.search("timeout")

    def test_search_malformed_response_graceful_failure(self, monkeypatch):
        # Return invalid XML to simulate parsing error
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, headers=None, params=None: self.make_response(200, "<bad<xml>")
        )
        with pytest.raises(ValueError):
            self.client.search("badxml")

    def test_search_pagination_large_query(self, monkeypatch):
        # Two pages of results
        xml_page1 = "<PubmedArticleSet><IdList><Id>1</Id></IdList></PubmedArticleSet>"
        xml_page2 = "<PubmedArticleSet><IdList><Id>2</Id></IdList></PubmedArticleSet>"
        responses = [self.make_response(200, xml_page1), self.make_response(200, xml_page2)]
        call_counter = {"count": 0}

        def fake_get(self, url, headers=None, params=None):
            idx = call_counter["count"]
            call_counter["count"] += 1
            return responses[idx]

        monkeypatch.setattr(requests.Session, "get", fake_get)
        result = self.client.search("bigquery")
        assert result == ["1", "2"]
        assert call_counter["count"] == 2

    def test_auth_header_inclusion(self, monkeypatch):
        captured = {}
        def record_get(self, url, headers=None, params=None):
            captured["headers"] = headers
            return self.make_response(200, "<PubmedArticleSet><IdList></IdList></PubmedArticleSet>")

        monkeypatch.setattr(requests.Session, "get", record_get)
        self.client.search("auth-test")
        assert "Authorization" in captured["headers"]
        assert captured["headers"]["Authorization"] == "Bearer test_key"