# --- Additional pytest test cases for BaseClient ---

import httpx
from adaptive_graph_of_thoughts.services.api_clients import base_client as base_client_module
from adaptive_graph_of_thoughts.services.api_clients.base_client import BaseClient, BaseClientError, BaseClientTimeoutError

class TestBaseClientGet:
    def test_get_success(self, monkeypatch):
        """Happy path: BaseClient.get returns JSON on successful request."""
        dummy_data = {"key": "value"}
        class DummyResponse:
            def raise_for_status(self): pass
            def json(self): return dummy_data
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): pass
            def get(self, path, params=None): return DummyResponse()
        monkeypatch.setattr(base_client_module.httpx, "Client", lambda *args, **kwargs: DummyHTTPXClient())
        client = BaseClient("http://example.com", timeout=1.0, headers={'X-Test': 'value'})
        result = client.get("/endpoint", params={"param": "test"})
        assert result == dummy_data

    def test_get_timeout_raises_baseclient_timeout_error(self, monkeypatch):
        """Error path: BaseClient.get raises BaseClientTimeoutError on timeout."""
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): pass
            def get(self, path, params=None):
                raise httpx.TimeoutException("timeout")
        monkeypatch.setattr(base_client_module.httpx, "Client", lambda *args, **kwargs: DummyHTTPXClient())
        client = BaseClient("http://example.com")
        with pytest.raises(BaseClientTimeoutError):
            client.get("/endpoint")

    def test_get_http_error_raises_baseclient_error(self, monkeypatch):
        """Error path: BaseClient.get raises BaseClientError on HTTP error."""
        class DummyResponse:
            def raise_for_status(self):
                raise httpx.HTTPError("error")
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): pass
            def get(self, path, params=None):
                return DummyResponse()
        monkeypatch.setattr(base_client_module.httpx, "Client", lambda *args, **kwargs: DummyHTTPXClient())
        client = BaseClient("http://example.com")
        with pytest.raises(BaseClientError):
            client.get("/endpoint")

class TestBaseClientInitialization:
    def test_headers_and_auth_set_correctly(self):
        """Test that provided headers are applied to the HTTPX client."""
        headers = {"Authorization": "Bearer token", "Custom-Header": "value"}
        client = BaseClient("http://example.com", headers=headers)
        assert client.client.headers.get("Authorization") == "Bearer token"
        assert client.client.headers.get("Custom-Header") == "value"

    def test_invalid_base_url_type_raises_value_error(self):
        """Test that initializing BaseClient with non-string base_url raises ValueError."""
        with pytest.raises(ValueError):
            BaseClient(12345)

class TestBaseClientRetryLogic:
    def test_retry_logic_on_http_error(self, monkeypatch):
        """Test BaseClient.get retries failed requests before returning a successful response."""
        call_count = {"count": 0}
        dummy_data = {"retry": "success"}
        class DummyResponse:
            def raise_for_status(self):
                if call_count["count"] < 2:
                    call_count["count"] += 1
                    raise httpx.HTTPError("temporary error")
            def json(self):
                return dummy_data
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): pass
            def get(self, path, params=None):
                return DummyResponse()
        monkeypatch.setattr(base_client_module.httpx, "Client", lambda *args, **kwargs: DummyHTTPXClient())
        client = BaseClient("http://example.com")
        result = client.get("/endpoint")
        assert call_count["count"] == 2
        assert result == dummy_data