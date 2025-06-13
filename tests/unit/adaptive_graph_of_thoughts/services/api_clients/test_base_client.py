# --- Additional pytest test cases for BaseClient ---

import httpx
from adaptive_graph_of_thoughts.services.api_clients import base_client as base_client_module
from adaptive_graph_of_thoughts.services.api_clients.base_client import BaseClient, BaseClientError, BaseClientTimeoutError

class TestBaseClientGet:
    def test_get_success(self, monkeypatch):
        """
        Tests that BaseClient.get returns the expected JSON data on a successful GET request.
        """
        dummy_data = {"key": "value"}
        class DummyResponse:
            def raise_for_status(self): """
Raises an HTTP error if the response status indicates a failure.

Typically used to trigger an exception for non-successful HTTP status codes.
"""
pass
            def json(self): """
Returns the dummy JSON data for testing purposes.
"""
return dummy_data
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): """
Initializes a BaseClient instance with the specified base URL, timeout, and headers.

Args:
	base_url: The base URL for API requests.
	timeout: The request timeout value.
	headers: A dictionary of HTTP headers to include with each request.
"""
pass
            def get(self, path, params=None): """
Simulates an HTTP GET request and returns a dummy response.

Args:
    path: The request path.
    params: Optional query parameters for the request.

Returns:
    A DummyResponse object representing the simulated response.
"""
return DummyResponse()
        monkeypatch.setattr(base_client_module.httpx, "Client", lambda *args, **kwargs: DummyHTTPXClient())
        client = BaseClient("http://example.com", timeout=1.0, headers={'X-Test': 'value'})
        result = client.get("/endpoint", params={"param": "test"})
        assert result == dummy_data

    def test_get_timeout_raises_baseclient_timeout_error(self, monkeypatch):
        """
        Tests that BaseClient.get raises BaseClientTimeoutError when a timeout occurs during the request.
        """
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): """
Initializes a BaseClient instance with the specified base URL, timeout, and headers.

Args:
	base_url: The base URL for API requests.
	timeout: The request timeout value.
	headers: A dictionary of HTTP headers to include with each request.
"""
pass
            def get(self, path, params=None):
                """
                Simulates a GET request that always raises a timeout exception.
                
                Intended for use in tests to verify timeout handling logic.
                """
                raise httpx.TimeoutException("timeout")
        monkeypatch.setattr(base_client_module.httpx, "Client", lambda *args, **kwargs: DummyHTTPXClient())
        client = BaseClient("http://example.com")
        with pytest.raises(BaseClientTimeoutError):
            client.get("/endpoint")

    def test_get_http_error_raises_baseclient_error(self, monkeypatch):
        """
        Tests that BaseClient.get raises BaseClientError when an HTTP error occurs during a GET request.
        """
        class DummyResponse:
            def raise_for_status(self):
                """
                Raises an HTTPError to simulate an HTTP error response.
                
                Intended for use in tests to mimic error conditions during HTTP requests.
                """
                raise httpx.HTTPError("error")
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): """
Initializes a BaseClient instance with the specified base URL, timeout, and headers.

Args:
	base_url: The base URL for API requests.
	timeout: The request timeout value.
	headers: A dictionary of HTTP headers to include with each request.
"""
pass
            def get(self, path, params=None):
                """
                Simulates an HTTP GET request and returns a dummy response.
                
                This method is intended for testing purposes and does not perform any real network operations.
                """
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
        """
        Tests that initializing BaseClient with a non-string base_url raises a ValueError.
        """
        with pytest.raises(ValueError):
            BaseClient(12345)

class TestBaseClientRetryLogic:
    def test_retry_logic_on_http_error(self, monkeypatch):
        """
        Tests that BaseClient.get retries on HTTP errors and eventually returns a successful response after transient failures.
        """
        call_count = {"count": 0}
        dummy_data = {"retry": "success"}
        class DummyResponse:
            def raise_for_status(self):
                """
                Simulates raising an HTTP error on the first two calls to mimic transient failures.
                
                Raises:
                    httpx.HTTPError: On the first two invocations to simulate a temporary error.
                """
                if call_count["count"] < 2:
                    call_count["count"] += 1
                    raise httpx.HTTPError("temporary error")
            def json(self):
                """
                Returns the dummy JSON data for the mocked response.
                """
                return dummy_data
        class DummyHTTPXClient:
            def __init__(self, base_url, timeout, headers): """
Initializes a BaseClient instance with the specified base URL, timeout, and headers.

Args:
	base_url: The base URL for API requests.
	timeout: The request timeout value.
	headers: A dictionary of HTTP headers to include with each request.
"""
pass
            def get(self, path, params=None):
                """
                Simulates an HTTP GET request and returns a dummy response.
                
                This method is intended for testing purposes and does not perform any real network operations.
                """
                return DummyResponse()
        monkeypatch.setattr(base_client_module.httpx, "Client", lambda *args, **kwargs: DummyHTTPXClient())
        client = BaseClient("http://example.com")
        result = client.get("/endpoint")
        assert call_count["count"] == 2
        assert result == dummy_data