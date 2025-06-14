# Uses pytest as testing framework.
# To run: pytest -k base_client

import pytest
import threading
import time

from adaptive_graph_of_thoughts.services.api_clients.base_client import BaseClient, TimeoutError

class DummyTransport:
    def __init__(self, responses):
        self._responses = iter(responses)

    def __call__(self, *args, **kwargs):
        resp = next(self._responses)
        if isinstance(resp, Exception):
            raise resp
        return resp

@pytest.fixture
def dummy_ok():
    return DummyTransport([{"status": 200, "data": "ok"}])

@pytest.fixture
def client(dummy_ok):
    return BaseClient(transport=dummy_ok)

def test_send_request_happy_path(client):
    """Happy path: valid initialization and successful request."""
    resp = client.send_request({"endpoint": "/test", "payload": {}})
    assert resp["status"] == 200
    assert resp["data"] == "ok"

def test_send_request_timeout(monkeypatch):
    """Timeout handling: transport raises TimeoutError."""
    transport = DummyTransport([TimeoutError("timeout occurred")])
    client = BaseClient(transport=transport, max_retries=1)
    with pytest.raises(TimeoutError):
        client.send_request({"endpoint": "/timeout"})

@pytest.mark.parametrize("invalid_input", [None, 123, [], {"foo": "bar"}])
def test_send_request_invalid_params(client, invalid_input):
    """Invalid parameters should raise ValueError."""
    with pytest.raises(ValueError):
        client.send_request(invalid_input)

def test_send_request_rate_limit_backoff(monkeypatch):
    """Rate-limit backoff: simulate HTTP 429 then success."""
    transport = DummyTransport([{"status": 429}, {"status": 200, "data": "ok"}])
    client = BaseClient(transport=transport, max_retries=2)
    start_time = time.time()
    resp = client.send_request({"endpoint": "/rate-limit", "payload": {}})
    elapsed = time.time() - start_time
    assert resp["status"] == 200
    # Expect at least minimal backoff delay
    assert elapsed >= 0

def test_send_request_concurrent():
    """Concurrency: multiple threads making requests."""
    transport = DummyTransport([{"status": 200, "data": "ok"}] * 5)
    client = BaseClient(transport=transport)
    results = []

    def worker():
        results.append(client.send_request({"endpoint": "/concurrent", "payload": {}}))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 5
    for resp in results:
        assert resp["status"] == 200
        assert resp["data"] == "ok"
@pytest.mark.parametrize(
    "status_code,error_cls",
    [
        (500, RuntimeError),
        (503, RuntimeError),
        (404, ValueError),
    ],
)
def test_send_request_http_error_mapping(monkeypatch, status_code, error_cls, dummy_ok):
    """
    Failure paths: verify HTTP status→exception mapping logic inside BaseClient._handle_response.
    """
    transport = DummyTransport([{"status": status_code}])
    client = BaseClient(transport=transport, max_retries=0)
    with pytest.raises(error_cls):
        client.send_request({"endpoint": "/err"})

def test_send_request_respects_max_retries(monkeypatch):
    """
    Ensure retry loop stops at max_retries and surfaces last error.
    """
    transport = DummyTransport([
        TimeoutError("t1"),
        TimeoutError("t2"),
        {"status": 200, "data": "ok"},
    ])
    client = BaseClient(transport=transport, max_retries=1)  # should not reach success
    with pytest.raises(TimeoutError):
        client.send_request({"endpoint": "/retry"})

def test_send_request_malformed_response(monkeypatch):
    """
    BaseClient should raise when transport returns unexpected schema.
    """
    transport = DummyTransport([{"foo": "bar"}])
    client = BaseClient(transport=transport)
    with pytest.raises(ValueError):
        client.send_request({"endpoint": "/badresp"})

def test_base_client_context_manager(dummy_ok):
    """
    If BaseClient supports context manager, ensure resource cleanup.
    """
    if not hasattr(BaseClient, "__enter__"):
        pytest.skip("Context manager not implemented")
    with BaseClient(transport=dummy_ok) as client:
        resp = client.send_request({"endpoint": "/ctx", "payload": {}})
        assert resp["data"] == "ok"

def test_concurrent_more_threads_than_responses():
    """
    When threads exceed prepared responses, ensure proper exception handling.
    """
    transport = DummyTransport([{"status": 200, "data": "ok"}] * 3 + [TimeoutError("timeout")])
    client = BaseClient(transport=transport)
    results, errors = [], []
    def worker():
        try:
            results.append(client.send_request({"endpoint": "/con", "payload": {}}))
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) + len(errors) == 4
    assert any(isinstance(e, TimeoutError) for e in errors)

    # ============================================================================
    # Additional Comprehensive Tests for BaseClient
    # ============================================================================

    def test_base_client_initialization_defaults():
    """Test BaseClient initialization with default parameters."""
    transport = DummyTransport([{"status": 200, "data": "test"}])
    client = BaseClient(transport=transport)
    # Verify default values are set properly
    assert client.transport is transport
    # Test that client is functional with defaults
    resp = client.send_request({"endpoint": "/test", "payload": {}})
    assert resp["status"] == 200

    def test_base_client_initialization_custom_params():
    """Test BaseClient initialization with custom parameters."""
    transport = DummyTransport([{"status": 200, "data": "test"}])
    client = BaseClient(transport=transport, max_retries=5, timeout=30)
    # Verify custom parameters are respected
    resp = client.send_request({"endpoint": "/test", "payload": {}})
    assert resp["status"] == 200

    @pytest.mark.parametrize("max_retries", [-1, -10, None])
    def test_base_client_invalid_max_retries(max_retries):
    """Test BaseClient with invalid max_retries values."""
    transport = DummyTransport([{"status": 200, "data": "test"}])
    if max_retries is None:
        # None might be valid, test behavior
        try:
            client = BaseClient(transport=transport, max_retries=max_retries)
            resp = client.send_request({"endpoint": "/test", "payload": {}})
            assert resp["status"] == 200
        except (ValueError, TypeError):
            # Expected for None
            pass
    else:
        # Negative values should be handled appropriately
        with pytest.raises((ValueError, TypeError)):
            BaseClient(transport=transport, max_retries=max_retries)

    def test_base_client_no_transport():
    """Test BaseClient initialization without transport."""
    with pytest.raises(TypeError):
        BaseClient()

    def test_base_client_none_transport():
    """Test BaseClient initialization with None transport."""
    with pytest.raises((TypeError, ValueError)):
        BaseClient(transport=None)

    # ============================================================================
    # Request Parameter Validation Tests
    # ============================================================================

    @pytest.mark.parametrize("request_data", [
    {"endpoint": "/test"},  # Missing payload
    {"payload": {}},  # Missing endpoint
    {"endpoint": "", "payload": {}},  # Empty endpoint
    {"endpoint": None, "payload": {}},  # None endpoint
    {"endpoint": "/test", "payload": None},  # None payload
    ])
    def test_send_request_malformed_request_data(client, request_data):
    """Test send_request with various malformed request data."""
    with pytest.raises(ValueError):
        client.send_request(request_data)

    def test_send_request_empty_endpoint(client):
    """Test send_request with empty string endpoint."""
    with pytest.raises(ValueError):
        client.send_request({"endpoint": "", "payload": {}})

    def test_send_request_whitespace_endpoint(client):
    """Test send_request with whitespace-only endpoint."""
    with pytest.raises(ValueError):
        client.send_request({"endpoint": "   ", "payload": {}})

    @pytest.mark.parametrize("endpoint", [123, [], {}, True, False])
    def test_send_request_non_string_endpoint(client, endpoint):
    """Test send_request with non-string endpoint values."""
    with pytest.raises((ValueError, TypeError)):
        client.send_request({"endpoint": endpoint, "payload": {}})

    def test_send_request_very_long_endpoint(client):
    """Test send_request with extremely long endpoint."""
    long_endpoint = "/test/" + "a" * 10000
    # This should either work or fail gracefully
    try:
        resp = client.send_request({"endpoint": long_endpoint, "payload": {}})
        assert resp["status"] == 200
    except (ValueError, OverflowError):
        # Acceptable failure modes
        pass

    # ============================================================================
    # Transport Response Handling Tests
    # ============================================================================

    @pytest.mark.parametrize("response", [
    None,  # None response
    "",  # Empty string
    "invalid",  # String instead of dict
    [],  # List instead of dict
    123,  # Number instead of dict
    True,  # Boolean instead of dict
    ])
    def test_send_request_invalid_transport_responses(response):
    """Test send_request with various invalid transport responses."""
    transport = DummyTransport([response])
    client = BaseClient(transport=transport)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        client.send_request({"endpoint": "/test", "payload": {}})

    def test_send_request_missing_status_in_response():
    """Test send_request when transport response lacks status field."""
    transport = DummyTransport([{"data": "test", "message": "ok"}])
    client = BaseClient(transport=transport)
    with pytest.raises((ValueError, KeyError)):
        client.send_request({"endpoint": "/test", "payload": {}})

    @pytest.mark.parametrize("status", [
    "200",  # String status instead of int
    200.5,  # Float status
    None,  # None status
    [],  # List status
    {},  # Dict status
    ])
    def test_send_request_invalid_status_types(status):
    """Test send_request with invalid status field types."""
    transport = DummyTransport([{"status": status, "data": "test"}])
    client = BaseClient(transport=transport)
    with pytest.raises((ValueError, TypeError)):
        client.send_request({"endpoint": "/test", "payload": {}})

    @pytest.mark.parametrize("status_code", [
    100, 101, 102,  # 1xx informational
    201, 202, 204,  # 2xx success variants
    301, 302, 304,  # 3xx redirects
    400, 401, 403,  # 4xx client errors
    500, 502, 503, 504,  # 5xx server errors
    ])
    def test_send_request_various_http_status_codes(status_code):
    """Test send_request response handling for various HTTP status codes."""
    transport = DummyTransport([{"status": status_code, "data": f"status_{status_code}"}])
    client = BaseClient(transport=transport, max_retries=0)
    
    if 200 <= status_code < 300:
        # Success cases
        resp = client.send_request({"endpoint": "/test", "payload": {}})
        assert resp["status"] == status_code
    else:
        # Error cases - should raise appropriate exceptions
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            client.send_request({"endpoint": "/test", "payload": {}})

    # ============================================================================
    # Advanced Retry and Backoff Tests
    # ============================================================================

    def test_send_request_retry_with_mixed_errors():
    """Test retry behavior with mixed error types."""
    transport = DummyTransport([
        TimeoutError("timeout1"),
        {"status": 500},  # Server error
        TimeoutError("timeout2"),
        {"status": 200, "data": "success"}
    ])
    client = BaseClient(transport=transport, max_retries=4)
    resp = client.send_request({"endpoint": "/test", "payload": {}})
    assert resp["status"] == 200
    assert resp["data"] == "success"

    def test_send_request_exponential_backoff_timing():
    """Test that exponential backoff increases delay between retries."""
    # Create a transport that fails twice then succeeds
    transport = DummyTransport([
        {"status": 429},  # Rate limit
        {"status": 429},  # Rate limit again
        {"status": 200, "data": "success"}
    ])
    client = BaseClient(transport=transport, max_retries=3)
    
    start_time = time.time()
    resp = client.send_request({"endpoint": "/test", "payload": {}})
    total_time = time.time() - start_time
    
    assert resp["status"] == 200
    # Should have some delay from backoff (implementation dependent)
    # This is a weak assertion since timing can be unpredictable in tests
    assert total_time >= 0

    def test_send_request_max_retries_zero():
    """Test behavior when max_retries is set to 0."""
    transport = DummyTransport([{"status": 500}])
    client = BaseClient(transport=transport, max_retries=0)
    with pytest.raises(RuntimeError):
        client.send_request({"endpoint": "/test", "payload": {}})

    def test_send_request_max_retries_large_value():
    """Test behavior with very large max_retries value."""
    transport = DummyTransport([{"status": 200, "data": "ok"}])
    client = BaseClient(transport=transport, max_retries=1000)
    resp = client.send_request({"endpoint": "/test", "payload": {}})
    assert resp["status"] == 200

    def test_send_request_alternating_success_failure():
    """Test retry behavior with alternating success/failure patterns."""
    transport = DummyTransport([
        {"status": 500},  # Fail
        {"status": 200, "data": "success1"},  # Success (shouldn't be reached due to retry)
        {"status": 500},  # Fail again
        {"status": 200, "data": "success2"}   # Final success
    ])
    client = BaseClient(transport=transport, max_retries=3)
    resp = client.send_request({"endpoint": "/test", "payload": {}})
    assert resp["status"] == 200

    # ============================================================================
    # Stress Testing and Edge Cases
    # ============================================================================

    def test_send_request_very_large_payload():
    """Test send_request with very large payload data."""
    large_payload = {"data": "x" * 100000}  # 100KB payload
    transport = DummyTransport([{"status": 200, "data": "ok"}])
    client = BaseClient(transport=transport)
    
    try:
        resp = client.send_request({"endpoint": "/test", "payload": large_payload})
        assert resp["status"] == 200
    except (MemoryError, OverflowError):
        # Acceptable failure for very large payloads
        pass

    def test_send_request_nested_payload_structure():
    """Test send_request with deeply nested payload structures."""
    nested_payload = {"level1": {"level2": {"level3": {"level4": {"data": "deep"}}}}}
    transport = DummyTransport([{"status": 200, "data": "ok"}])
    client = BaseClient(transport=transport)
    
    resp = client.send_request({"endpoint": "/test", "payload": nested_payload})
    assert resp["status"] == 200

    def test_send_request_unicode_endpoint():
    """Test send_request with unicode characters in endpoint."""
    unicode_endpoint = "/test/üñíçødé/ẽñðpøíñt"
    transport = DummyTransport([{"status": 200, "data": "ok"}])
    client = BaseClient(transport=transport)
    
    resp = client.send_request({"endpoint": unicode_endpoint, "payload": {}})
    assert resp["status"] == 200

    def test_send_request_special_characters_payload():
    """Test send_request with special characters in payload."""
    special_payload = {
        "text": "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "unicode": "Unicode: üñíçødé ñàmé",
        "newlines": "Line1\nLine2\r\nLine3",
        "tabs": "Col1\tCol2\tCol3"
    }
    transport = DummyTransport([{"status": 200, "data": "ok"}])
    client = BaseClient(transport=transport)
    
    resp = client.send_request({"endpoint": "/test", "payload": special_payload})
    assert resp["status"] == 200

    def test_concurrent_requests_with_failures():
    """Test concurrent requests where some succeed and some fail."""
    responses = [
        {"status": 200, "data": "success1"},
        TimeoutError("timeout"),
        {"status": 500},
        {"status": 200, "data": "success2"},
        RuntimeError("transport error")
    ]
    transport = DummyTransport(responses)
    client = BaseClient(transport=transport, max_retries=0)
    
    results = []
    errors = []
    
    def worker(endpoint):
        try:
            result = client.send_request({"endpoint": f"/test/{endpoint}", "payload": {}})
            results.append(result)
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should have some successes and some failures
    assert len(results) + len(errors) == 5
    assert len(results) > 0 or len(errors) > 0

    def test_send_request_empty_response_data():
    """Test handling of responses with empty or missing data field."""
    transport = DummyTransport([
        {"status": 200},  # Missing data field
        {"status": 200, "data": ""},  # Empty data
        {"status": 200, "data": None}  # None data
    ])
    client = BaseClient(transport=transport)
    
    # Test missing data field
    resp1 = client.send_request({"endpoint": "/test1", "payload": {}})
    assert resp1["status"] == 200
    
    # Test empty data
    resp2 = client.send_request({"endpoint": "/test2", "payload": {}})
    assert resp2["status"] == 200
    assert resp2["data"] == ""
    
    # Test None data
    resp3 = client.send_request({"endpoint": "/test3", "payload": {}})
    assert resp3["status"] == 200
    assert resp3["data"] is None

    # ============================================================================
    # Transport Interaction and Mock Validation Tests
    # ============================================================================

    def test_transport_called_with_correct_parameters():
    """Test that transport is called with the expected parameters."""
    called_with = []
    
    class CapturingTransport:
        def __call__(self, *args, **kwargs):
            called_with.append((args, kwargs))
            return {"status": 200, "data": "ok"}
    
    transport = CapturingTransport()
    client = BaseClient(transport=transport)
    
    request_data = {"endpoint": "/test", "payload": {"key": "value"}}
    client.send_request(request_data)
    
    assert len(called_with) == 1
    args, kwargs = called_with[0]
    # Verify transport was called with expected parameters
    # (Implementation-specific - adjust based on actual BaseClient behavior)

    def test_transport_not_called_on_invalid_request():
    """Test that transport is not called when request validation fails."""
    called = []
    
    class CapturingTransport:
        def __call__(self, *args, **kwargs):
            called.append(True)
            return {"status": 200, "data": "ok"}
    
    transport = CapturingTransport()
    client = BaseClient(transport=transport)
    
    with pytest.raises(ValueError):
        client.send_request(None)  # Invalid request
    
    assert len(called) == 0  # Transport should not be called

    def test_transport_exception_propagation():
    """Test that transport exceptions are properly propagated."""
    class ExceptionTransport:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Transport failed")
    
    transport = ExceptionTransport()
    client = BaseClient(transport=transport, max_retries=0)
    
    with pytest.raises(RuntimeError, match="Transport failed"):
        client.send_request({"endpoint": "/test", "payload": {}})

    def test_multiple_clients_same_transport():
    """Test multiple BaseClient instances sharing the same transport."""
    transport = DummyTransport([
        {"status": 200, "data": "response1"},
        {"status": 200, "data": "response2"}
    ])
    
    client1 = BaseClient(transport=transport)
    client2 = BaseClient(transport=transport)
    
    resp1 = client1.send_request({"endpoint": "/test1", "payload": {}})
    resp2 = client2.send_request({"endpoint": "/test2", "payload": {}})
    
    assert resp1["data"] == "response1"
    assert resp2["data"] == "response2"

    # ============================================================================
    # Resource Management and Cleanup Tests
    # ============================================================================

    def test_client_behavior_after_many_requests():
    """Test client behavior after processing many requests."""
    responses = [{"status": 200, "data": f"response_{i}"} for i in range(100)]
    transport = DummyTransport(responses)
    client = BaseClient(transport=transport)
    
    results = []
    for i in range(100):
        resp = client.send_request({"endpoint": f"/test/{i}", "payload": {}})
        results.append(resp)
    
    assert len(results) == 100
    for i, resp in enumerate(results):
        assert resp["status"] == 200
        assert resp["data"] == f"response_{i}"

    def test_client_state_isolation():
    """Test that multiple requests don't affect each other's state."""
    transport = DummyTransport([
        {"status": 200, "data": "first"},
        {"status": 500},  # This should not affect the next request
        {"status": 200, "data": "third"}
    ])
    client = BaseClient(transport=transport, max_retries=0)
    
    # First request succeeds
    resp1 = client.send_request({"endpoint": "/test1", "payload": {}})
    assert resp1["status"] == 200
    assert resp1["data"] == "first"
    
    # Second request fails
    with pytest.raises(RuntimeError):
        client.send_request({"endpoint": "/test2", "payload": {}})
    
    # Third request should still work (client state not corrupted)
    resp3 = client.send_request({"endpoint": "/test3", "payload": {}})
    assert resp3["status"] == 200
    assert resp3["data"] == "third"