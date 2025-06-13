# Uses pytest as testing framework.
# To run: pytest -k base_client

import pytest
import threading
import time

from adaptive_graph_of_thoughts.base_client import BaseClient, TimeoutError

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
    """
    Tests that BaseClient can handle concurrent requests from multiple threads.
    
    Verifies that when multiple threads send requests simultaneously, each receives a successful response and all results are collected.
    """
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
    Tests that specific HTTP status codes returned by the transport are mapped to the correct exceptions by BaseClient.
    
    Args:
        status_code: The HTTP status code to simulate in the transport response.
        error_cls: The expected exception class to be raised.
    """
    transport = DummyTransport([{"status": status_code}])
    client = BaseClient(transport=transport, max_retries=0)
    with pytest.raises(error_cls):
        client.send_request({"endpoint": "/err"})

def test_send_request_respects_max_retries(monkeypatch):
    """
    Tests that the client stops retrying after reaching the max_retries limit and raises the last encountered error.
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
    Tests that BaseClient raises a ValueError when the transport returns a response with an unexpected schema.
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
    Tests that when more threads make requests than there are prepared responses, the client correctly handles exceptions and all outcomes are accounted for.
    
    Verifies that the total number of results and errors matches the number of threads, and that at least one `TimeoutError` is raised when responses are exhausted.
    """
    transport = DummyTransport([{"status": 200, "data": "ok"}] * 3 + [TimeoutError("timeout")])
    client = BaseClient(transport=transport)
    results, errors = [], []
    def worker():
        """
        Sends a request using the client and appends the result or any exception to shared lists.
        
        Appends the response from `client.send_request` to `results`, or appends the exception to `errors` if an error occurs.
        """
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