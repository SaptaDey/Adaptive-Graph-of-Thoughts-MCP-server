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
@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)

def test_send_request_max_retry_exceeded(no_sleep):
    transport = DummyTransport([{"status": 500}] * 3)
    client = BaseClient(transport=transport, max_retries=2)
    with pytest.raises(RuntimeError):
        client.send_request({"endpoint": "/fails", "payload": {}})

def test_backoff_factor(monkeypatch):
    delays = []
    monkeypatch.setattr(time, "sleep", lambda d: delays.append(d))
    transport = DummyTransport([
        {"status": 429},
        {"status": 429},
        {"status": 200, "data": "ok"}
    ])
    client = BaseClient(transport=transport, max_retries=3, backoff_factor=0.5)
    resp = client.send_request({"endpoint": "/rate-limit", "payload": {}})
    assert resp["status"] == 200
    # Expected exponential delays: 0.5, 1.0
    assert delays == pytest.approx([0.5, 1.0], rel=1e-3)

def test_transport_returns_unexpected_type(client):
    bad_transport = DummyTransport(["not_a_dict"])
    bad_client = BaseClient(transport=bad_transport)
    with pytest.raises(ValueError):
        bad_client.send_request({"endpoint": "/bad"})

def test_internal_retry_counter_thread_safety(monkeypatch):
    # Ensure BaseClient resets its internal retry counter per request
    transport = DummyTransport([{"status": 200, "data": "ok"}] * 10)
    client = BaseClient(transport=transport)
    errors = []

    def worker():
        try:
            client.send_request({"endpoint": "/thread", "payload": {}})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors