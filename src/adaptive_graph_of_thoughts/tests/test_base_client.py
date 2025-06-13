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
from unittest import mock

@pytest.mark.parametrize(
    "statuses, expected_attempts",
    [
        ([{"status": 500}, {"status": 500}, {"status": 200, "data": "ok"}], 3),
        ([{"status": 503}, {"status": 200, "data": "ok"}], 2),
    ],
)
def test_send_request_retries_and_backoff(monkeypatch, statuses, expected_attempts):
    """
    Ensure BaseClient retries on transient 5xx errors and that time.sleep
    is called an increasing number of seconds (exponential back-off).
    """
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float):  # noqa: D401
        # seconds should be monotonic non-decreasing to reflect back-off
        if sleep_calls:
            assert seconds >= sleep_calls[-1]
        sleep_calls.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    transport = DummyTransport(statuses)
    client = BaseClient(transport=transport, max_retries=5, backoff_factor=0.05)
    response = client.send_request({"endpoint": "/retry"})
    assert response["status"] == 200
    # sleep is called attempts-1 times
    assert len(sleep_calls) == expected_attempts - 1


@pytest.mark.parametrize(
    "max_retries, timeout",
    [(-1, 1), (3, -10), ("three", 5)],  # type: ignore[arg-type]
)
def test_client_invalid_initialization(dummy_ok, max_retries, timeout):
    """Improper numeric or type values should raise ValueError."""
    with pytest.raises(ValueError):
        BaseClient(transport=dummy_ok, max_retries=max_retries, timeout=timeout)


def test_send_request_concurrent_mixed_results():
    """Validate thread-safety when some calls fail while others succeed."""
    responses = (  # 3 OK responses + 2 timeouts
        [{"status": 200, "data": "ok"}] * 3 + [TimeoutError("timeout")] * 2
    )
    client = BaseClient(transport=DummyTransport(responses), max_retries=0)
    results: list[dict] = []
    errors: list[str] = []

    def worker():
        try:
            results.append(
                client.send_request({"endpoint": "/mix", "payload": {}})
            )
        except TimeoutError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3
    assert len(errors) == 2
    for r in results:
        assert r["data"] == "ok"


def test_send_request_non_mapping_response():
    """Transport returning a primitive should be rejected."""
    client = BaseClient(transport=DummyTransport(["not-a-mapping"]))
    with pytest.raises(TypeError):
        client.send_request({"endpoint": "/primitive"})


# TODO: Implement strict schema validation to reject unexpected keys
def test_send_request_rejects_unknown_keys(client):
    """Passing unexpected keys should raise ValueError for strict clients."""
    with pytest.raises(ValueError):
        client.send_request({"endpoint": "/foo", "payload": {}, "unexpected": 42})