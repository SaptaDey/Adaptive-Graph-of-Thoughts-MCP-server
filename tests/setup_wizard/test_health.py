"""
Health endpoint unit tests using pytest and FastAPI TestClient.

This test suite provides comprehensive coverage for the /health endpoint,
testing various scenarios including successful connections, different types
of database failures, edge cases, and HTTP method validation.

Testing Framework: pytest with FastAPI TestClient
Mocking: Built-in monkeypatch fixture for dependency injection
"""
import json
import sys
import types
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

stub_config = types.ModuleType("adaptive_graph_of_thoughts.config")
stub_config.Settings = object
stub_config.runtime_settings = types.SimpleNamespace(
    neo4j=types.SimpleNamespace(
        uri="bolt://localhost", user="neo4j", password="test", database="neo4j"
    ),
    app=types.SimpleNamespace(
        log_level="INFO",
        name="testapp",
        version="0.1",
        cors_allowed_origins_str="*",
        auth_token=None,
    ),
    asr_got={},
)
stub_config.settings = stub_config.runtime_settings
stub_config.env_settings = types.SimpleNamespace(
    llm_provider="openai",
    openai_api_key="test",
    anthropic_api_key=None,
)
stub_config.RuntimeSettings = object
sys.modules.setdefault("adaptive_graph_of_thoughts.config", stub_config)
sys.modules.setdefault("src.adaptive_graph_of_thoughts.config", stub_config)

from adaptive_graph_of_thoughts.app_setup import create_app


def test_health_ok(monkeypatch):
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_down(monkeypatch):
    app = create_app()
    client = TestClient(app)

    class BadDriver:
        def session(self, **_kw):
            raise Exception("fail")

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: BadDriver())
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    resp = client.get("/health", headers=headers)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_response_structure(monkeypatch):
    """Test that health response has correct JSON structure."""
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    resp = client.get("/health")

    assert resp.status_code == 200
    assert "neo4j" in resp.json()
    assert "status" in resp.json()
    assert isinstance(resp.json(), dict)
    assert resp.headers["content-type"] == "application/json"


def test_health_connection_timeout(monkeypatch):
    """Test health check when Neo4j connection times out."""
    app = create_app()
    client = TestClient(app)

    class TimeoutDriver:
        def session(self, **_kw):
            raise TimeoutError("Connection timeout")

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: TimeoutDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
    assert resp.json()["status"] == "unhealthy"


def test_health_connection_refused(monkeypatch):
    """Test health check when Neo4j connection is refused."""
    app = create_app()
    client = TestClient(app)

    class RefusedDriver:
        def session(self, **_kw):
            raise ConnectionRefusedError("Connection refused")

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: RefusedDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_authentication_error(monkeypatch):
    """Test health check when Neo4j authentication fails."""
    app = create_app()
    client = TestClient(app)

    class MockAuthError(Exception):
        pass

    class AuthErrorDriver:
        def session(self, **_kw):
            raise MockAuthError("Invalid credentials")

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: AuthErrorDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_session_context_manager_exit_error(monkeypatch):
    """Test health check when session context manager fails on exit."""
    app = create_app()
    client = TestClient(app)

    class BadExitDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    raise Exception("Exit error")

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: BadExitDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_query_execution_error(monkeypatch):
    """Test health check when Neo4j query execution fails."""
    app = create_app()
    client = TestClient(app)

    class QueryErrorDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    raise Exception("Query execution failed")
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: QueryErrorDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_driver_creation_failure(monkeypatch):
    """Test health check when Neo4j driver creation fails."""
    app = create_app()
    client = TestClient(app)

    def failing_driver(*_a, **_k):
        raise Exception("Driver creation failed")

    monkeypatch.setattr("neo4j.GraphDatabase.driver", failing_driver)
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_health_endpoint_http_methods(monkeypatch, method):
    """Test that health endpoint only accepts GET requests."""
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())

    response = getattr(client, method.lower())("/health")
    assert response.status_code == 405  # Method Not Allowed


def test_health_endpoint_with_query_parameters(monkeypatch):
    """Test health endpoint behavior with query parameters."""
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    resp = client.get("/health?test=param&debug=true")
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_endpoint_with_headers(monkeypatch):
    """Test health endpoint with custom headers."""
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    headers = {"User-Agent": "test-client", "Accept": "application/json"}
    resp = client.get("/health", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_multiple_consecutive_calls(monkeypatch):
    """Test multiple consecutive health check calls to ensure consistency."""
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    for _ in range(5):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["neo4j"] == "up"
        assert resp.json()["status"] == "ok"


def test_health_driver_close_error(monkeypatch):
    """Test health check when driver close() method fails."""
    app = create_app()
    client = TestClient(app)

    class BadCloseDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            raise Exception("Close failed")

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: BadCloseDriver())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_json_serialization(monkeypatch):
    """Test health check response JSON serialization."""
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    resp = client.get("/health")

    json_data = resp.json()
    assert json.dumps(json_data) is not None
    assert json.loads(json.dumps(json_data)) == json_data


def test_health_response_content_type(monkeypatch):
    """Test that health endpoint returns correct content type."""
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass
            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    resp = client.get("/health")
    assert "application/json" in resp.headers.get("content-type", "")
    assert resp.json() is not None


def test_health_service_unavailable_error(monkeypatch):
    """Test health check when Neo4j service is unavailable."""
    app = create_app()
    client = TestClient(app)

    class ServiceUnavailableDriver:
        def session(self, **_kw):
            class MockServiceUnavailableError(Exception):
                pass
            raise MockServiceUnavailableError("Service unavailable")

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: ServiceUnavailableDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
    assert resp.json()["status"] == "unhealthy"

def test_health_driver_timeout(monkeypatch):
    """Test health check when driver() call itself times out."""
    app = create_app()
    client = TestClient(app)

    class TimeoutDriver:
        def __init__(self, *a, **k):
            raise TimeoutError("Timeout creating driver")

    monkeypatch.setattr("neo4j.GraphDatabase.driver", TimeoutDriver)
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    resp = client.get("/health", headers=headers)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"

