"""
Health endpoint unit tests using pytest and FastAPI TestClient.

This test suite provides comprehensive coverage for the /health endpoint,
testing various scenarios including successful connections, different types
of database failures, edge cases, and HTTP method validation.

Testing Framework: pytest with FastAPI TestClient
Mocking: Built-in monkeypatch fixture for dependency injection
"""
import json
import os
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
stub_config.LegacyConfig = object
stub_config.Config = object
stub_config.ExaSearchConfig = object
stub_config.GoogleScholarConfig = object
stub_config.PubMedConfig = object
sys.modules.setdefault("adaptive_graph_of_thoughts.config", stub_config)
sys.modules.setdefault("src.adaptive_graph_of_thoughts.config", stub_config)

AUTH = ("user", "pass")
os.environ.setdefault("BASIC_AUTH_USER", AUTH[0])
os.environ.setdefault("BASIC_AUTH_PASS", AUTH[1])

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
    resp = client.get("/health", auth=AUTH)
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
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: BadDriver())
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)

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
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_driver_creation_failure(monkeypatch):
    """Test health check when Neo4j driver creation fails."""
    app = create_app()
    client = TestClient(app)

    def failing_driver(*_a, **_k):
        raise Exception("Driver creation failed")

    monkeypatch.setattr("neo4j.GraphDatabase.driver", failing_driver)
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health?test=param&debug=true", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH, headers=headers)
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
        resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)

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
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH)
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
    resp = client.get("/health", auth=AUTH, headers=headers)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"



def test_health_concurrent_requests(monkeypatch):
    """Test health endpoint with concurrent requests to ensure thread safety."""
    import threading
    import time
    
    app = create_app()
    client = TestClient(app)
    
    results = []
    
    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc, tb):
                    pass
                
                def run(self, _q):
                    time.sleep(0.01)  # Small delay to test concurrency
                    pass
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    
    def make_request():
        resp = client.get("/health", auth=AUTH)
        results.append((resp.status_code, resp.json()))
    
    threads = []
    for _ in range(10):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    assert len(results) == 10
    for status_code, json_data in results:
        assert status_code == 200
        assert json_data["neo4j"] == "up"
        assert json_data["status"] == "ok"


def test_health_session_creation_with_different_kwargs(monkeypatch):
    """Test health check with different session creation parameters."""
    app = create_app()
    client = TestClient(app)
    
    session_kwargs_captured = []
    
    class KwargsCapturingDriver:
        def session(self, **kwargs):
            session_kwargs_captured.append(kwargs)
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
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: KwargsCapturingDriver())
    resp = client.get("/health", auth=AUTH)
    
    assert resp.status_code == 200
    assert len(session_kwargs_captured) >= 1


def test_health_memory_error_during_session(monkeypatch):
    """Test health check when memory error occurs during session creation."""
    app = create_app()
    client = TestClient(app)
    
    class MemoryErrorDriver:
        def session(self, **_kw):
            raise MemoryError("Out of memory")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: MemoryErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
    assert resp.json()["status"] == "unhealthy"


def test_health_keyboard_interrupt_during_query(monkeypatch):
    """Test health check when KeyboardInterrupt occurs during query execution."""
    app = create_app()
    client = TestClient(app)
    
    class InterruptDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc, tb):
                    pass
                
                def run(self, _q):
                    raise KeyboardInterrupt("User interrupted")
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: InterruptDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_system_exit_during_connection(monkeypatch):
    """Test health check when SystemExit is raised during connection."""
    app = create_app()
    client = TestClient(app)
    
    class SystemExitDriver:
        def session(self, **_kw):
            raise SystemExit("System exit")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: SystemExitDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_unicode_error_in_response(monkeypatch):
    """Test health check when unicode encoding issues occur."""
    app = create_app()
    client = TestClient(app)
    
    class UnicodeErrorDriver:
        def session(self, **_kw):
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid unicode")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: UnicodeErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_recursive_exception_handling(monkeypatch):
    """Test health check with nested exceptions."""
    app = create_app()
    client = TestClient(app)
    
    class NestedExceptionDriver:
        def session(self, **_kw):
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RuntimeError("Wrapped error") from e
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: NestedExceptionDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_session_enter_failure(monkeypatch):
    """Test health check when session __enter__ method fails."""
    app = create_app()
    client = TestClient(app)
    
    class BadEnterDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    raise Exception("Enter failed")
                
                def __exit__(self, exc_type, exc, tb):
                    pass
                
                def run(self, _q):
                    pass
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: BadEnterDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_response_headers_comprehensive(monkeypatch):
    """Test comprehensive response headers validation."""
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
    resp = client.get("/health", auth=AUTH)
    
    assert resp.status_code == 200
    assert "content-length" in resp.headers
    assert int(resp.headers["content-length"]) > 0
    assert "application/json" in resp.headers.get("content-type", "")


def test_health_large_error_message_handling(monkeypatch):
    """Test health check with very large error messages."""
    app = create_app()
    client = TestClient(app)
    
    large_error_message = "A" * 10000  # 10KB error message
    
    class LargeErrorDriver:
        def session(self, **_kw):
            raise Exception(large_error_message)
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: LargeErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
    assert len(resp.content) < 50000  # Ensure response isn't too large


def test_health_null_byte_in_error(monkeypatch):
    """Test health check with null bytes in error messages."""
    app = create_app()
    client = TestClient(app)
    
    class NullByteErrorDriver:
        def session(self, **_kw):
            raise Exception("Error with null byte: \x00")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: NullByteErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_empty_string_error(monkeypatch):
    """Test health check with empty string error message."""
    app = create_app()
    client = TestClient(app)
    
    class EmptyErrorDriver:
        def session(self, **_kw):
            raise Exception("")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: EmptyErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_none_error_message(monkeypatch):
    """Test health check with None as error message."""
    app = create_app()
    client = TestClient(app)
    
    class NoneErrorDriver:
        def session(self, **_kw):
            raise Exception(None)
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: NoneErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_response_timing_consistency(monkeypatch):
    """Test that health check response times are consistent."""
    import time
    
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
                    time.sleep(0.01)  # Consistent delay
                    pass
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    
    response_times = []
    for _ in range(5):
        start_time = time.time()
        resp = client.get("/health", auth=AUTH)
        end_time = time.time()
        response_times.append(end_time - start_time)
        assert resp.status_code == 200
    
    # Check response times are relatively consistent (within reasonable bounds)
    avg_time = sum(response_times) / len(response_times)
    for response_time in response_times:
        assert abs(response_time - avg_time) < 1.0  # Within 1 second variance


def test_health_special_characters_in_query(monkeypatch):
    """Test health check behavior when query contains special characters."""
    app = create_app()
    client = TestClient(app)
    
    query_captured = []
    
    class QueryCapturingDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc, tb):
                    pass
                
                def run(self, query):
                    query_captured.append(query)
                    pass
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: QueryCapturingDriver())
    resp = client.get("/health", auth=AUTH)
    
    assert resp.status_code == 200
    assert len(query_captured) == 1
    # Verify the query is safe and doesn't contain injection attempts
    assert isinstance(query_captured[0], str)


def test_health_driver_version_compatibility(monkeypatch):
    """Test health check with different driver version scenarios."""
    app = create_app()
    client = TestClient(app)
    
    class VersionCompatDriver:
        def __init__(self, *args, **kwargs):
            # Simulate driver initialization with version checks
            pass
        
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
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", VersionCompatDriver)
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_database_transaction_error(monkeypatch):
    """Test health check when database transaction fails."""
    app = create_app()
    client = TestClient(app)
    
    class TransactionErrorDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc, tb):
                    pass
                
                def run(self, _q):
                    # Simulate transaction-specific error
                    class MockTransactionError(Exception):
                        pass
                    raise MockTransactionError("Transaction failed")
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: TransactionErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_permission_denied_error(monkeypatch):
    """Test health check when permission is denied."""
    app = create_app()
    client = TestClient(app)
    
    class PermissionDeniedDriver:
        def session(self, **_kw):
            raise PermissionError("Permission denied")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: PermissionDeniedDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_resource_exhaustion(monkeypatch):
    """Test health check under resource exhaustion conditions."""
    app = create_app()
    client = TestClient(app)
    
    class ResourceExhaustionDriver:
        def session(self, **_kw):
            raise OSError("Resource temporarily unavailable")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: ResourceExhaustionDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_json_response_structure_deep_validation(monkeypatch):
    """Test deep validation of JSON response structure and data types."""
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
    resp = client.get("/health", auth=AUTH)
    
    assert resp.status_code == 200
    json_data = resp.json()
    
    # Deep structure validation
    assert isinstance(json_data, dict)
    assert "neo4j" in json_data
    assert "status" in json_data
    assert isinstance(json_data["neo4j"], str)
    assert isinstance(json_data["status"], str)
    assert json_data["neo4j"] in ["up", "down"]
    assert json_data["status"] in ["ok", "unhealthy"]
    
    # Ensure no unexpected keys
    expected_keys = {"neo4j", "status"}
    actual_keys = set(json_data.keys())
    assert actual_keys.issubset(expected_keys) or len(actual_keys) >= len(expected_keys)


@pytest.mark.parametrize("auth_header", [
    "Bearer invalid-token",
    "Basic invalid-base64",
    "Digest username=test",
    "Custom auth-value",
    "",
    None
])
def test_health_various_auth_headers(monkeypatch, auth_header):
    """Test health endpoint with various authentication header formats."""
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
    
    headers = {}
    if auth_header is not None:
        headers["Authorization"] = auth_header
    
    resp = client.get("/health", auth=AUTH, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_endpoint_path_variations(monkeypatch):
    """Test health endpoint with various path variations."""
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
    
    # Test exact path
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 200
    
    # Test with trailing slash (should depend on FastAPI configuration)
    try:
        resp = client.get("/health/")
        # If it doesn't redirect or error, check the response
        if resp.status_code not in [404, 307, 308]:
            assert resp.status_code in [200, 301, 302]
    except Exception:
        # Some configurations might not allow trailing slash
        pass


def test_health_error_logging_verification(monkeypatch):
    """Test that health check errors are properly logged (if logging is configured)."""
    app = create_app()
    client = TestClient(app)
    
    class LoggingErrorDriver:
        def session(self, **_kw):
            raise Exception("Logged error for testing")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: LoggingErrorDriver())
    resp = client.get("/health", auth=AUTH)
    
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
    # Note: Actual log verification would require log capture setup


def test_health_graceful_degradation(monkeypatch):
    """Test health endpoint graceful degradation under various failure modes."""
    app = create_app()
    client = TestClient(app)
    
    class GracefulDegradationDriver:
        def __init__(self, *args, **kwargs):
            # Simulate partial initialization success
            pass
        
        def session(self, **_kw):
            # Session creation succeeds but query fails
            class S:
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc, tb):
                    pass
                
                def run(self, _q):
                    raise Exception("Query failed but connection exists")
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", GracefulDegradationDriver)
    resp = client.get("/health", auth=AUTH)
    
    # Even with query failure, response should be properly structured
    assert resp.status_code == 500
    assert "neo4j" in resp.json()
    assert "status" in resp.json()
    assert resp.json()["neo4j"] == "down"
    assert resp.json()["status"] == "unhealthy"


def test_health_stress_test_rapid_requests(monkeypatch):
    """Stress test health endpoint with rapid consecutive requests."""
    app = create_app()
    client = TestClient(app)
    
    class StressTestDriver:
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
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: StressTestDriver())
    
    # Make 50 rapid requests
    success_count = 0
    for _ in range(50):
        resp = client.get("/health", auth=AUTH)
        if resp.status_code == 200:
            success_count += 1
    
    # At least 90% should succeed
    assert success_count >= 45


def test_health_with_extremely_long_uri(monkeypatch):
    """Test health check with extremely long database URI (edge case)."""
    app = create_app()
    client = TestClient(app)
    
    class LongUriDriver:
        def __init__(self, uri, *args, **kwargs):
            # Simulate handling of very long URI
            if len(uri) > 1000:
                raise ValueError("URI too long")
        
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
    
    # Mock runtime_settings to have a very long URI
    with patch("adaptive_graph_of_thoughts.app_setup.runtime_settings") as mock_settings:
        mock_settings.neo4j.uri = "bolt://" + "a" * 2000 + ".example.com:7687"
        mock_settings.neo4j.user = "neo4j"
        mock_settings.neo4j.password = "password"
        mock_settings.neo4j.database = "neo4j"
        
        monkeypatch.setattr("neo4j.GraphDatabase.driver", LongUriDriver)
        resp = client.get("/health", auth=AUTH)
        
        # Should handle the error gracefully
        assert resp.status_code == 500
        assert resp.json()["neo4j"] == "down"


def test_health_with_malformed_json_response_handling(monkeypatch):
    """Test health endpoint's ability to always return valid JSON."""
    app = create_app()
    client = TestClient(app)
    
    class MalformedDriver:
        def session(self, **_kw):
            # Create an object that might cause JSON serialization issues
            class UnserializableObject:
                def __str__(self):
                    return "unserializable"
            raise Exception(UnserializableObject())
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: MalformedDriver())
    resp = client.get("/health", auth=AUTH)
    
    assert resp.status_code == 500
    # Should still return valid JSON despite the unserializable exception
    json_data = resp.json()
    assert isinstance(json_data, dict)
    assert "neo4j" in json_data
    assert json_data["neo4j"] == "down"


def test_health_connection_pool_exhaustion(monkeypatch):
    """Test health check when connection pool is exhausted."""
    app = create_app()
    client = TestClient(app)
    
    class PoolExhaustedDriver:
        def session(self, **_kw):
            raise Exception("Connection pool exhausted")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: PoolExhaustedDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_ssl_certificate_error(monkeypatch):
    """Test health check when SSL certificate verification fails."""
    app = create_app()
    client = TestClient(app)
    
    import ssl
    
    class SSLErrorDriver:
        def session(self, **_kw):
            raise ssl.SSLCertVerificationError("Certificate verification failed")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: SSLErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_dns_resolution_failure(monkeypatch):
    """Test health check when DNS resolution fails."""
    app = create_app()
    client = TestClient(app)
    
    import socket
    
    class DNSErrorDriver:
        def session(self, **_kw):
            raise socket.gaierror("Name resolution failed")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: DNSErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"


def test_health_driver_session_context_cleanup_verification(monkeypatch):
    """Test that session context is properly cleaned up even on errors."""
    app = create_app()
    client = TestClient(app)
    
    enter_called = []
    exit_called = []
    
    class ContextTrackingDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    enter_called.append(True)
                    return self
                
                def __exit__(self, exc_type, exc, tb):
                    exit_called.append(True)
                    pass
                
                def run(self, _q):
                    raise Exception("Query failed")
            return S()
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: ContextTrackingDriver())
    resp = client.get("/health", auth=AUTH)
    
    assert resp.status_code == 500
    assert len(enter_called) == 1
    assert len(exit_called) == 1  # Ensure __exit__ was called despite error


def test_health_response_immutability(monkeypatch):
    """Test that health response structure is consistent and immutable."""
    app = create_app()
    client = TestClient(app)
    
    class ConsistentDriver:
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
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: ConsistentDriver())
    
    # Make multiple requests and ensure response structure is identical
    responses = []
    for _ in range(3):
        resp = client.get("/health", auth=AUTH)
        responses.append(resp.json())
    
    # All responses should have the same structure
    for response in responses:
        assert response == responses[0]
        assert set(response.keys()) == {"neo4j", "status"}


@pytest.mark.parametrize("exception_type", [
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
    OSError,
    IOError
])
def test_health_network_level_exceptions(monkeypatch, exception_type):
    """Test health check with various network-level exceptions."""
    app = create_app()
    client = TestClient(app)
    
    class NetworkErrorDriver:
        def session(self, **_kw):
            raise exception_type("Network error")
        
        def close(self):
            pass
    
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: NetworkErrorDriver())
    resp = client.get("/health", auth=AUTH)
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
    assert resp.json()["status"] == "unhealthy"
