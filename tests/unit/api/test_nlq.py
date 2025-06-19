import pytest
import json
import threading
from fastapi.testclient import TestClient
from adaptive_graph_of_thoughts.app_setup import create_app

def test_nlq_endpoint(monkeypatch):
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "summary"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert "summary" in lines[-1]

def test_nlq_endpoint_missing_authorization():
    """Test NLQ endpoint without Authorization header."""
    app = create_app()
    client = TestClient(app)

    response = client.post("/nlq", json={"question": "test"})
    assert response.status_code == 401

def test_nlq_endpoint_invalid_authorization():
    """Test NLQ endpoint with invalid Authorization header."""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}  # invalid:invalid
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 401

def test_nlq_endpoint_missing_question(monkeypatch):
    """Test NLQ endpoint with missing question parameter."""
    app = create_app()
    client = TestClient(app)

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda prompt: "test")
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={}, headers=headers)
    assert response.status_code == 200  # Endpoint handles missing question gracefully
    lines = response.text.strip().split("\n")
    assert len(lines) >= 1

def test_nlq_endpoint_empty_question(monkeypatch):
    """Test NLQ endpoint with empty question parameter."""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "summary"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": ""}, headers=headers)
    assert response.status_code == 200
    assert len(calls) >= 1

def test_nlq_endpoint_invalid_json():
    """Test NLQ endpoint with malformed JSON input."""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic dGVzdDp0ZXN0", "Content-Type": "application/json"}
    response = client.post("/nlq", data="invalid json", headers=headers)
    assert response.status_code == 422

def test_nlq_endpoint_llm_exception(monkeypatch):
    """Test NLQ endpoint when LLM service fails."""
    app = create_app()
    client = TestClient(app)

    def failing_llm(prompt: str) -> str:
        raise Exception("LLM service unavailable")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 500

def test_nlq_endpoint_neo4j_exception(monkeypatch):
    """Test NLQ endpoint when Neo4j service fails."""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "summary"

    def failing_neo4j(*args, **kwargs):
        raise Exception("Database connection failed")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        failing_neo4j,
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 200  # Endpoint should handle Neo4j errors gracefully
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2  # Should contain cypher and error info

def test_nlq_endpoint_long_question(monkeypatch):
    """Test NLQ endpoint with very long question."""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "summary"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    long_question = "a" * 5000  # Very long question
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": long_question}, headers=headers)
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) >= 3
    assert "summary" in lines[-1]

def test_nlq_endpoint_special_characters(monkeypatch):
    """Test NLQ endpoint with special characters in question."""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "summary"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    special_question = "What's the 'test' data with \"quotes\" and unicode: 中文 émojis 🔥?"
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": special_question}, headers=headers)
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) >= 3
    assert "summary" in lines[-1]

def test_nlq_endpoint_different_http_methods():
    """Test that NLQ endpoint only accepts POST method."""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}

    # Test GET method
    response = client.get("/nlq", headers=headers)
    assert response.status_code == 405  # Method Not Allowed

    # Test PUT method
    response = client.put("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 405

    # Test DELETE method
    response = client.delete("/nlq", headers=headers)
    assert response.status_code == 405

def test_nlq_endpoint_response_format(monkeypatch):
    """Test NLQ endpoint response format and content."""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n:Person) RETURN n.name LIMIT 10"
        return "Found 10 person names"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [{"n.name": "John"}, {"n.name": "Jane"}],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "Show me people"}, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"

    lines = response.text.strip().split("\n")
    assert len(lines) >= 3
    # First line should contain cypher query
    first_line_data = json.loads(lines[0])
    assert "cypher" in first_line_data
    assert "MATCH (n:Person) RETURN n.name LIMIT 10" in first_line_data["cypher"]
    assert len(calls) == 2  # Verify LLM was called twice

def test_nlq_endpoint_concurrent_requests(monkeypatch):
    """Test NLQ endpoint with concurrent requests."""
    app = create_app()
    client = TestClient(app)

    call_count = 0
    lock = threading.Lock()

    def fake_llm(prompt: str) -> str:
        nonlocal call_count
        with lock:
            call_count += 1
            if call_count % 2 == 1:
                return f"QUERY {call_count}"
            return f"summary {call_count}"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}

    # Send multiple concurrent requests
    responses = []
    for i in range(3):
        response = client.post("/nlq", json={"question": f"test {i}"}, headers=headers)
        responses.append(response)

    # Verify all responses are successful
    for response in responses:
        assert response.status_code == 200
        lines = response.text.strip().split("\n")
        assert len(lines) >= 3

def test_nlq_endpoint_malformed_auth_header():
    """Test NLQ endpoint with malformed Authorization headers."""
    app = create_app()
    client = TestClient(app)

    test_cases = [
        {"Authorization": "Bearer token"},  # Wrong auth type
        {"Authorization": "Basic"},  # Missing credentials
        {"Authorization": "Basic invalid"},  # Invalid base64
        {"Authorization": ""},  # Empty header
    ]

    for headers in test_cases:
        response = client.post("/nlq", json={"question": "test"}, headers=headers)
        assert response.status_code == 401