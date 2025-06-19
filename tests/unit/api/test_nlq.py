import pytest
import json
import asyncio

from fastapi.testclient import TestClient
from adaptive_graph_of_thoughts.app_setup import create_app

def test_nlq_endpoint_happy_path(monkeypatch):
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

def test_nlq_endpoint_no_auth():
    """Test NLQ endpoint without authentication"""
    app = create_app()
    client = TestClient(app)

    response = client.post("/nlq", json={"question": "test"})
    assert response.status_code == 401

def test_nlq_endpoint_invalid_auth():
    """Test NLQ endpoint with invalid authentication"""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}  # invalid:invalid
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 401

def test_nlq_endpoint_malformed_auth():
    """Test NLQ endpoint with malformed authorization header"""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Bearer invalid-token"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 401

def test_nlq_endpoint_empty_auth():
    """Test NLQ endpoint with empty authorization header"""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": ""}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 401

def test_nlq_endpoint_missing_question():
    """Test NLQ endpoint with missing question field"""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={}, headers=headers)
    assert response.status_code == 200  # Should handle gracefully with empty question

def test_nlq_endpoint_empty_question(monkeypatch):
    """Test NLQ endpoint with empty question"""
    app = create_app()
    client = TestClient(app)

    def fake_llm(prompt: str) -> str:
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": ""}, headers=headers)
    assert response.status_code == 200

def test_nlq_endpoint_null_question(monkeypatch):
    """Test NLQ endpoint with null question"""
    app = create_app()
    client = TestClient(app)

    def fake_llm(prompt: str) -> str:
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": None}, headers=headers)
    assert response.status_code == 200

def test_nlq_endpoint_very_long_question(monkeypatch):
    """Test NLQ endpoint with extremely long question"""
    app = create_app()
    client = TestClient(app)

    def fake_llm(prompt: str) -> str:
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    long_question = "What is " + "very " * 1000 + "long question?"
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": long_question}, headers=headers)
    assert response.status_code == 200

def test_nlq_endpoint_llm_service_failure(monkeypatch):
    """Test NLQ endpoint when LLM service fails"""
    app = create_app()
    client = TestClient(app)

    def failing_llm(prompt: str) -> str:
        raise Exception("LLM service unavailable")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_llm)

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 500

def test_nlq_endpoint_neo4j_service_failure(monkeypatch):
    """Test NLQ endpoint when Neo4j service fails"""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "summary"

    async def failing_execute_query(*args, **kwargs):
        raise Exception("Database connection failed")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        failing_execute_query,
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 500

def test_nlq_endpoint_invalid_cypher_query(monkeypatch):
    """Test NLQ endpoint when LLM returns invalid Cypher"""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm_invalid_cypher(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "INVALID CYPHER QUERY"
        return "summary"

    async def execute_query_with_error(*args, **kwargs):
        raise Exception("Invalid Cypher syntax")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm_invalid_cypher)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        execute_query_with_error,
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 500

def test_nlq_endpoint_complex_question(monkeypatch):
    """Test NLQ endpoint with complex multi-part question"""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n:Person)-[:WORKS_FOR]->(c:Company) WHERE c.name = 'Tech Corp' RETURN n.name, c.name"
        return "Found 5 employees working for Tech Corp including John Doe, Jane Smith, and others."

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [
            {"n.name": "John Doe", "c.name": "Tech Corp"},
            {"n.name": "Jane Smith", "c.name": "Tech Corp"}
        ],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "Who works for Tech Corp?"}, headers=headers)
    assert response.status_code == 200
    assert len(calls) == 2  # One for query generation, one for summary
    lines = response.text.strip().split("\n")
    assert len(lines) >= 3

def test_nlq_endpoint_no_results(monkeypatch):
    """Test NLQ endpoint when query returns no results"""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n:NonExistent) RETURN n"
        return "No results found for your query."

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "Find something that doesn't exist"}, headers=headers)
    assert response.status_code == 200
    assert "No results found" in response.text or len(response.text.strip().split("\n")) >= 1

def test_nlq_endpoint_special_characters(monkeypatch):
    """Test NLQ endpoint with special characters in question"""
    app = create_app()
    client = TestClient(app)

    def fake_llm(prompt: str) -> str:
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    special_question = "What about nodes with 'quotes' and \"double quotes\" and \n newlines?"
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": special_question}, headers=headers)
    assert response.status_code == 200

def test_nlq_endpoint_get_method():
    """Test NLQ endpoint with GET method (should fail)"""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.get("/nlq", headers=headers)
    assert response.status_code == 405  # Method not allowed

def test_nlq_endpoint_put_method():
    """Test NLQ endpoint with PUT method (should fail)"""
    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.put("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 405  # Method not allowed

def test_nlq_endpoint_malformed_json():
    """Test NLQ endpoint with malformed JSON"""
    app = create_app()
    client = TestClient(app)

    headers = {
        "Authorization": "Basic dGVzdDp0ZXN0",
        "Content-Type": "application/json"
    }
    response = client.post("/nlq", data='{"question": "test"', headers=headers)  # Missing closing brace
    assert response.status_code == 422  # Validation error

def test_nlq_endpoint_multiple_calls_same_question(monkeypatch):
    """Test NLQ endpoint with multiple identical calls"""
    app = create_app()
    client = TestClient(app)

    call_count = 0

    def fake_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "summary"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    question = {"question": "same question"}

    # Make multiple calls
    for i in range(3):
        response = client.post("/nlq", json=question, headers=headers)
        assert response.status_code == 200

    assert call_count == 6  # 2 calls per request (query + summary) * 3 requests

def test_nlq_endpoint_unicode_question(monkeypatch):
    """Test NLQ endpoint with Unicode characters"""
    app = create_app()
    client = TestClient(app)

    def fake_llm(prompt: str) -> str:
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )

    unicode_question = "What about nodes with emoji 🚀 and Chinese 中文 and Arabic العربية?"
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": unicode_question}, headers=headers)
    assert response.status_code == 200

def test_nlq_endpoint_large_response(monkeypatch):
    """Test NLQ endpoint with large result set"""
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1000"
        return "Large dataset with 1000 records returned successfully."

    # Create a large mock result set
    large_results = [{"n.id": f"node_{i}", "n.name": f"Name {i}"} for i in range(1000)]

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: large_results,
    )

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "Find all nodes"}, headers=headers)
    assert response.status_code == 200
    assert len(calls) == 2