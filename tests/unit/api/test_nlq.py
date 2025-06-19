import json
import pytest
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


def test_nlq_endpoint_no_auth():
    """Test NLQ endpoint without authentication returns 401"""
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/nlq", json={"question": "test"})
    assert response.status_code == 401


def test_nlq_endpoint_invalid_auth():
    """Test NLQ endpoint with invalid authentication returns 401"""
    app = create_app()
    client = TestClient(app)
    
    headers = {"Authorization": "Basic invalid_credentials"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 401


def test_nlq_endpoint_missing_question(monkeypatch):
    """Test NLQ endpoint with missing question field"""
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
    response = client.post("/nlq", json={}, headers=headers)
    assert response.status_code == 200  # Empty question becomes empty string


def test_nlq_endpoint_empty_question(monkeypatch):
    """Test NLQ endpoint with empty question"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "No specific question asked"
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": ""}, headers=headers)
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert "No specific question asked" in lines[-1]


def test_nlq_endpoint_special_characters(monkeypatch):
    """Test NLQ endpoint with special characters in question"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) WHERE n.name CONTAINS 'quotes' RETURN n"
        return "Found nodes with special characters"
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [{"name": "test'quote"}],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    special_question = "What about nodes with 'quotes' and \"double quotes\" and \n newlines?"
    response = client.post("/nlq", json={"question": special_question}, headers=headers)
    assert response.status_code == 200
    assert "Found nodes with special characters" in response.text


def test_nlq_endpoint_unicode_characters(monkeypatch):
    """Test NLQ endpoint with unicode characters"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) WHERE n.name CONTAINS 'émoji' RETURN n"
        return "Unicode characters handled successfully 🚀"
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [{"name": "test🚀"}],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    unicode_question = "What about émojis 🚀 and àccénts?"
    response = client.post("/nlq", json={"question": unicode_question}, headers=headers)
    assert response.status_code == 200
    assert "Unicode characters handled successfully 🚀" in response.text


def test_nlq_endpoint_llm_failure(monkeypatch):
    """Test NLQ endpoint when LLM service fails"""
    app = create_app()
    client = TestClient(app)
    
    def failing_llm(prompt: str) -> str:
        raise Exception("LLM service unavailable")
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_llm)
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 500


def test_nlq_endpoint_neo4j_failure(monkeypatch):
    """Test NLQ endpoint when Neo4j query execution fails"""
    app = create_app()
    client = TestClient(app)
    
    call_count = 0
    
    def fake_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "Database query failed"
    
    def failing_neo4j(*args, **kwargs):
        raise Exception("Neo4j connection failed")
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        failing_neo4j,
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 200
    # Should contain error in records response
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    records_line = json.loads(lines[1])
    assert "error" in records_line["records"]


def test_nlq_endpoint_large_result_set(monkeypatch):
    """Test NLQ endpoint with large result set from Neo4j"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1000"
        return "summary of large dataset"
    
    large_results = [{"node_id": f"node_{i}", "value": i} for i in range(100)]
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: large_results,
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "get all nodes"}, headers=headers)
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert "summary of large dataset" in lines[-1]
    
    records_line = json.loads(lines[1])
    assert len(records_line["records"]) == 100


def test_nlq_endpoint_empty_result_set(monkeypatch):
    """Test NLQ endpoint with empty result set from Neo4j"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n:NonExistent) RETURN n"
        return "no results found"
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "find nonexistent nodes"}, headers=headers)
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert "no results found" in lines[-1]


def test_nlq_endpoint_malformed_json():
    """Test NLQ endpoint with malformed JSON"""
    app = create_app()
    client = TestClient(app)
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0", "Content-Type": "application/json"}
    response = client.post("/nlq", data='{"question": "test"', headers=headers)
    assert response.status_code == 422


def test_nlq_endpoint_wrong_content_type():
    """Test NLQ endpoint with wrong content type"""
    app = create_app()
    client = TestClient(app)
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0", "Content-Type": "text/plain"}
    response = client.post("/nlq", data="question=test", headers=headers)
    assert response.status_code == 422


def test_nlq_endpoint_response_format(monkeypatch):
    """Test that NLQ endpoint returns proper streaming response format"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n:Person) RETURN n.name, n.age LIMIT 5"
        return "Found 3 people with their names and ages"
    
    mock_results = [
        {"n.name": "Alice", "n.age": 30},
        {"n.name": "Bob", "n.age": 25},
        {"n.name": "Charlie", "n.age": 35}
    ]
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: mock_results,
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "show me people"}, headers=headers)
    
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    
    cypher_line = json.loads(lines[0])
    assert "cypher" in cypher_line
    assert cypher_line["cypher"] == "MATCH (n:Person) RETURN n.name, n.age LIMIT 5"
    
    records_line = json.loads(lines[1])
    assert "records" in records_line
    assert len(records_line["records"]) == 3
    
    summary_line = json.loads(lines[2])
    assert "summary" in summary_line
    assert "Found 3 people with their names and ages" in summary_line["summary"]


@pytest.mark.parametrize("question", [
    "What are all the nodes?",
    "Show me relationships",
    "Find users with age > 30",
    "What is the most connected node?",
    "How many relationships exist?",
])
def test_nlq_endpoint_various_questions(monkeypatch, question):
    """Test NLQ endpoint with various types of questions"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 10"
        return f"answer for: {question}"
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [{"data": "result"}],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": question}, headers=headers)
    assert response.status_code == 200
    assert len(calls) == 2
    assert f"answer for: {question}" in response.text


def test_nlq_endpoint_multiple_calls_different_questions(monkeypatch):
    """Test NLQ endpoint with multiple different questions in sequence"""
    app = create_app()
    client = TestClient(app)
    
    call_count = 0
    
    def fake_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return f"MATCH (n) RETURN n LIMIT {call_count}"
        return f"summary {call_count // 2}"
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [{"result": "data"}],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    
    response1 = client.post("/nlq", json={"question": "first question"}, headers=headers)
    assert response1.status_code == 200
    response2 = client.post("/nlq", json={"question": "second question"}, headers=headers)
    assert response2.status_code == 200
    assert response1.text != response2.text
    assert "summary 1" in response1.text
    assert "summary 2" in response2.text


def test_nlq_endpoint_very_long_question(monkeypatch):
    """Test NLQ endpoint with extremely long question"""
    app = create_app()
    client = TestClient(app)
    
    calls = []
    
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        return "Processed very long question successfully"
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    long_question = "What is the meaning of life and how does it relate to graph databases? " * 100
    response = client.post("/nlq", json={"question": long_question}, headers=headers)
    assert response.status_code in [200, 413, 422]
    if response.status_code == 200:
        assert "Processed very long question successfully" in response.text


def test_nlq_endpoint_summary_llm_failure(monkeypatch):
    """Test NLQ endpoint when summary LLM call fails but query succeeds"""
    app = create_app()
    client = TestClient(app)
    
    call_count = 0
    
    def failing_summary_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "MATCH (n) RETURN n LIMIT 1"
        raise Exception("Summary LLM failed")
    
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_summary_llm)
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
        lambda *args, **kwargs: [{"data": "test"}],
    )
    
    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test"}, headers=headers)
    assert response.status_code == 500