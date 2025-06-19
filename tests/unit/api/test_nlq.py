import pytest
import threading
import time
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from adaptive_graph_of_thoughts.app_setup import create_app

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Valid Basic authentication headers (test:test encoded in base64)."""
    return {"Authorization": "Basic dGVzdDp0ZXN0"}

@pytest.fixture
def invalid_auth_headers():
    """Invalid Basic authentication headers."""
    return {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}  # invalid:invalid

@pytest.fixture
def mock_llm_service():
    """Mock LLM service that returns predictable responses."""
    def fake_llm(prompt: str) -> str:
        if "Convert this natural language" in prompt:
            return "MATCH (n) RETURN n LIMIT 1"
        elif "Summarize these results" in prompt:
            return "Query executed successfully"
        return "Mock LLM response"
    return fake_llm

@pytest.fixture
def mock_neo4j_service():
    """Mock Neo4j service that returns predictable results."""
    return lambda query: [{"n": {"id": 1, "name": "test_node"}}]

def test_nlq_endpoint_basic_functionality(client, auth_headers, monkeypatch, mock_llm_service, mock_neo4j_service):
    """Test basic NLQ endpoint functionality with valid input."""
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", mock_llm_service)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", mock_neo4j_service)

    response = client.post("/nlq", json={"question": "Show me all nodes"}, headers=auth_headers)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("Query:")
    assert lines[1].startswith("Results:")
    assert lines[2].startswith("Summary:")
    assert "MATCH (n) RETURN n LIMIT 1" in lines[0]

def test_nlq_endpoint_empty_results(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when Neo4j query returns empty results."""
    def fake_llm(prompt: str) -> str:
        if "Convert this natural language" in prompt:
            return "MATCH (n:NonExistent) RETURN n"
        return "No results found in the database"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])

    response = client.post("/nlq", json={"question": "Find non-existent data"}, headers=auth_headers)
    assert response.status_code == 200

    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert "No results found in the database" in lines[2]

def test_nlq_endpoint_complex_question(client, auth_headers, monkeypatch):
    """Test NLQ endpoint with complex multi-part question."""
    def fake_llm(prompt: str) -> str:
        if "Convert this natural language" in prompt:
            return "MATCH (p:Person)-[:KNOWS]->(f:Person) WHERE p.age > 25 RETURN p, f"
        return "Found relationships between people over 25"

    complex_results = [
        {"p": {"name": "Alice", "age": 30}, "f": {"name": "Bob", "age": 28}},
        {"p": {"name": "Charlie", "age": 35}, "f": {"name": "Diana", "age": 27}}
    ]

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: complex_results)

    question = "Find all people over 25 years old who know other people"
    response = client.post("/nlq", json={"question": question}, headers=auth_headers)
    
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert "MATCH (p:Person)-[:KNOWS]->(f:Person)" in lines[0]
    assert "Found relationships between people over 25" in lines[2]

def test_nlq_endpoint_empty_question(client, auth_headers, monkeypatch, mock_llm_service, mock_neo4j_service):
    """Test NLQ endpoint with empty question string."""
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", mock_llm_service)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", mock_neo4j_service)

    response = client.post("/nlq", json={"question": ""}, headers=auth_headers)
    assert response.status_code == 200  # Should still process empty questions

def test_nlq_endpoint_very_long_question(client, auth_headers, monkeypatch):
    """Test NLQ endpoint with extremely long question."""
    def fake_llm(prompt: str) -> str:
        if "Convert this natural language" in prompt:
            return "MATCH (n) RETURN n LIMIT 100"
        return "Processed long query successfully"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])

    long_question = "What is the meaning of life and how does it relate to graph databases " * 50
    response = client.post("/nlq", json={"question": long_question}, headers=auth_headers)
    
    assert response.status_code == 200
    assert "Processed long query successfully" in response.text

def test_nlq_endpoint_special_characters(client, auth_headers, monkeypatch):
    """Test NLQ endpoint with special characters and symbols."""
    def fake_llm(prompt: str) -> str:
        if "Convert this natural language" in prompt:
            return "MATCH (n) WHERE n.name =~ '.*[!@#$%].*' RETURN n"
        return "Found nodes with special characters: !@#$%^&*()"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [{"n": {"name": "test!@#$%"}}])

    special_question = "Find nodes with names containing @#$%^&*()!"
    response = client.post("/nlq", json={"question": special_question}, headers=auth_headers)
    
    assert response.status_code == 200
    assert "Found nodes with special characters" in response.text

def test_nlq_endpoint_unicode_characters(client, auth_headers, monkeypatch):
    """Test NLQ endpoint with unicode and international characters."""
    def fake_llm(prompt: str) -> str:
        if "Convert this natural language" in prompt:
            return "MATCH (n) WHERE n.name CONTAINS 'ñoño' RETURN n"
        return "Encontrado nodos con caracteres unicode: 你好世界"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [{"n": {"name": "niño 你好"}}])

    unicode_question = "Buscar nodos con nombres que contengan 'ñoño' y 你好"
    response = client.post("/nlq", json={"question": unicode_question}, headers=auth_headers)
    
    assert response.status_code == 200
    assert "Encontrado nodos con caracteres unicode" in response.text

def test_nlq_endpoint_json_injection_attempt(client, auth_headers, monkeypatch):
    """Test NLQ endpoint with potential JSON injection in question."""
    def fake_llm(prompt: str) -> str:
        if "Convert this natural language" in prompt:
            return "MATCH (n) RETURN n"
        return "Processed potentially malicious input safely"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])

    malicious_question = '{"malicious": "payload", "injection": true}'
    response = client.post("/nlq", json={"question": malicious_question}, headers=auth_headers)
    
    assert response.status_code == 200
    assert "Processed potentially malicious input safely" in response.text

def test_nlq_endpoint_missing_authorization_header(client):
    """Test NLQ endpoint without any authorization header."""
    response = client.post("/nlq", json={"question": "test question"})
    assert response.status_code == 401

def test_nlq_endpoint_invalid_credentials(client, invalid_auth_headers):
    """Test NLQ endpoint with incorrect username/password."""
    response = client.post("/nlq", json={"question": "test question"}, headers=invalid_auth_headers)
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]

def test_nlq_endpoint_malformed_basic_auth(client):
    """Test NLQ endpoint with malformed Basic authentication header."""
    malformed_headers = {"Authorization": "Basic not-base64-encoded"}
    response = client.post("/nlq", json={"question": "test"}, headers=malformed_headers)
    assert response.status_code == 401

def test_nlq_endpoint_wrong_auth_scheme(client):
    """Test NLQ endpoint with wrong authentication scheme (Bearer instead of Basic)."""
    bearer_headers = {"Authorization": "Bearer some-token"}
    response = client.post("/nlq", json={"question": "test"}, headers=bearer_headers)
    assert response.status_code == 401

def test_nlq_endpoint_empty_authorization_header(client):
    """Test NLQ endpoint with empty authorization header value."""
    empty_headers = {"Authorization": ""}
    response = client.post("/nlq", json={"question": "test"}, headers=empty_headers)
    assert response.status_code == 401

def test_nlq_endpoint_partial_credentials(client):
    """Test NLQ endpoint with only username in Basic auth (missing password)."""
    # Base64 encoding of "test:" (username only, no password)
    partial_headers = {"Authorization": "Basic dGVzdDo="}
    response = client.post("/nlq", json={"question": "test"}, headers=partial_headers)
    assert response.status_code == 401

def test_nlq_endpoint_missing_question_field(client, auth_headers):
    """Test NLQ endpoint with missing required 'question' field."""
    response = client.post("/nlq", json={}, headers=auth_headers)
    assert response.status_code == 422  # Unprocessable Entity
    error_detail = response.json()["detail"][0]
    assert "question" in error_detail["loc"]
    assert error_detail["type"] == "missing"

def test_nlq_endpoint_null_question_value(client, auth_headers):
    """Test NLQ endpoint with null question value."""
    response = client.post("/nlq", json={"question": None}, headers=auth_headers)
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert "question" in error_detail["loc"]

def test_nlq_endpoint_non_string_question(client, auth_headers):
    """Test NLQ endpoint with non-string question (integer)."""
    response = client.post("/nlq", json={"question": 12345}, headers=auth_headers)
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert error_detail["type"] == "string_type"

def test_nlq_endpoint_list_question(client, auth_headers):
    """Test NLQ endpoint with list as question value."""
    response = client.post("/nlq", json={"question": ["invalid", "question", "format"]}, headers=auth_headers)
    assert response.status_code == 422

def test_nlq_endpoint_invalid_json_body(client, auth_headers):
    """Test NLQ endpoint with malformed JSON in request body."""
    response = client.post(
        "/nlq",
        data='{"question": "test", invalid json}',
        headers={**auth_headers, "Content-Type": "application/json"}
    )
    assert response.status_code == 422

def test_nlq_endpoint_wrong_content_type(client, auth_headers):
    """Test NLQ endpoint with form data instead of JSON."""
    response = client.post(
        "/nlq",
        data="question=test",
        headers={**auth_headers, "Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 422

def test_nlq_endpoint_extra_fields_ignored(client, auth_headers, monkeypatch, mock_llm_service, mock_neo4j_service):
    """Test NLQ endpoint ignores extra fields in request body."""
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", mock_llm_service)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", mock_neo4j_service)

    response = client.post("/nlq", json={
        "question": "test question",
        "extra_field": "should be ignored",
        "another_field": 123
    }, headers=auth_headers)
    
    assert response.status_code == 200  # Should still work despite extra fields

def test_nlq_endpoint_llm_service_exception(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when LLM service raises an exception."""
    def failing_llm(prompt: str) -> str:
        raise Exception("LLM service is temporarily unavailable")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_llm)
    
    response = client.post("/nlq", json={"question": "test question"}, headers=auth_headers)
    assert response.status_code == 500

def test_nlq_endpoint_neo4j_service_exception(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when Neo4j service raises an exception."""
    def fake_llm(prompt: str) -> str:
        return "MATCH (n) RETURN n" if "Convert" in prompt else "Summary"

    def failing_neo4j(query: str):
        raise Exception("Neo4j database connection timeout")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", fake_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", failing_neo4j)
    
    response = client.post("/nlq", json={"question": "test question"}, headers=auth_headers)
    assert response.status_code == 500

def test_nlq_endpoint_first_llm_call_fails(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when first LLM call (Cypher generation) fails."""
    call_count = {"count": 0}
    def failing_first_llm(prompt: str) -> str:
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise Exception("Failed to generate Cypher query")
        return "This shouldn't be reached"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_first_llm)
    
    response = client.post("/nlq", json={"question": "test question"}, headers=auth_headers)
    assert response.status_code == 500

def test_nlq_endpoint_second_llm_call_fails(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when second LLM call (summary generation) fails."""
    call_count = {"count": 0}
    def failing_second_llm(prompt: str) -> str:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return "MATCH (n) RETURN n"
        raise Exception("Failed to generate summary")

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_second_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])
    
    response = client.post("/nlq", json={"question": "test question"}, headers=auth_headers)
    assert response.status_code == 500

def test_nlq_endpoint_llm_returns_invalid_cypher(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when LLM returns syntactically invalid Cypher."""
    call_count = {"count": 0}
    def invalid_cypher_llm(prompt: str) -> str:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return "THIS IS NOT VALID CYPHER SYNTAX !!!"
        return "Error processing invalid query"

    def failing_neo4j_for_invalid_cypher(query: str):
        if "THIS IS NOT VALID CYPHER" in query:
            raise Exception("Cypher syntax error")
        return []

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", invalid_cypher_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", failing_neo4j_for_invalid_cypher)
    
    response = client.post("/nlq", json={"question": "test question"}, headers=auth_headers)
    assert response.status_code == 500

def test_nlq_endpoint_llm_returns_empty_response(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when LLM returns empty strings."""
    def empty_llm(prompt: str) -> str:
        return ""

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", empty_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])
    
    response = client.post("/nlq", json={"question": "test question"}, headers=auth_headers)
    assert response.status_code == 200  # Should handle empty responses gracefully
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert "Query:" in lines[0]

def test_nlq_endpoint_get_method_not_allowed(client, auth_headers):
    """Test that GET method returns 405 Method Not Allowed."""
    response = client.get("/nlq", headers=auth_headers)
    assert response.status_code == 405

def test_nlq_endpoint_put_method_not_allowed(client, auth_headers):
    """Test that PUT method returns 405 Method Not Allowed."""
    response = client.put("/nlq", json={"question": "test"}, headers=auth_headers)
    assert response.status_code == 405

def test_nlq_endpoint_delete_method_not_allowed(client, auth_headers):
    """Test that DELETE method returns 405 Method Not Allowed."""
    response = client.delete("/nlq", headers=auth_headers)
    assert response.status_code == 405

def test_nlq_endpoint_patch_method_not_allowed(client, auth_headers):
    """Test that PATCH method returns 405 Method Not Allowed."""
    response = client.patch("/nlq", json={"question": "test"}, headers=auth_headers)
    assert response.status_code == 405

def test_nlq_endpoint_head_method(client, auth_headers):
    """Test HEAD method behavior on NLQ endpoint."""
    response = client.head("/nlq", headers=auth_headers)
    assert response.status_code in [200, 405]

def test_nlq_endpoint_options_method(client):
    """Test OPTIONS method for CORS preflight requests."""
    response = client.options("/nlq")
    assert response.status_code in [200, 204, 405]

def test_nlq_endpoint_response_content_type(client, auth_headers, monkeypatch, mock_llm_service, mock_neo4j_service):
    """Test that response has correct content type for streaming."""
    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", mock_llm_service)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", mock_neo4j_service)

    response = client.post("/nlq", json={"question": "test"}, headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"

def test_nlq_endpoint_response_structure(client, auth_headers, monkeypatch):
    """Test that response follows expected 3-line structure."""
    def structured_llm(prompt: str) -> str:
        if "Convert" in prompt:
            return "MATCH (p:Person) RETURN p.name"
        return "Found 3 people: Alice, Bob, Charlie"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", structured_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [
        {"p.name": "Alice"}, {"p.name": "Bob"}, {"p.name": "Charlie"}
    ])

    response = client.post("/nlq", json={"question": "List all people"}, headers=auth_headers)
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("Query:")
    assert lines[1].startswith("Results:")
    assert lines[2].startswith("Summary:")
    assert "MATCH (p:Person) RETURN p.name" in lines[0]
    assert "Alice" in lines[1] or "Alice" in lines[2]
    assert "Found 3 people" in lines[2]

def test_nlq_endpoint_large_dataset_response(client, auth_headers, monkeypatch):
    """Test NLQ endpoint with large result sets."""
    def llm_for_large_data(prompt: str) -> str:
        if "Convert" in prompt:
            return "MATCH (n) RETURN n LIMIT 1000"
        return "Large dataset with 1000 nodes processed successfully"

    large_results = [{"n": {"id": i, "name": f"node_{i}", "property": f"value_{i}"}} for i in range(1000)]

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", llm_for_large_data)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: large_results)

    response = client.post("/nlq", json={"question": "Get all nodes"}, headers=auth_headers)
    assert response.status_code == 200
    assert "Large dataset with 1000 nodes processed successfully" in response.text
    assert len(response.text) > 5000

def test_nlq_endpoint_multiline_summary(client, auth_headers, monkeypatch):
    """Test NLQ endpoint when LLM returns multiline summary."""
    def multiline_llm(prompt: str) -> str:
        if "Convert" in prompt:
            return "MATCH (n) RETURN n"
        return "Summary line 1\nSummary line 2\nSummary line 3"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", multiline_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])

    response = client.post("/nlq", json={"question": "test"}, headers=auth_headers)
    assert response.status_code == 200
    content = response.text
    assert "Summary line 1" in content
    assert "Summary line 2" in content  
    assert "Summary line 3" in content

def test_nlq_endpoint_response_encoding(client, auth_headers, monkeypatch):
    """Test NLQ endpoint handles unicode characters in response properly."""
    def unicode_llm(prompt: str) -> str:
        if "Convert" in prompt:
            return "MATCH (n:用户) RETURN n.姓名"
        return "找到用户: 张三, 李四, 王五 🚀✨"

    unicode_results = [{"n.姓名": "张三"}, {"n.姓名": "李四"}, {"n.姓名": "王五"}]

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", unicode_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: unicode_results)

    response = client.post("/nlq", json={"question": "查找所有用户"}, headers=auth_headers)
    assert response.status_code == 200
    content = response.text
    assert "MATCH (n:用户) RETURN n.姓名" in content
    assert "找到用户: 张三, 李四, 王五 🚀✨" in content

def test_nlq_endpoint_concurrent_requests(client, auth_headers, monkeypatch):
    """Test NLQ endpoint handles concurrent requests properly."""
    call_times = []

    def slow_llm(prompt: str) -> str:
        call_times.append(time.time())
        time.sleep(0.1)
        if "Convert" in prompt:
            return f"MATCH (n) RETURN n // Request at {time.time()}"
        return f"Concurrent processing completed at {time.time()}"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", slow_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])

    responses = []
    def make_request(idx):
        resp = client.post("/nlq", json={"question": f"concurrent {idx}"}, headers=auth_headers)
        responses.append(resp)

    threads = [threading.Thread(target=make_request, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(responses) == 3
    for resp in responses:
        assert resp.status_code == 200
        assert "Concurrent processing completed" in resp.text
    assert len(call_times) >= 3

def test_nlq_endpoint_timeout_resilience(client, auth_headers, monkeypatch):
    """Test NLQ endpoint behavior with slow services."""
    def very_slow_llm(prompt: str) -> str:
        time.sleep(2)
        if "Convert" in prompt:
            return "MATCH (n) RETURN n"
        return "Slow processing completed"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", very_slow_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: [])

    start = time.time()
    resp = client.post("/nlq", json={"question": "slow test"}, headers=auth_headers)
    duration = time.time() - start

    assert resp.status_code == 200
    assert duration >= 4  # two calls each 2s
    assert "Slow processing completed" in resp.text

def test_nlq_endpoint_memory_usage_with_large_data(client, auth_headers, monkeypatch):
    """Test NLQ endpoint memory handling with very large datasets."""
    def memory_test_llm(prompt: str) -> str:
        if "Convert" in prompt:
            return "MATCH (n) RETURN n"
        return "Successfully processed large dataset without memory issues"

    huge_results = []
    for i in range(10000):
        huge_results.append({
            "n": {
                "id": i,
                "name": f"node_{i}",
                "description": f"This is a very long description for node {i} " * 10,
                "properties": {f"prop_{j}": f"value_{j}_for_node_{i}" for j in range(10)}
            }
        })

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", memory_test_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda query: huge_results)

    resp = client.post("/nlq", json={"question": "memory test"}, headers=auth_headers)
    assert resp.status_code == 200
    assert "Successfully processed large dataset" in resp.text