import pytest
import json
from unittest.mock import Mock, patch
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


class TestNLQEndpointAuthentication:
    """Test authentication scenarios for the NLQ endpoint."""

    def test_nlq_endpoint_no_auth(self):
        """Test NLQ endpoint without authentication headers."""
        app = create_app()
        client = TestClient(app)

        response = client.post("/nlq", json={"question": "test"})
        assert response.status_code == 401

    def test_nlq_endpoint_invalid_auth(self):
        """Test NLQ endpoint with invalid authentication."""
        app = create_app()
        client = TestClient(app)

        headers = {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}  # invalid:invalid
        response = client.post("/nlq", json={"question": "test"}, headers=headers)
        assert response.status_code == 401

    def test_nlq_endpoint_malformed_auth_header(self):
        """Test NLQ endpoint with malformed authorization header."""
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


class TestNLQEndpointInputValidation:
    """Test input validation for the NLQ endpoint."""

    def setup_method(self):
        """Setup for each test method."""
        self.app = create_app()
        self.client = TestClient(self.app)
        self.valid_headers = {"Authorization": "Basic dGVzdDp0ZXN0"}

    def test_nlq_endpoint_missing_question(self, monkeypatch):
        """Test NLQ endpoint with missing question field."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda x: "MATCH (n) RETURN n")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

        response = self.client.post("/nlq", json={}, headers=self.valid_headers)
        assert response.status_code == 200  # Should handle missing question gracefully with empty string
        assert '"cypher"' in response.text

    def test_nlq_endpoint_empty_question(self, monkeypatch):
        """Test NLQ endpoint with empty question."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda x: "MATCH (n) RETURN n")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

        response = self.client.post("/nlq", json={"question": ""}, headers=self.valid_headers)
        assert response.status_code == 200  # Should handle empty question gracefully
        assert '"cypher"' in response.text

    def test_nlq_endpoint_null_question(self, monkeypatch):
        """Test NLQ endpoint with null question."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda x: "MATCH (n) RETURN n")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

        response = self.client.post("/nlq", json={"question": None}, headers=self.valid_headers)
        assert response.status_code == 200  # Should handle null as empty string

    @pytest.mark.parametrize("question", [
        "a" * 1000,  # Very long question
        "What is the meaning of life?" * 100,  # Repeated long question
        "🤖🔍📊💡🌟" * 50,  # Unicode/emoji heavy
        "SELECT * FROM users; DROP TABLE users; --",  # SQL injection attempt
        "<script>alert('xss')</script>",  # XSS attempt
        "Question with\nnewlines\nand\ttabs",  # Special characters
        "Question with 'quotes' and \"double quotes\"",  # Quote handling
    ])
    def test_nlq_endpoint_edge_case_inputs(self, monkeypatch, question):
        """Test NLQ endpoint with various edge case inputs."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda x: "MATCH (n) RETURN n")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

        response = self.client.post("/nlq", json={"question": question}, headers=self.valid_headers)
        assert response.status_code == 200  # Should handle all inputs gracefully
        assert '"cypher"' in response.text

    def test_nlq_endpoint_invalid_json(self):
        """Test NLQ endpoint with invalid JSON payload."""
        response = self.client.post(
            "/nlq",
            data="invalid json",
            headers={**self.valid_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_nlq_endpoint_wrong_content_type(self):
        """Test NLQ endpoint with wrong content type."""
        response = self.client.post(
            "/nlq",
            data="question=test",
            headers={**self.valid_headers, "Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422  # FastAPI returns 422 for wrong content type


class TestNLQEndpointLLMIntegration:
    """Test LLM service integration scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        self.app = create_app()
        self.client = TestClient(self.app)
        self.valid_headers = {"Authorization": "Basic dGVzdDp0ZXN0"}

    def test_nlq_endpoint_llm_service_failure(self, monkeypatch):
        """Test NLQ endpoint when LLM service fails during Cypher generation."""
        def failing_llm(prompt: str) -> str:
            raise Exception("LLM service unavailable")

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_llm)

        response = self.client.post("/nlq", json={"question": "test"}, headers=self.valid_headers)
        assert response.status_code == 200  # Streaming response starts before error
        assert '"error"' in response.text

    def test_nlq_endpoint_llm_summary_failure(self, monkeypatch):
        """Test NLQ endpoint when LLM service fails during summary generation."""
        call_count = 0

        def failing_summary_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "MATCH (n) RETURN n LIMIT 1"
            raise Exception("Summary generation failed")

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", failing_summary_llm)
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

        response = self.client.post("/nlq", json={"question": "test"}, headers=self.valid_headers)
        assert response.status_code == 200
        assert '"cypher"' in response.text
        assert '"error"' in response.text

    def test_nlq_endpoint_llm_returns_empty(self, monkeypatch):
        """Test NLQ endpoint when LLM returns empty response."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda x: "")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

        response = self.client.post("/nlq", json={"question": "test"}, headers=self.valid_headers)
        assert response.status_code == 200  # Should handle empty LLM response gracefully
        assert '"cypher": ""' in response.text

    def test_nlq_endpoint_llm_returns_invalid_query(self, monkeypatch):
        """Test NLQ endpoint when LLM returns invalid Cypher query."""
        def ask_llm(prompt: str) -> str:
            if "Translate" in prompt:
                return "INVALID CYPHER QUERY"
            return "Could not process invalid query"

        def failing_neo4j(*args, **kwargs):
            raise Exception("Cypher syntax error")

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", ask_llm)
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", failing_neo4j)

        response = self.client.post("/nlq", json={"question": "test"}, headers=self.valid_headers)
        assert response.status_code == 200  # Should handle invalid queries gracefully
        assert '"cypher"' in response.text
        assert '"error"' in response.text

    def test_nlq_endpoint_multiple_llm_calls(self, monkeypatch):
        """Test the sequence of LLM calls in the NLQ processing."""
        calls = []

        def mock_llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return "MATCH (n:Person) RETURN n.name LIMIT 5"
            elif len(calls) == 2:
                return "Found 5 people in the database with names John and Jane"
            return "default response"

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", mock_llm)
        monkeypatch.setattr(
            "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
            lambda *args, **kwargs: [{"n.name": "John"}, {"n.name": "Jane"}]
        )

        response = self.client.post("/nlq", json={"question": "Who are the people?"}, headers=self.valid_headers)
        assert response.status_code == 200
        assert len(calls) == 2  # Should make exactly 2 LLM calls
        assert "Translate the question" in calls[0]
        assert "Answer the question" in calls[1]
        lines = response.text.strip().split("\n")
        assert len(lines) == 3  # cypher, records, summary


class TestNLQEndpointNeo4jIntegration:
    """Test Neo4j integration scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        self.app = create_app()
        self.client = TestClient(self.app)
        self.valid_headers = {"Authorization": "Basic dGVzdDp0ZXN0"}

    def test_nlq_endpoint_neo4j_connection_failure(self, monkeypatch):
        """Test NLQ endpoint when Neo4j connection fails."""
        def failing_neo4j(*args, **kwargs):
            raise Exception("Neo4j connection failed")

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda x: "MATCH (n) RETURN n" if "Translate" in x else "Database connection error")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", failing_neo4j)

        response = self.client.post("/nlq", json={"question": "test"}, headers=self.valid_headers)
        assert response.status_code == 200  # Streaming response starts successfully
        assert '"cypher"' in response.text
        assert '"error"' in response.text

    def test_nlq_endpoint_neo4j_empty_result(self, monkeypatch):
        """Test NLQ endpoint when Neo4j returns empty results."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", 
                          lambda x: "MATCH (n:NonExistent) RETURN n" if "Translate" in x else "No results found for your query")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

        response = self.client.post("/nlq", json={"question": "find non-existent data"}, headers=self.valid_headers)
        assert response.status_code == 200
        assert '"records": []' in response.text
        assert '"summary"' in response.text

    def test_nlq_endpoint_neo4j_large_result_set(self, monkeypatch):
        """Test NLQ endpoint with large Neo4j result set."""
        large_result = [{"id": i, "name": f"Item{i}"} for i in range(100)]  # Reduced for test performance

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", 
                          lambda x: "MATCH (n) RETURN n" if "Translate" in x else "Large dataset with 100 items returned")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: large_result)

        response = self.client.post("/nlq", json={"question": "get all data"}, headers=self.valid_headers)
        assert response.status_code == 200
        assert len(response.text) > 1000  # Should handle large datasets
        assert '"records"' in response.text
        assert '"summary"' in response.text

    def test_nlq_endpoint_neo4j_timeout(self, monkeypatch):
        """Test NLQ endpoint when Neo4j query times out."""
        import time

        def slow_neo4j(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow query
            raise Exception("Query timeout")

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", lambda x: "MATCH (n) RETURN n" if "Translate" in x else "Query timed out")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", slow_neo4j)

        response = self.client.post("/nlq", json={"question": "slow query"}, headers=self.valid_headers)
        assert response.status_code == 200
        assert '"error"' in response.text


class TestNLQEndpointResponseFormat:
    """Test response format and streaming behavior."""

    def setup_method(self):
        """Setup for each test method."""
        self.app = create_app()
        self.client = TestClient(self.app)
        self.valid_headers = {"Authorization": "Basic dGVzdDp0ZXN0"}

    def test_nlq_endpoint_response_format(self, monkeypatch):
        """Test that the response format is correct."""
        calls = []

        def mock_llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return "MATCH (n:User) RETURN n.name LIMIT 3"
            return "Query executed successfully, found 3 users named Alice, Bob, and Charlie"

        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", mock_llm)
        monkeypatch.setattr(
            "adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query",
            lambda *args, **kwargs: [{"n.name": "Alice"}, {"n.name": "Bob"}, {"n.name": "Charlie"}]
        )

        response = self.client.post("/nlq", json={"question": "Show me users"}, headers=self.valid_headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json; charset=utf-8"

        # Check response format - should be 3 JSON lines
        lines = response.text.strip().split("\n")
        assert len(lines) == 3

        # Parse each line as JSON
        cypher_line = json.loads(lines[0])
        records_line = json.loads(lines[1])
        summary_line = json.loads(lines[2])

        assert "cypher" in cypher_line
        assert "MATCH" in cypher_line["cypher"]
        assert "records" in records_line
        assert len(records_line["records"]) == 3
        assert "summary" in summary_line
        assert "Alice" in summary_line["summary"]

    def test_nlq_endpoint_streaming_response_chunked(self, monkeypatch):
        """Test that the endpoint properly streams responses in chunks."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", 
                          lambda x: "MATCH (n) RETURN n" if "Translate" in x else "streaming test complete")
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [{"test": "data"}])

        response = self.client.post("/nlq", json={"question": "test streaming"}, headers=self.valid_headers)
        assert response.status_code == 200

        # Check if response is properly formatted as streaming JSON
        lines = response.text.strip().split("\n")
        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            json.loads(line)  # Should not raise exception

    def test_nlq_endpoint_json_serialization_edge_cases(self, monkeypatch):
        """Test JSON serialization with edge case data."""
        monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", 
                          lambda x: "MATCH (n) RETURN n" if "Translate" in x else "Special characters handled")

        # Neo4j result with special characters
        special_data = [
            {"name": "John's \"Special\" Data", "unicode": "🌟", "newline": "line1\nline2"},
            {"null_value": None, "number": 42, "boolean": True}
        ]
        monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: special_data)

        response = self.client.post("/nlq", json={"question": "test special chars"}, headers=self.valid_headers)
        assert response.status_code == 200

        lines = response.text.strip().split("\n")
        records_line = json.loads(lines[1])
        assert len(records_line["records"]) == 2
        assert records_line["records"][0]["unicode"] == "🌟"
        assert records_line["records"][1]["null_value"] is None


@pytest.mark.parametrize("question,expected_in_response", [
    ("Find all users", ["MATCH", "users"]),
    ("Count the nodes", ["count", "nodes"]),
    ("Show relationships", ["relationship"]),
    ("What is the structure?", ["structure"]),
    ("How many people are there?", ["people", "many"]),
])
def test_nlq_endpoint_question_types(monkeypatch, question, expected_in_response):
    """Test different types of natural language questions."""
    app = create_app()
    client = TestClient(app)

    def smart_llm(prompt: str) -> str:
        if "Translate" in prompt:
            # First call - generate query based on question type
            if "user" in question.lower():
                return "MATCH (u:User) RETURN u"
            elif "count" in question.lower() or "many" in question.lower():
                return "MATCH (n) RETURN count(n)"
            elif "relationship" in question.lower():
                return "MATCH ()-[r]->() RETURN type(r)"
            elif "structure" in question.lower():
                return "CALL db.schema.visualization()"
            else:
                return "MATCH (n) RETURN n LIMIT 10"
        else:
            # Second call - generate summary
            return f"Analysis complete for: {question}. Results show relevant data."

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", smart_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [{"result": "test_data"}])

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": question}, headers=headers)
    assert response.status_code == 200

    response_text = response.text.lower()
    # At least one expected keyword should appear in the response
    assert any(keyword.lower() in response_text for keyword in expected_in_response)


def test_nlq_endpoint_logging_functionality(monkeypatch):
    """Test that LLM queries are properly logged."""
    from adaptive_graph_of_thoughts.services.llm import LLM_QUERY_LOGS

    # Clear logs before test
    LLM_QUERY_LOGS.clear()

    app = create_app()
    client = TestClient(app)

    def mock_llm(prompt: str) -> str:
        if "Translate" in prompt:
            return "MATCH (n) RETURN n"
        return "Test summary"

    monkeypatch.setattr("adaptive_graph_of_thoughts.services.llm.ask_llm", mock_llm)
    monkeypatch.setattr("adaptive_graph_of_thoughts.domain.services.neo4j_utils.execute_query", lambda *args, **kwargs: [])

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    response = client.post("/nlq", json={"question": "test logging"}, headers=headers)

    assert response.status_code == 200
    assert len(LLM_QUERY_LOGS) == 2  # Should log both LLM calls
    assert "Translate the question" in LLM_QUERY_LOGS[0]["prompt"]
    assert "Answer the question" in LLM_QUERY_LOGS[1]["prompt"]


def test_original_functionality_regression(monkeypatch):
    """Ensure the original test functionality is preserved - regression test."""
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


# Testing Framework Documentation:
# This test suite uses pytest as the primary testing framework with the following components:
# - FastAPI TestClient for HTTP endpoint testing
# - monkeypatch fixture for dependency injection and mocking
# - pytest.mark.parametrize for data-driven tests
# - unittest.mock.Mock for advanced mocking scenarios
# - Standard Python json library for response validation
# - Class-based test organization for better structure and setup/teardown
#
# The tests comprehensively cover:
# - Authentication scenarios (valid, invalid, missing, malformed)
# - Input validation (empty, null, invalid JSON, edge cases)
# - Service integration (LLM failures, Neo4j failures, timeouts)
# - Response format validation (streaming JSON, serialization)
# - Error handling and edge cases
# - Different types of natural language queries
# - Logging functionality verification