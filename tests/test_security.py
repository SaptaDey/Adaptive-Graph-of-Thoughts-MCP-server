import pytest
from fastapi.testclient import TestClient
from src.adaptive_graph_of_thoughts.app_setup import create_app
import re


def validate_identifier(identifier: str, allowed_labels: set[str]) -> str:
    """Validate Neo4j identifier to prevent injection attacks."""
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", identifier):
        raise ValueError("Invalid identifier")
    if identifier not in allowed_labels:
        raise ValueError("Identifier not allowed")
    return identifier

class TestSecurity:
    def test_authentication_required(self):
        app = create_app()
        client = TestClient(app)
        
        response = client.post("/mcp", json={"method": "initialize"})
        assert response.status_code == 401
    
    def test_sql_injection_prevention(self):
        # Test Cypher injection attempts
        malicious_inputs = [
            "'; DROP DATABASE neo4j; //",
            "MATCH (n) DETACH DELETE n",
            "CALL dbms.shutdown()",
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError):
                validate_identifier(malicious_input, {"User"})
