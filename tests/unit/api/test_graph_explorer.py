from fastapi.testclient import TestClient
from adaptive_graph_of_thoughts.app_setup import create_app
import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException

@pytest.fixture
def client(monkeypatch):
    """Setup test client with mocked database queries."""
    # Clear auth env vars so tests don't fail due to environment contamination
    monkeypatch.delenv("BASIC_AUTH_USER", raising=False)
    monkeypatch.delenv("BASIC_AUTH_PASS", raising=False)

    async def fake_query(query: str, params=None, database=None, tx_type="read"):
        return [
            {
                "sid": 1,
                "slabels": ["Gene"],
                "sprops": {"name": "TP53"},
                "tid": 2,
                "tlabels": ["Pathway"],
                "tprops": {"name": "Apoptosis"},
                "rid": 10,
                "rtype": "LINKED",
                "rprops": {},
            }
        ]

    async def fake_execute(*args, **kwargs):
        return await fake_query("", {})

    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    app = create_app()
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Valid authentication headers."""
    return {"Authorization": "Basic dGVzdDp0ZXN0"}

def test_graph_explorer_basic_functionality(client, auth_headers):
    """Test basic graph explorer functionality with valid authentication."""
    resp = client.get("/graph", headers=auth_headers)
    
    assert resp.status_code == 200
    data = resp.json()
    
    # Validate response structure
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)
    
    # Validate node and edge counts
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    
    # Validate node structure
    for node in data["nodes"]:
        assert "id" in node
        assert "labels" in node
        assert "properties" in node
    
    # Validate edge structure
    for edge in data["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge
        assert "properties" in edge

def test_graph_explorer_missing_auth(client):
    """Test graph explorer without authentication headers."""
    resp = client.get("/graph")
    assert resp.status_code == 401

def test_graph_explorer_invalid_auth(client):
    """Test graph explorer with invalid authentication."""
    invalid_headers = {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}
    resp = client.get("/graph", headers=invalid_headers)
    assert resp.status_code == 401

def test_graph_explorer_malformed_auth(client):
    """Test graph explorer with malformed authorization header."""
    malformed_headers = {"Authorization": "Bearer invalid_token"}
    resp = client.get("/graph", headers=malformed_headers)
    assert resp.status_code == 401

def test_graph_explorer_empty_auth(client):
    """Test graph explorer with empty authorization header."""
    empty_headers = {"Authorization": ""}
    resp = client.get("/graph", headers=empty_headers)
    assert resp.status_code == 401

def test_graph_explorer_empty_database_response(client, auth_headers, monkeypatch):
    """Test graph explorer when database returns empty results."""
    async def fake_empty_execute(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_empty_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 0
    assert len(data["edges"]) == 0

def test_graph_explorer_multiple_relationships(client, auth_headers, monkeypatch):
    """Test graph explorer with multiple nodes and relationships."""
    async def fake_multi_execute(*args, **kwargs):
        return [
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
                "rid": 10, "rtype": "LINKED", "rprops": {"strength": 0.8},
            },
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 3, "tlabels": ["Disease"], "tprops": {"name": "Cancer"},
                "rid": 11, "rtype": "ASSOCIATED", "rprops": {"confidence": 0.9},
            },
            {
                "sid": 2, "slabels": ["Pathway"], "sprops": {"name": "Apoptosis"},
                "tid": 3, "tlabels": ["Disease"], "tprops": {"name": "Cancer"},
                "rid": 12, "rtype": "INVOLVED_IN", "rprops": {},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_multi_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 3

def test_graph_explorer_database_error(client, auth_headers, monkeypatch):
    """Test graph explorer when database query fails."""
    async def fake_error_execute(*args, **kwargs):
        raise Exception("Database connection failed")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_error_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_invalid_data_format(client, auth_headers, monkeypatch):
    """Test graph explorer with malformed database response."""
    async def fake_invalid_execute(*args, **kwargs):
        return [{"invalid": "data"}]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_invalid_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    # Should handle gracefully or return appropriate error
    assert resp.status_code in [200, 400, 500]

def test_graph_explorer_partial_data(client, auth_headers, monkeypatch):
    """Test graph explorer with partial/incomplete data."""
    async def fake_partial_execute(*args, **kwargs):
        return [
            {
                "sid": 1,
                "slabels": ["Gene"],
                "sprops": {"name": "TP53"},
                # Missing target node and relationship data
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_partial_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code in [200, 400]

def test_graph_explorer_with_query_params(client, auth_headers):
    """Test graph explorer with various query parameters."""
    # Test with limit parameter
    resp = client.get("/graph?limit=10", headers=auth_headers)
    assert resp.status_code == 200
    
    # Test with offset parameter
    resp = client.get("/graph?offset=5", headers=auth_headers)
    assert resp.status_code == 200
    
    # Test with multiple parameters
    resp = client.get("/graph?limit=10&offset=5", headers=auth_headers)
    assert resp.status_code == 200

def test_graph_explorer_invalid_query_params(client, auth_headers):
    """Test graph explorer with invalid query parameters."""
    # Test with negative limit
    resp = client.get("/graph?limit=-1", headers=auth_headers)
    assert resp.status_code in [200, 400]  # Depends on validation logic
    
    # Test with non-numeric parameters
    resp = client.get("/graph?limit=abc", headers=auth_headers)
    assert resp.status_code in [200, 400]

def test_graph_explorer_response_content_type(client, auth_headers):
    """Test that graph explorer returns proper content type."""
    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]

def test_graph_explorer_large_dataset(client, auth_headers, monkeypatch):
    """Test graph explorer with a large dataset response."""
    # Create a large fake dataset
    large_data = []
    for i in range(100):
        large_data.append({
            "sid": i,
            "slabels": ["Node"],
            "sprops": {"name": f"Node_{i}"},
            "tid": i + 100,
            "tlabels": ["Target"],
            "tprops": {"name": f"Target_{i}"},
            "rid": i + 200,
            "rtype": "CONNECTS",
            "rprops": {"weight": i * 0.1},
        })

    async def fake_large_execute(*args, **kwargs):
        return large_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_large_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 200  # 100 source + 100 target nodes
    assert len(data["edges"]) == 100

def test_graph_explorer_node_deduplication(client, auth_headers, monkeypatch):
    """Test that duplicate nodes are properly handled."""
    async def fake_duplicate_execute(*args, **kwargs):
        return [
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
                "rid": 10, "rtype": "LINKED", "rprops": {},
            },
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 3, "tlabels": ["Disease"], "tprops": {"name": "Cancer"},
                "rid": 11, "rtype": "ASSOCIATED", "rprops": {},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_duplicate_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    # Should have 3 unique nodes (1 source appears twice but should be deduplicated)
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2
    
    # Verify node with id=1 appears only once
    node_ids = [node["id"] for node in data["nodes"]]
    assert node_ids.count(1) == 1