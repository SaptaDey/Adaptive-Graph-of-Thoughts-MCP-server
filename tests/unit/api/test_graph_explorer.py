import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from adaptive_graph_of_thoughts.app_setup import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

@pytest.fixture
def auth_headers():
    return {"Authorization": "Basic dGVzdDp0ZXN0"}

@pytest.fixture
def sample_graph_data():
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

def test_graph_explorer_success(client, auth_headers, sample_graph_data, monkeypatch):
    """Test successful graph exploration with valid authentication."""
    async def fake_execute(*args, **kwargs):
        return sample_graph_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

    # Verify node structure
    nodes = data["nodes"]
    assert any(node["id"] == "1" and node["labels"] == ["Gene"] for node in nodes)
    assert any(node["id"] == "2" and node["labels"] == ["Pathway"] for node in nodes)

    # Verify edge structure
    edges = data["edges"]
    assert edges[0]["source"] == "1"
    assert edges[0]["target"] == "2"
    assert edges[0]["type"] == "LINKED"

def test_graph_explorer_no_auth(client):
    """Test graph explorer endpoint without authentication."""
    resp = client.get("/graph")
    assert resp.status_code == 401

def test_graph_explorer_invalid_auth(client):
    """Test graph explorer endpoint with invalid authentication."""
    invalid_headers = {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}
    resp = client.get("/graph", headers=invalid_headers)
    assert resp.status_code == 401

def test_graph_explorer_malformed_auth(client):
    """Test graph explorer endpoint with malformed authentication header."""
    malformed_headers = {"Authorization": "Bearer invalid-token"}
    resp = client.get("/graph", headers=malformed_headers)
    assert resp.status_code == 401

def test_graph_explorer_empty_results(client, auth_headers, monkeypatch):
    """Test graph explorer with empty database results."""
    async def fake_execute_empty(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_empty,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 0
    assert len(data["edges"]) == 0

def test_graph_explorer_database_error(client, auth_headers, monkeypatch):
    """Test graph explorer handling of database connection errors."""
    async def fake_execute_error(*args, **kwargs):
        raise ConnectionError("Database connection failed")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_error,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_timeout_error(client, auth_headers, monkeypatch):
    """Test graph explorer handling of query timeout."""
    async def fake_execute_timeout(*args, **kwargs):
        raise TimeoutError("Query timeout")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_timeout,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_generic_exception(client, auth_headers, monkeypatch):
    """Test graph explorer handling of unexpected exceptions."""
    async def fake_execute_error(*args, **kwargs):
        raise ValueError("Unexpected error")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_error,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_complex_graph(client, auth_headers, monkeypatch):
    """Test graph explorer with complex graph structure."""
    complex_data = [
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53", "type": "tumor_suppressor"},
            "tid": 2,
            "tlabels": ["Pathway"],
            "tprops": {"name": "Apoptosis", "category": "cellular_process"},
            "rid": 10,
            "rtype": "REGULATES",
            "rprops": {"strength": 0.8},
        },
        {
            "sid": 2,
            "slabels": ["Pathway"],
            "sprops": {"name": "Apoptosis", "category": "cellular_process"},
            "tid": 3,
            "tlabels": ["Disease"],
            "tprops": {"name": "Cancer", "severity": "high"},
            "rid": 11,
            "rtype": "ASSOCIATED_WITH",
            "rprops": {"confidence": 0.9},
        },
        {
            "sid": 3,
            "slabels": ["Disease"],
            "sprops": {"name": "Cancer", "severity": "high"},
            "tid": 1,
            "tlabels": ["Gene"],
            "tprops": {"name": "TP53", "type": "tumor_suppressor"},
            "rid": 12,
            "rtype": "MUTATED_IN",
            "rprops": {"frequency": 0.5},
        },
    ]

    async def fake_execute_complex(*args, **kwargs):
        return complex_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_complex,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 3

    # Verify unique nodes
    node_ids = [node["id"] for node in data["nodes"]]
    assert len(set(node_ids)) == 3

    # Verify edge properties are preserved
    edges_with_props = [edge for edge in data["edges"] if edge.get("properties")]
    assert len(edges_with_props) == 3

def test_graph_explorer_missing_properties(client, auth_headers, monkeypatch):
    """Test graph explorer with missing or null properties."""
    data_with_nulls = [
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": None,
            "tid": 2,
            "tlabels": ["Pathway"],
            "tprops": {"name": "Apoptosis"},
            "rid": 10,
            "rtype": "LINKED",
            "rprops": {},
        }
    ]

    async def fake_execute_nulls(*args, **kwargs):
        return data_with_nulls

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_nulls,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

    # Verify handling of null properties
    source_node = next(node for node in data["nodes"] if node["id"] == "1")
    assert source_node["properties"] is None

def test_graph_explorer_with_limit_param(client, auth_headers, sample_graph_data, monkeypatch):
    """Test graph explorer with limit query parameter."""
    async def fake_execute_with_params(query, params=None, **kwargs):
        # Verify the limit parameter is passed correctly
        assert params is not None
        assert params.get("limit") == 10
        return sample_graph_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_with_params,
    )

    resp = client.get("/graph?limit=10", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

def test_graph_explorer_default_limit(client, auth_headers, sample_graph_data, monkeypatch):
    """Test graph explorer uses default limit when not specified."""
    async def fake_execute_default_limit(query, params=None, **kwargs):
        # Verify the default limit of 50 is used
        assert params is not None
        assert params.get("limit") == 50
        return sample_graph_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_default_limit,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

def test_graph_explorer_post_method(client, auth_headers):
    """Test graph explorer endpoint with POST method (should not be allowed)."""
    resp = client.post("/graph", headers=auth_headers, json={})
    assert resp.status_code == 405  # Method Not Allowed

def test_graph_explorer_invalid_endpoint(client, auth_headers):
    """Test invalid graph explorer endpoint."""
    resp = client.get("/graph/invalid", headers=auth_headers)
    assert resp.status_code == 404

def test_graph_explorer_response_format(client, auth_headers, sample_graph_data, monkeypatch):
    """Test the exact format of the graph explorer response."""
    async def fake_execute_format(*args, **kwargs):
        return sample_graph_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_format,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"

    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)

    # Verify node structure
    if data["nodes"]:
        node = data["nodes"][0]
        required_fields = ["id", "labels", "properties"]
        for field in required_fields:
            assert field in node

    # Verify edge structure
    if data["edges"]:
        edge = data["edges"][0]
        required_fields = ["id", "source", "target", "type", "properties"]
        for field in required_fields:
            assert field in edge

def test_graph_explorer_large_dataset(client, auth_headers, monkeypatch):
    """Test graph explorer with a large dataset."""
    # Generate a large dataset
    large_data = []
    for i in range(100):
        large_data.append({
            "sid": i,
            "slabels": ["Node"],
            "sprops": {"name": f"Node{i}"},
            "tid": i + 1,
            "tlabels": ["Node"],
            "tprops": {"name": f"Node{i+1}"},
            "rid": i + 1000,
            "rtype": "CONNECTS",
            "rprops": {"weight": i * 0.01},
        })

    async def fake_execute_large(*args, **kwargs):
        return large_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_large,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 101  # 0 to 100 unique nodes
    assert len(data["edges"]) == 100

def test_graph_explorer_duplicate_nodes(client, auth_headers, monkeypatch):
    """Test graph explorer correctly deduplicates nodes."""
    duplicate_data = [
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53"},
            "tid": 2,
            "tlabels": ["Pathway"],
            "tprops": {"name": "Apoptosis"},
            "rid": 10,
            "rtype": "REGULATES",
            "rprops": {},
        },
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53"},
            "tid": 3,
            "tlabels": ["Disease"],
            "tprops": {"name": "Cancer"},
            "rid": 11,
            "rtype": "ASSOCIATED_WITH",
            "rprops": {},
        },
    ]

    async def fake_execute_duplicates(*args, **kwargs):
        return duplicate_data

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_duplicates,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    # Should have only 3 unique nodes despite duplicates in data
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2

    # Verify node IDs are unique
    node_ids = [node["id"] for node in data["nodes"]]
    assert len(set(node_ids)) == 3

@pytest.mark.parametrize("limit_value,expected_status", [
    (1, 200),
    (50, 200),
    (100, 200),
    (0, 200),
    (-1, 200),
])
def test_graph_explorer_limit_parameters(client, auth_headers, monkeypatch, limit_value, expected_status):
    """Test graph explorer with various limit parameter values."""
    async def fake_execute_limit(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_limit,
    )

    resp = client.get(f"/graph?limit={limit_value}", headers=auth_headers)
    assert resp.status_code == expected_status

def test_graph_explorer_invalid_limit_parameter(client, auth_headers):
    """Test graph explorer with invalid limit parameter."""
    resp = client.get("/graph?limit=invalid", headers=auth_headers)
    assert resp.status_code == 422  # Validation error

def test_graph_explorer_query_structure(client, auth_headers, monkeypatch):
    """Test that the correct Neo4j query is constructed."""
    captured_query = {}

    async def fake_execute_capture(query, params=None, **kwargs):
        captured_query["query"] = query
        captured_query["params"] = params
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_capture,
    )

    client.get("/graph?limit=25", headers=auth_headers)

    # Verify the query contains expected elements
    assert "MATCH (n)-[r]->(m)" in captured_query["query"]
    assert "RETURN" in captured_query["query"]
    assert "LIMIT $limit" in captured_query["query"]
    assert captured_query["params"]["limit"] == 25