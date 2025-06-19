import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException
from adaptive_graph_of_thoughts.app_setup import create_app
from adaptive_graph_of_thoughts.domain.services.neo4j_utils import Neo4jError, ServiceUnavailable

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Standard authentication headers for tests."""
    return {"Authorization": "Basic dGVzdDp0ZXN0"}

@pytest.fixture
def mock_query_result():
    """Standard mock query result for testing."""
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

@pytest.fixture
def mock_empty_result():
    """Empty mock query result for testing."""
    return []

@pytest.fixture
def mock_multiple_relationships():
    """Mock query result with multiple relationships."""
    return [
        {
            "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
            "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
            "rid": 10, "rtype": "LINKED", "rprops": {},
        },
        {
            "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
            "tid": 3, "tlabels": ["Disease"], "tprops": {"name": "Cancer"},
            "rid": 11, "rtype": "ASSOCIATED", "rprops": {"confidence": 0.9},
        },
        {
            "sid": 2, "slabels": ["Pathway"], "sprops": {"name": "Apoptosis"},
            "tid": 3, "tlabels": ["Disease"], "tprops": {"name": "Cancer"},
            "rid": 12, "rtype": "INVOLVES", "rprops": {},
        }
    ]

def test_graph_explorer_success(client, auth_headers, mock_query_result, monkeypatch):
    """Test successful graph explorer endpoint with valid data and default limit."""
    async def fake_execute(query, params=None, database=None, tx_type="read"):
        # Verify the query and parameters
        assert "MATCH (n)-[r]->(m)" in query
        assert "LIMIT $limit" in query
        assert params["limit"] == 50  # Default limit
        return mock_query_result

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

    # Verify node structure and content
    nodes_by_id = {str(node["id"]): node for node in data["nodes"]}

    assert "1" in nodes_by_id
    node_1 = nodes_by_id["1"]
    assert node_1["labels"] == ["Gene"]
    assert node_1["properties"]["name"] == "TP53"

    assert "2" in nodes_by_id
    node_2 = nodes_by_id["2"]
    assert node_2["labels"] == ["Pathway"]
    assert node_2["properties"]["name"] == "Apoptosis"

    # Verify edge structure and content
    edge = data["edges"][0]
    assert edge["id"] == "10"
    assert edge["type"] == "LINKED"
    assert edge["source"] == "1"
    assert edge["target"] == "2"
    assert edge["properties"] == {}

def test_graph_explorer_no_auth_header(client, monkeypatch):
    """Test graph explorer endpoint without authentication header."""
    async def fake_execute(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph")
    assert resp.status_code == 401

def test_graph_explorer_invalid_credentials(client, monkeypatch):
    """Test graph explorer endpoint with invalid authentication credentials."""
    async def fake_execute(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    headers = {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}
    resp = client.get("/graph", headers=headers)
    assert resp.status_code == 401

def test_graph_explorer_malformed_auth_header(client, monkeypatch):
    """Test graph explorer endpoint with malformed authentication header."""
    async def fake_execute(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    test_cases = [
        {"Authorization": "Bearer invalid-token"},
        {"Authorization": "Basic"},
        {"Authorization": "Basic !!!invalid-base64!!!"},
        {"Authorization": "InvalidScheme dGVzdDp0ZXN0"},
    ]

    for headers in test_cases:
        resp = client.get("/graph", headers=headers)
        assert resp.status_code == 401, f"Failed for headers: {headers}"

def test_graph_explorer_missing_basic_auth_env(client, monkeypatch, auth_headers):
    """Test graph explorer when basic auth environment variables are not set."""
    monkeypatch.setenv("BASIC_AUTH_USER", "")
    monkeypatch.setenv("BASIC_AUTH_PASS", "")

    async def fake_execute(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code in [200, 401]

def test_graph_explorer_custom_limit(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint with custom limit parameter."""
    async def fake_execute(query, params=None, database=None, tx_type="read"):
        assert params["limit"] == 25
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph?limit=25", headers=auth_headers)
    assert resp.status_code == 200

@pytest.mark.parametrize("limit_value,expected_status", [
    ("10", 200),
    ("100", 200),
    ("0", 200),
    ("1000", 200),
    ("-1", 422),
    ("abc", 422),
    ("", 422),
])
def test_graph_explorer_limit_parameter_validation(client, auth_headers, monkeypatch, limit_value, expected_status):
    """Test graph explorer endpoint with various limit parameter values."""
    async def fake_execute(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get(f"/graph?limit={limit_value}", headers=auth_headers)
    assert resp.status_code == expected_status

def test_graph_explorer_empty_result(client, auth_headers, mock_empty_result, monkeypatch):
    """Test graph explorer endpoint with empty query result."""
    async def fake_execute(*args, **kwargs):
        return mock_empty_result

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []

def test_graph_explorer_multiple_relationships(client, auth_headers, mock_multiple_relationships, monkeypatch):
    """Test graph explorer endpoint with multiple relationships and node deduplication."""
    async def fake_execute(*args, **kwargs):
        return mock_multiple_relationships

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 3

    node_ids = [node["id"] for node in data["nodes"]]
    assert sorted(node_ids) == ["1", "2", "3"]

    edge_ids = sorted([edge["id"] for edge in data["edges"]])
    assert edge_ids == ["10", "11", "12"]

def test_graph_explorer_node_multiple_labels(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint with nodes having multiple labels."""
    async def fake_execute(*args, **kwargs):
        return [
            {
                "sid": 1, "slabels": ["Gene", "Protein"], "sprops": {"name": "TP53"},
                "tid": 2, "tlabels": ["Pathway", "Process"], "tprops": {"name": "Apoptosis"},
                "rid": 10, "rtype": "LINKED", "rprops": {},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 2

    nodes_by_id = {node["id"]: node for node in data["nodes"]}
    assert nodes_by_id["1"]["labels"] == ["Gene", "Protein"]
    assert nodes_by_id["2"]["labels"] == ["Pathway", "Process"]

def test_graph_explorer_relationship_with_properties(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint with relationship properties."""
    async def fake_execute(*args, **kwargs):
        return [
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
                "rid": 10, "rtype": "LINKED",
                "rprops": {"confidence": 0.95, "source": "database", "score": None},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    edge = data["edges"][0]
    assert edge["properties"]["confidence"] == 0.95
    assert edge["properties"]["source"] == "database"
    assert "score" in edge["properties"]

def test_graph_explorer_neo4j_error(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint when Neo4j raises a specific error."""
    async def fake_execute(*args, **kwargs):
        raise Neo4jError("Cypher query syntax error")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_service_unavailable(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint when Neo4j service is unavailable."""
    async def fake_execute(*args, **kwargs):
        raise ServiceUnavailable("Neo4j service is not available")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_generic_exception(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint when an unexpected exception occurs."""
    async def fake_execute(*args, **kwargs):
        raise Exception("Unexpected database error")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_timeout_error(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint when query times out."""
    async def fake_execute(*args, **kwargs):
        raise TimeoutError("Query execution timed out")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_malformed_database_response(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint with malformed data from database."""
    async def fake_execute(*args, **kwargs):
        return [
            {
                "sid": 1,
                "slabels": ["Gene"],
                "sprops": {"name": "TP53"},
                # Missing tid, tlabels, tprops, rid, rtype, rprops
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_null_values_in_response(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint handles null/None values in database response."""
    async def fake_execute(*args, **kwargs):
        return [
            {
                "sid": 1, "slabels": ["Gene"], "sprops": None,
                "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
                "rid": 10, "rtype": "LINKED", "rprops": None,
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    nodes_by_id = {node["id"]: node for node in data["nodes"]}
    assert nodes_by_id["1"]["properties"] is None
    assert data["edges"][0]["properties"] is None

def test_graph_explorer_large_dataset(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint with large dataset to verify performance handling."""
    large_dataset = []
    for i in range(100):
        large_dataset.append({
            "sid": i, "slabels": ["Node"], "sprops": {"name": f"Node{i}"},
            "tid": i + 100, "tlabels": ["Target"], "tprops": {"name": f"Target{i}"},
            "rid": i + 200, "rtype": "CONNECTS", "rprops": {"weight": i * 0.1},
        })

    async def fake_execute(*args, **kwargs):
        return large_dataset

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 200
    assert len(data["edges"]) == 100

    node_ids = {node["id"] for node in data["nodes"]}
    edge_count_by_source = {}
    for edge in data["edges"]:
        source = edge["source"]
        edge_count_by_source[source] = edge_count_by_source.get(source, 0) + 1
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids

def test_graph_explorer_duplicate_relationships(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint handles duplicate relationships correctly."""
    async def fake_execute(*args, **kwargs):
        return [
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
                "rid": 10, "rtype": "LINKED", "rprops": {},
            },
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
                "rid": 10, "rtype": "LINKED", "rprops": {},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) >= 1

def test_graph_explorer_special_characters_in_data(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint with special characters and unicode in data."""
    async def fake_execute(*args, **kwargs):
        return [
            {
                "sid": 1, "slabels": ["Gene"], "sprops": {"name": "TP53", "description": "Special chars: !@#$%^&*()"},
                "tid": 2, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis", "unicode": "café résumé naïve"},
                "rid": 10, "rtype": "LINKED", "rprops": {"notes": "Unicode test: 한글 中文 العربية"},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    nodes_by_id = {node["id"]: node for node in data["nodes"]}

    assert "Special chars: !@#$%^&*()" in nodes_by_id["1"]["properties"]["description"]
    assert "café résumé naïve" in nodes_by_id["2"]["properties"]["unicode"]
    assert "Unicode test: 한글 中文 العربية" in data["edges"][0]["properties"]["notes"]

def test_graph_explorer_numeric_string_ids(client, auth_headers, monkeypatch):
    """Test graph explorer endpoint correctly handles numeric IDs as strings."""
    async def fake_execute(*args, **kwargs):
        return [
            {
                "sid": 12345, "slabels": ["Gene"], "sprops": {"name": "TP53"},
                "tid": 67890, "tlabels": ["Pathway"], "tprops": {"name": "Apoptosis"},
                "rid": 11111, "rtype": "LINKED", "rprops": {},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    node_ids = [node["id"] for node in data["nodes"]]
    assert "12345" in node_ids
    assert "67890" in node_ids

    edge = data["edges"][0]
    assert edge["id"] == "11111"
    assert edge["source"] == "12345"
    assert edge["target"] == "67890"

@pytest.mark.parametrize("invalid_query_param", [
    "invalid=value",
    "limit=50&extra=param",
    "limit=50&filter=test&sort=name",
])
def test_graph_explorer_ignores_unknown_query_params(client, auth_headers, mock_query_result, monkeypatch, invalid_query_param):
    """Test graph explorer endpoint ignores unknown query parameters gracefully."""
    async def fake_execute(*args, **kwargs):
        return mock_query_result

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get(f"/graph?{invalid_query_param}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

def test_graph_explorer_response_headers(client, auth_headers, mock_query_result, monkeypatch):
    """Test graph explorer endpoint returns correct response headers."""
    async def fake_execute(*args, **kwargs):
        return mock_query_result

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    data = resp.json()
    assert isinstance(data, dict)
    assert "nodes" in data
    assert "edges" in data