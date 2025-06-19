import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from adaptive_graph_of_thoughts.app_setup import create_app

@pytest.fixture
def mock_empty_query_response():
    """Mock response for empty graph query"""
    return []

@pytest.fixture
def mock_single_node_response():
    """Mock response with single node, no edges"""
    return [
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53", "description": "Tumor protein p53"},
            "tid": None,
            "tlabels": None,
            "tprops": None,
            "rid": None,
            "rtype": None,
            "rprops": None,
        }
    ]

@pytest.fixture
def mock_complex_graph_response():
    """Mock response with multiple nodes and edges"""
    return [
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53", "description": "Tumor protein p53"},
            "tid": 2,
            "tlabels": ["Pathway"],
            "tprops": {"name": "Apoptosis", "description": "Cell death pathway"},
            "rid": 10,
            "rtype": "REGULATES",
            "rprops": {"strength": 0.95, "evidence": "experimental"},
        },
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53", "description": "Tumor protein p53"},
            "tid": 3,
            "tlabels": ["Disease"],
            "tprops": {"name": "Cancer", "type": "oncology"},
            "rid": 11,
            "rtype": "ASSOCIATED_WITH",
            "rprops": {"confidence": 0.87},
        },
        {
            "sid": 4,
            "slabels": ["Protein"],
            "sprops": {"name": "p53", "molecular_weight": 53000},
            "tid": 2,
            "tlabels": ["Pathway"],
            "tprops": {"name": "Apoptosis", "description": "Cell death pathway"},
            "rid": 12,
            "rtype": "PARTICIPATES_IN",
            "rprops": {"role": "activator"},
        }
    ]

@pytest.fixture
def test_client():
    """Create test client with the FastAPI app"""
    app = create_app()
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Valid authorization headers"""
    return {"Authorization": "Basic dGVzdDp0ZXN0"}

@pytest.fixture
def invalid_auth_headers():
    """Invalid authorization headers"""
    return {"Authorization": "Basic aW52YWxpZDppbnZhbGlk"}

def test_graph_explorer_basic_functionality(monkeypatch, test_client, auth_headers):
    """Test basic graph explorer functionality with mocked data"""
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
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

    node_ids = [node["id"] for node in data["nodes"]]
    assert 1 in node_ids
    assert 2 in node_ids

    edge = data["edges"][0]
    assert edge["source"] == 1
    assert edge["target"] == 2

def test_graph_explorer_empty_response(monkeypatch, test_client, auth_headers, mock_empty_query_response):
    """Test graph explorer with empty database response"""
    async def fake_execute(*args, **kwargs):
        return mock_empty_query_response

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []

def test_graph_explorer_single_node(monkeypatch, test_client, auth_headers, mock_single_node_response):
    """Test graph explorer with single node, no relationships"""
    async def fake_execute(*args, **kwargs):
        return mock_single_node_response

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 1
    assert len(data["edges"]) == 0

    node = data["nodes"][0]
    assert node["id"] == 1
    assert node["labels"] == ["Gene"]
    assert node["properties"]["name"] == "TP53"

def test_graph_explorer_complex_graph(monkeypatch, test_client, auth_headers, mock_complex_graph_response):
    """Test graph explorer with complex graph containing multiple nodes and edges"""
    async def fake_execute(*args, **kwargs):
        return mock_complex_graph_response

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 4
    assert len(data["edges"]) == 3

    node_ids = {node["id"] for node in data["nodes"]}
    assert node_ids == {1, 2, 3, 4}

    edge_types = {edge["type"] for edge in data["edges"]}
    assert "REGULATES" in edge_types
    assert "ASSOCIATED_WITH" in edge_types
    assert "PARTICIPATES_IN" in edge_types

def test_graph_explorer_missing_auth(test_client):
    """Test graph explorer without authorization header"""
    resp = test_client.get("/graph")
    assert resp.status_code == 401

def test_graph_explorer_invalid_auth(test_client, invalid_auth_headers):
    """Test graph explorer with invalid authorization"""
    resp = test_client.get("/graph", headers=invalid_auth_headers)
    assert resp.status_code == 401

def test_graph_explorer_malformed_auth(test_client):
    """Test graph explorer with malformed authorization header"""
    headers = {"Authorization": "Bearer invalid-token"}
    resp = test_client.get("/graph", headers=headers)
    assert resp.status_code == 401

@pytest.mark.parametrize("query_params,expected_calls", [
    ({}, 1),
    ({"limit": "10"}, 1),
    ({"node_type": "Gene"}, 1),
    ({"depth": "2"}, 1),
    ({"limit": "5", "node_type": "Pathway"}, 1),
])
def test_graph_explorer_query_parameters(monkeypatch, test_client, auth_headers, query_params, expected_calls):
    """Test graph explorer with various query parameters"""
    call_count = 0

    async def fake_execute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = test_client.get("/graph", headers=auth_headers, params=query_params)
    assert resp.status_code == 200
    assert call_count == expected_calls

def test_graph_explorer_invalid_query_parameters(test_client, auth_headers):
    """Test graph explorer with invalid query parameters"""
    invalid_params = {"limit": "not_a_number"}
    resp = test_client.get("/graph", headers=auth_headers, params=invalid_params)
    assert resp.status_code in [200, 400]

def test_graph_explorer_database_error(monkeypatch, test_client, auth_headers):
    """Test graph explorer when database query fails"""
    async def fake_execute_with_error(*args, **kwargs):
        raise Exception("Database connection failed")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_with_error,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 500

def test_graph_explorer_timeout_error(monkeypatch, test_client, auth_headers):
    """Test graph explorer when database query times out"""
    async def fake_execute_timeout(*args, **kwargs):
        raise TimeoutError("Query timeout")

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_timeout,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code in [500, 504]

def test_graph_explorer_malformed_database_response(monkeypatch, test_client, auth_headers):
    """Test graph explorer with malformed database response"""
    async def fake_execute_malformed(*args, **kwargs):
        return [{"invalid": "structure"}]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_malformed,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code in [200, 500]
    if resp.status_code == 200:
        data = resp.json()
        assert "nodes" in data and "edges" in data

def test_graph_explorer_post_method_not_allowed(test_client, auth_headers):
    """Test that POST method is not allowed on graph endpoint"""
    resp = test_client.post("/graph", headers=auth_headers, json={})
    assert resp.status_code == 405

def test_graph_explorer_put_method_not_allowed(test_client, auth_headers):
    """Test that PUT method is not allowed on graph endpoint"""
    resp = test_client.put("/graph", headers=auth_headers, json={})
    assert resp.status_code == 405

def test_graph_explorer_delete_method_not_allowed(test_client, auth_headers):
    """Test that DELETE method is not allowed on graph endpoint"""
    resp = test_client.delete("/graph", headers=auth_headers)
    assert resp.status_code == 405

def test_graph_explorer_response_format(monkeypatch, test_client, auth_headers):
    """Test that response has correct content type and structure"""
    async def fake_execute(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"

    data = resp.json()
    assert isinstance(data, dict)
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)

def test_graph_explorer_large_dataset(monkeypatch, test_client, auth_headers):
    """Test graph explorer with large dataset"""
    large_dataset = []
    for i in range(100):
        large_dataset.append({
            "sid": i,
            "slabels": ["Node"],
            "sprops": {"name": f"Node_{i}"},
            "tid": i + 1 if i < 99 else 0,
            "tlabels": ["Node"],
            "tprops": {"name": f"Node_{i + 1 if i < 99 else 0}"},
            "rid": 1000 + i,
            "rtype": "CONNECTS",
            "rprops": {},
        })

    async def fake_execute_large(*args, **kwargs):
        return large_dataset

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_large,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 100
    assert len(data["edges"]) == 100

def test_graph_explorer_special_characters_in_data(monkeypatch, test_client, auth_headers):
    """Test graph explorer with special characters in node/edge data"""
    async def fake_execute_special_chars(*args, **kwargs):
        return [
            {
                "sid": 1,
                "slabels": ["Gene"],
                "sprops": {"name": "TP53", "description": "Special chars: åäö & <script>"},
                "tid": 2,
                "tlabels": ["Pathway"],
                "tprops": {"name": "Apoptosis", "unicode": "αβγδε"},
                "rid": 10,
                "rtype": "LINKED",
                "rprops": {"note": "Contains 'quotes' and \"double quotes\""},
            }
        ]

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute_special_chars,
    )

    resp = test_client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

    node_names = [node["properties"]["name"] for node in data["nodes"]]
    assert "TP53" in node_names
    assert "Apoptosis" in node_names

@pytest.mark.asyncio
async def test_graph_explorer_concurrent_requests(monkeypatch, auth_headers):
    """Test graph explorer handling concurrent requests"""
    import asyncio
    from httpx import AsyncClient

    async def fake_execute(*args, **kwargs):
        await asyncio.sleep(0.1)
        return []

    monkeypatch.setattr(
        "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
        fake_execute,
    )

    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        tasks = [client.get("/graph", headers=auth_headers) for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200
            assert "nodes" in resp.json()
            assert "edges" in resp.json()

class TestGraphExplorerEdgeCases:
    """Test class for edge cases and boundary conditions"""

    def test_graph_explorer_none_values(self, monkeypatch, test_client, auth_headers):
        """Test handling of None values in database response"""
        async def fake_execute_with_nones(*args, **kwargs):
            return [
                {
                    "sid": 1,
                    "slabels": ["Gene"],
                    "sprops": {"name": "TP53", "description": None},
                    "tid": None,
                    "tlabels": None,
                    "tprops": None,
                    "rid": None,
                    "rtype": None,
                    "rprops": None,
                }
            ]

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute_with_nones,
        )

        resp = test_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 0

    def test_graph_explorer_empty_strings(self, monkeypatch, test_client, auth_headers):
        """Test handling of empty strings in database response"""
        async def fake_execute_empty_strings(*args, **kwargs):
            return [
                {
                    "sid": 1,
                    "slabels": [""],
                    "sprops": {"name": "", "description": "Valid description"},
                    "tid": 2,
                    "tlabels": ["Pathway"],
                    "tprops": {"name": "Apoptosis"},
                    "rid": 10,
                    "rtype": "",
                    "rprops": {},
                }
            ]

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute_empty_strings,
        )

        resp = test_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1