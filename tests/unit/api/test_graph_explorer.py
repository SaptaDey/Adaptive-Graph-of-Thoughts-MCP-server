import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from adaptive_graph_of_thoughts.app_setup import create_app

@pytest.fixture
def mock_client():
    """Create a test client for the application."""
    app = create_app()
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Standard authentication headers for tests."""
    return {"Authorization": "Basic dGVzdDp0ZXN0"}

@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing."""
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
def empty_graph_data():
    """Empty graph data for testing."""
    return []

@pytest.fixture
def complex_graph_data():
    """Complex graph data with multiple nodes and relationships."""
    return [
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53", "description": "Tumor protein p53"},
            "tid": 2,
            "tlabels": ["Pathway"],
            "tprops": {"name": "Apoptosis", "category": "Cell Death"},
            "rid": 10,
            "rtype": "LINKED",
            "rprops": {"strength": 0.8},
        },
        {
            "sid": 2,
            "slabels": ["Pathway"],
            "sprops": {"name": "Apoptosis", "category": "Cell Death"},
            "tid": 3,
            "tlabels": ["Disease"],
            "tprops": {"name": "Cancer", "severity": "High"},
            "rid": 11,
            "rtype": "ASSOCIATED_WITH",
            "rprops": {"confidence": 0.9},
        },
        {
            "sid": 1,
            "slabels": ["Gene"],
            "sprops": {"name": "TP53", "description": "Tumor protein p53"},
            "tid": 3,
            "tlabels": ["Disease"],
            "tprops": {"name": "Cancer", "severity": "High"},
            "rid": 12,
            "rtype": "CONTRIBUTES_TO",
            "rprops": {"evidence_level": "Strong"},
        }
    ]

class TestGraphExplorer:
    """Test suite for the graph explorer API endpoint using pytest framework."""

    def test_graph_explorer_basic_functionality(self, monkeypatch, mock_client, auth_headers, sample_graph_data):
        """Test basic graph explorer functionality with valid data."""
        async def fake_execute(*args, **kwargs):
            return sample_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        # Verify node structure
        nodes = data["nodes"]
        node_ids = [node["id"] for node in nodes]
        assert "1" in node_ids
        assert "2" in node_ids

        # Verify edge structure
        edges = data["edges"]
        edge = edges[0]
        assert edge["source"] == "1"
        assert edge["target"] == "2"
        assert edge["type"] == "LINKED"

    def test_graph_explorer_empty_result(self, monkeypatch, mock_client, auth_headers, empty_graph_data):
        """Test graph explorer when no data is returned."""
        async def fake_execute(*args, **kwargs):
            return empty_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 0
        assert len(data["edges"]) == 0

    def test_graph_explorer_complex_graph(self, monkeypatch, mock_client, auth_headers, complex_graph_data):
        """Test graph explorer with complex multi-node, multi-edge graph."""
        async def fake_execute(*args, **kwargs):
            return complex_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Should have 3 unique nodes (1, 2, 3)
        assert len(data["nodes"]) == 3
        # Should have 3 edges
        assert len(data["edges"]) == 3

        # Verify all expected nodes are present
        node_ids = {node["id"] for node in data["nodes"]}
        assert node_ids == {"1", "2", "3"}

        # Verify edge relationships
        edges = data["edges"]
        edge_pairs = {(edge["source"], edge["target"]) for edge in edges}
        assert ("1", "2") in edge_pairs
        assert ("2", "3") in edge_pairs
        assert ("1", "3") in edge_pairs

    def test_graph_explorer_with_limit_parameter(self, monkeypatch, mock_client, auth_headers, sample_graph_data):
        """Test graph explorer with limit parameter."""
        def fake_execute_with_params(*args, **kwargs):
            query, params = args
            assert "LIMIT $limit" in query
            assert params["limit"] == 25
            return sample_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute_with_params,
        )

        resp = mock_client.get("/graph?limit=25", headers=auth_headers)
        assert resp.status_code == 200

    def test_graph_explorer_database_error(self, monkeypatch, mock_client, auth_headers):
        """Test graph explorer when database query fails."""
        async def fake_execute_with_error(*args, **kwargs):
            raise Exception("Database connection failed")

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute_with_error,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 500

    def test_graph_explorer_node_properties_structure(self, monkeypatch, mock_client, auth_headers, sample_graph_data):
        """Test that node properties are correctly structured in response."""
        async def fake_execute(*args, **kwargs):
            return sample_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Check node structure and properties
        gene_node = next(node for node in data["nodes"] if "Gene" in node["labels"])
        assert gene_node["properties"]["name"] == "TP53"

        pathway_node = next(node for node in data["nodes"] if "Pathway" in node["labels"])
        assert pathway_node["properties"]["name"] == "Apoptosis"

    def test_graph_explorer_edge_properties_structure(self, monkeypatch, mock_client, auth_headers, sample_graph_data):
        """Test that edge properties are correctly structured in response."""
        async def fake_execute(*args, **kwargs):
            return sample_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Check edge structure
        edge = data["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge
        assert "properties" in edge
        assert edge["type"] == "LINKED"
        assert edge["id"] == "10"

    def test_graph_explorer_duplicate_nodes_handling(self, monkeypatch, mock_client, auth_headers):
        """Test graph explorer handles duplicate nodes correctly."""
        duplicate_data = [
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
            },
            {
                "sid": 1,
                "slabels": ["Gene"],
                "sprops": {"name": "TP53"},
                "tid": 3,
                "tlabels": ["Disease"],
                "tprops": {"name": "Cancer"},
                "rid": 11,
                "rtype": "ASSOCIATED",
                "rprops": {},
            }
        ]

        async def fake_execute(*args, **kwargs):
            return duplicate_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Should deduplicate nodes - only unique node IDs
        node_ids = [node["id"] for node in data["nodes"]]
        assert len(set(node_ids)) == len(data["nodes"])
        assert len(data["edges"]) == 2

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_graph_explorer_unsupported_methods(self, mock_client, auth_headers, method):
        """Test that unsupported HTTP methods return 405."""
        resp = getattr(mock_client, method.lower())("/graph", headers=auth_headers)
        assert resp.status_code == 405

    def test_graph_explorer_response_headers(self, monkeypatch, mock_client, auth_headers, sample_graph_data):
        """Test that response has correct headers."""
        async def fake_execute(*args, **kwargs):
            return sample_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]

    @pytest.mark.parametrize("limit", [1, 10, 50, 100])
    def test_graph_explorer_various_limits(self, monkeypatch, mock_client, auth_headers, sample_graph_data, limit):
        """Test graph explorer with various limit values."""
        def fake_execute_check_limit(*args, **kwargs):
            query, params = args
            assert params["limit"] == limit
            return sample_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute_check_limit,
        )

        resp = mock_client.get(f"/graph?limit={limit}", headers=auth_headers)
        assert resp.status_code == 200

    def test_graph_explorer_invalid_limit_parameter(self, monkeypatch, mock_client, auth_headers, sample_graph_data):
        """Test graph explorer with invalid limit parameter."""
        async def fake_execute(*args, **kwargs):
            return sample_graph_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph?limit=abc", headers=auth_headers)
        assert resp.status_code == 422

    def test_graph_explorer_missing_node_properties(self, monkeypatch, mock_client, auth_headers):
        """Test graph explorer with missing node properties."""
        malformed_data = [
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

        async def fake_execute(*args, **kwargs):
            return malformed_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Should handle None properties gracefully
        gene_node = next(node for node in data["nodes"] if "Gene" in node["labels"])
        assert gene_node["properties"] is None

    def test_graph_explorer_unicode_properties(self, monkeypatch, mock_client, auth_headers):
        """Test graph explorer with unicode characters in properties."""
        unicode_data = [
            {
                "sid": 1,
                "slabels": ["Gene"],
                "sprops": {"name": "TP53", "description": "Протеин p53 🧬"},
                "tid": 2,
                "tlabels": ["Pathway"],
                "tprops": {"name": "Apoptosis", "description": "細胞死 💀"},
                "rid": 10,
                "rtype": "LINKED",
                "rprops": {"note": "связь"},
            }
        ]

        async def fake_execute(*args, **kwargs):
            return unicode_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        gene_node = next(node for node in data["nodes"] if "Gene" in node["labels"])
        assert "🧬" in gene_node["properties"]["description"]

        pathway_node = next(node for node in data["nodes"] if "Pathway" in node["labels"])
        assert "💀" in pathway_node["properties"]["description"]

    def test_graph_explorer_large_graph_simulation(self, monkeypatch, mock_client, auth_headers):
        """Test graph explorer with simulated large dataset."""
        large_data = []
        for i in range(20):
            for j in range(i+1, min(i+3, 20)):
                large_data.append({
                    "sid": i,
                    "slabels": ["Node"],
                    "sprops": {"name": f"Node_{i}", "index": i},
                    "tid": j,
                    "tlabels": ["Node"],
                    "tprops": {"name": f"Node_{j}", "index": j},
                    "rid": i * 100 + j,
                    "rtype": "CONNECTED",
                    "rprops": {"weight": i + j},
                })

        async def fake_execute(*args, **kwargs):
            return large_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) <= 20
        assert len(data["edges"]) == len(large_data)

    def test_graph_explorer_empty_labels_and_properties(self, monkeypatch, mock_client, auth_headers):
        """Test graph explorer with empty labels and properties."""
        empty_metadata_data = [
            {
                "sid": 1,
                "slabels": [],
                "sprops": {},
                "tid": 2,
                "tlabels": [],
                "tprops": {},
                "rid": 10,
                "rtype": "LINKED",
                "rprops": {},
            }
        ]

        async def fake_execute(*args, **kwargs):
            return empty_metadata_data

        monkeypatch.setattr(
            "src.adaptive_graph_of_thoughts.api.routes.explorer.execute_query",
            fake_execute,
        )

        resp = mock_client.get("/graph", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        for node in data["nodes"]:
            assert node["labels"] == []
            assert node["properties"] == {}

# Keep the original test function for backward compatibility
def test_graph_explorer(monkeypatch):
    """Original test function - kept for backward compatibility."""
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

    app = create_app()
    client = TestClient(app)

    headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
    resp = client.get("/graph", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1