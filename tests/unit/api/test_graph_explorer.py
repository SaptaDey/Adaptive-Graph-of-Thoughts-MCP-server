from fastapi.testclient import TestClient
from adaptive_graph_of_thoughts.app_setup import create_app


def test_graph_explorer(monkeypatch):

    async def fake_query(query: str, params=None, database=None, tx_type="read"):
        """
        Simulate a database query by returning a fixed list representing a single graph edge between two nodes.
        
        Parameters:
        	query (str): The query string to simulate.
        
        Returns:
        	list: A list containing one dictionary representing an edge with source and target node details.
        """
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
        """
        Asynchronously executes a fake query and returns a predefined graph result.
        
        Returns:
            The result of the fake_query function, representing a fixed graph structure.
        """
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
