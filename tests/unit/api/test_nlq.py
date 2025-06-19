from fastapi.testclient import TestClient

from adaptive_graph_of_thoughts.app_setup import create_app


def test_nlq_endpoint(monkeypatch):
    app = create_app()
    client = TestClient(app)

    calls = []

    def fake_llm(prompt: str) -> str:
        """
        Mock implementation of a language model function for testing.
        
        Returns a Cypher query string on the first call and "summary" on subsequent calls.
        """
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
