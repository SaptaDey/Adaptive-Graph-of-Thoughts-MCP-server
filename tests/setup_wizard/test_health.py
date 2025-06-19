from fastapi.testclient import TestClient

from adaptive_graph_of_thoughts.app_setup import create_app


def test_health_ok(monkeypatch):
    """
    Test that the /health endpoint returns status 200 and reports Neo4j as "up" when the database connection succeeds.
    """
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            """
            Return a context manager simulating a Neo4j database session.
            
            The returned object supports use in a `with` statement and provides a no-op `run` method.
            """
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass

            return S()

        def close(self):
            """
            Closes the driver connection.
            
            This method is a placeholder and does not perform any action.
            """
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_down(monkeypatch):
    """
    Test that the /health endpoint returns a 500 status and reports Neo4j as "down" when the database driver fails to create a session.
    """
    app = create_app()
    client = TestClient(app)

    class BadDriver:
        def session(self, **_kw):
            """
            Raises an exception to simulate a failure when attempting to create a database session.
            """
            raise Exception("fail")

        def close(self):
            """
            Closes the driver connection.
            
            This method is a placeholder and does not perform any action.
            """
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: BadDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
