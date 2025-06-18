from fastapi.testclient import TestClient

from adaptive_graph_of_thoughts.app_setup import create_app


def test_health_ok(monkeypatch):
    app = create_app()
    client = TestClient(app)

    class GoodDriver:
        def session(self, **_kw):
            class S:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

                def run(self, _q):
                    pass

            return S()

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: GoodDriver())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["neo4j"] == "up"


def test_health_down(monkeypatch):
    app = create_app()
    client = TestClient(app)

    class BadDriver:
        def session(self, **_kw):
            raise Exception("fail")

        def close(self):
            pass

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_a, **_k: BadDriver())
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json()["neo4j"] == "down"
