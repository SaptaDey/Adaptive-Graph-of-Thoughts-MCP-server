import importlib.util
from pathlib import Path
import sys
import types

neo4j_mock = types.ModuleType("neo4j")


class FakeGraphDatabase:
    @staticmethod
    def driver(uri, auth):
        return DummyDriver()


neo4j_mock.GraphDatabase = FakeGraphDatabase
neo4j_mock.Driver = object
neo4j_mock.Record = object
neo4j_mock.Result = object
neo4j_mock.Transaction = object


def unit_of_work(func):
    return func


neo4j_mock.unit_of_work = unit_of_work

exc_mod = types.ModuleType("neo4j.exceptions")


class Neo4jError(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


exc_mod.Neo4jError = Neo4jError
exc_mod.ServiceUnavailable = ServiceUnavailable

sys.modules["neo4j"] = neo4j_mock
sys.modules["neo4j.exceptions"] = exc_mod

spec = importlib.util.spec_from_file_location(
    "neo4j_utils",
    Path(__file__).resolve().parents[4]
    / "src/adaptive_graph_of_thoughts/domain/services/neo4j_utils.py",
)
neo4j_utils = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(neo4j_utils)


class DummyDriver:
    def __init__(self):
        self.closed = False
        self.connected = False

    def verify_connectivity(self):
        self.connected = True

    def close(self):
        self.closed = True


def test_get_neo4j_driver_returns_connected_driver(monkeypatch):
    # reset global singletons
    monkeypatch.setattr(neo4j_utils, "_driver", None, raising=False)
    monkeypatch.setattr(neo4j_utils, "_neo4j_settings", None, raising=False)

    dummy = DummyDriver()

    def fake_driver(uri, auth):
        return dummy

    monkeypatch.setattr(neo4j_utils.GraphDatabase, "driver", fake_driver)

    driver = neo4j_utils.get_neo4j_driver()
    assert driver is dummy
    assert dummy.connected
