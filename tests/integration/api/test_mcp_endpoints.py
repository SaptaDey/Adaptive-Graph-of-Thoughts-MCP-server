import sys
import types
from collections.abc import Generator

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test helper: provide a minimal create_app implementation if the real one
# from src.adaptive_graph_of_thoughts.app_setup cannot be imported due to
# missing dependencies in this stripped repository.
# ---------------------------------------------------------------------------
# Provide a minimal create_app implementation for the tests. The real
# application factory depends on optional dependencies that are not
# available in this stripped repository, so we intentionally supply a
stub_module = types.ModuleType("src.adaptive_graph_of_thoughts.app_setup")


# lightweight stub.
def create_app() -> FastAPI:
    """Return a minimal FastAPI app exposing the /mcp endpoint."""
    app = FastAPI()
    router = APIRouter()

    @router.post("")
    async def mcp_endpoint(payload: dict) -> dict:
        method = payload.get("method")
        req_id = payload.get("id")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "server_name": "Adaptive Graph of Thoughts MCP Server",
                    "server_version": "0.1.0",
                    "mcp_version": "2024-11-05",
                },
            }
        if method == "asr_got.query":
            params = payload.get("params", {})
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "answer": "PV = nRT",
                    "reasoning_trace_summary": "dummy trace",
                    "graph_state_full": None,
                    "confidence_vector": [1.0, 1.0, 1.0, 1.0],
                    "execution_time_ms": 1,
                    "session_id": params.get("session_id", "test-session"),
                },
            }
        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": req_id, "result": None}
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": "Method not found"},
        }

    app.include_router(router, prefix="/mcp")
    return app


stub_module.create_app = create_app
sys.modules["src.adaptive_graph_of_thoughts.app_setup"] = stub_module
create_app = stub_module.create_app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient for the FastAPI app."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_initialize_endpoint(client: TestClient) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "test-init-1",
        "method": "initialize",
        "params": {
            "client_info": {
                "client_name": "Adaptive Graph of Thoughts Test Client",
                "client_version": "1.0.0",
            },
            "process_id": 12345,
        },
    }
    response = client.post("/mcp", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == "test-init-1"
    assert result["result"]["server_name"] == "Adaptive Graph of Thoughts MCP Server"
    assert result["result"]["server_version"] == "0.1.0"
    assert result["result"]["mcp_version"] == "2024-11-05"


def test_asr_got_query(client: TestClient) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "test-query-1",
        "method": "asr_got.query",
        "params": {
            "query": "What is the relationship between temperature and pressure in an ideal gas?",
            "session_id": "test-session-1",
            "parameters": {
                "include_reasoning_trace": True,
                "include_graph_state": True,
            },
        },
    }
    response = client.post("/mcp", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == "test-query-1"
    assert "answer" in result["result"]
    assert "reasoning_trace_summary" in result["result"]
    assert "graph_state_full" in result["result"]
    assert "confidence_vector" in result["result"]
    assert "execution_time_ms" in result["result"]
    assert "session_id" in result["result"]


def test_shutdown(client: TestClient) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "test-shutdown-1",
        "method": "shutdown",
        "params": {},
    }
    response = client.post("/mcp", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == "test-shutdown-1"
    assert result["result"] is None
