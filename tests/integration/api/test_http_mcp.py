"""
Test script specifically designed for HTTP-based MCP server verification
"""

import json

import pytest
import requests


def test_mcp_server() -> None:
    """Send a test "initialize" request to the local MCP server and verify the response."""
    print("Testing Adaptive Graph of Thoughts MCP Server...")

    # The initialize request payload
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

    url = "http://localhost:8000/mcp"
    headers = {"Content-Type": "application/json"}

    try:
        print(f"Sending initialize request to {url}...")
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:  # pragma: no cover - network issues
        pytest.skip(f"HTTP server not available: {e}")

    result = response.json()
    print("\nSuccess! Server responded:")
    print(json.dumps(result, indent=2))

    assert "result" in result and "server_name" in result["result"]
    assert "server_version" in result["result"]
    assert "mcp_version" in result["result"]


if __name__ == "__main__":
    test_mcp_server()
