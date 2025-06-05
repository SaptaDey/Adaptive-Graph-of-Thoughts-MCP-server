"""
Test script for the NexusMind MCP server endpoints.
"""
import json
from typing import Any, Dict, Optional

import requests

MCP_SERVER_URL = "http://localhost:8000/mcp"  # The MCP endpoint

def test_initialize_endpoint() -> Dict[str, Any]:
    """
    Sends a JSON-RPC initialize request to the MCP server and validates the response.
    
    Returns:
        The parsed JSON response from the server if successful, or a dictionary
        containing error details if the request fails or the response is invalid.
    """
    print("\n=== Testing Initialize Endpoint ===")
    payload = {
        "jsonrpc": "2.0",
        "id": "test-init-1",
        "method": "initialize",
        "params": {
            "client_info": {
                "client_name": "NexusMind Test Client",
                "client_version": "1.0.0"
            },
            "process_id": 12345
        }
    }

    try:
        print(f"Sending request to {MCP_SERVER_URL}...")
        response = requests.post(
            MCP_SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Response: {json.dumps(result, indent=2)}")

            # Add assertions to verify the correctness of the initialize endpoint response
            assert result["jsonrpc"] == "2.0", "Invalid JSON-RPC version"
            assert result["id"] == "test-init-1", "Invalid response ID"
            assert "result" in result, "Missing result in response"
            assert result["result"]["server_name"] == "NexusMind MCP Server", "Invalid server name"
            assert result["result"]["server_version"] == "0.1.0", "Invalid server version"
            assert result["result"]["mcp_version"] == "2024-11-05", "Invalid MCP version"

            return result
        else:
            print(f"Error: HTTP Status {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"HTTP Status {response.status_code}", "response": response.text}
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return {"error": str(e)}

def test_asr_got_query(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends a test JSON-RPC request to the asr_got.query endpoint with a scientific question.
    
    Args:
        session_id: Optional session identifier to include in the request.
    
    Returns:
        The parsed JSON response from the server if successful, or a dictionary containing error details on failure.
    """
    print("\n=== Testing ASR-GoT Query Endpoint ===")
    payload = {
        "jsonrpc": "2.0",
        "id": "test-query-1",
        "method": "asr_got.query",
        "params": {
            "query": "What is the relationship between temperature and pressure in an ideal gas?",
            "session_id": session_id or "test-session-1",
            "parameters": {
                "include_reasoning_trace": True,
                "include_graph_state": True
            }
        }
    }

    try:
        print(f"Sending query to {MCP_SERVER_URL}...")
        response = requests.post(
            MCP_SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Answer: {json.dumps(result.get('result', {}).get('answer', 'No answer'), indent=2)}")

            # Add assertions to verify the correctness of the asr_got.query endpoint response
            assert result["jsonrpc"] == "2.0", "Invalid JSON-RPC version"
            assert result["id"] == "test-query-1", "Invalid response ID"
            assert "result" in result, "Missing result in response"
            assert "answer" in result["result"], "Missing answer in response"
            assert "reasoning_trace_summary" in result["result"], "Missing reasoning trace summary in response"
            assert "graph_state_full" in result["result"], "Missing graph state in response"
            assert "confidence_vector" in result["result"], "Missing confidence vector in response"
            assert "execution_time_ms" in result["result"], "Missing execution time in response"
            assert "session_id" in result["result"], "Missing session ID in response"

            return result
        else:
            print(f"Error: HTTP Status {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"HTTP Status {response.status_code}", "response": response.text}
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return {"error": str(e)}

def test_shutdown() -> Dict[str, Any]:
    """
    Sends a JSON-RPC shutdown request to the MCP server and returns the parsed response.
    
    Returns:
        The parsed JSON response from the server if successful, or a dictionary with error details if the request fails.
    """
    print("\n=== Testing Shutdown Endpoint ===")
    payload = {
        "jsonrpc": "2.0",
        "id": "test-shutdown-1",
        "method": "shutdown",
        "params": {}
    }

    try:
        print(f"Sending shutdown request to {MCP_SERVER_URL}...")
        response = requests.post(
            MCP_SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: HTTP Status {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"HTTP Status {response.status_code}", "response": response.text}
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return {"error": str(e)}

def run_all_tests():
    """
    Executes the MCP endpoint test suite in sequence.
    
    Runs the initialization test, then the ASR query test if initialization succeeds, using a generated session ID. Prints a completion message after all tests.
    """
    # First test the initialize endpoint
    init_response = test_initialize_endpoint()

    if "error" not in init_response:
        # If initialization succeeded, test the query endpoint
        session_id = "test-session-" + str(hash(json.dumps(init_response)))[0:8]
        query_response = test_asr_got_query(session_id)

        # Don't actually shut down the server in normal testing
        # test_shutdown()

    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    run_all_tests()
