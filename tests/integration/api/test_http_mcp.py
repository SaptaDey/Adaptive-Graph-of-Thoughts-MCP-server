"""
Test script specifically designed for HTTP-based MCP server verification
"""
import json

import requests


def test_mcp_server():
    """
    Sends a test "initialize" request to the local MCP server and verifies its response.
    
    Returns:
        True if the server responds with the expected MCP initialization fields; False otherwise.
    """
    print("Testing NexusMind MCP Server...")

    # The initialize request payload
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

    url = "http://localhost:8000/mcp"
    headers = {"Content-Type": "application/json"}

    try:
        print(f"Sending initialize request to {url}...")
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! Server responded:")
            print(json.dumps(result, indent=2))

            # Check if the response is a valid MCP response
            if "result" in result and "server_name" in result["result"]:
                print("\n✅ MCP server is working correctly!")
                print(f"Server Name: {result['result']['server_name']}")
                print(f"Server Version: {result['result']['server_version']}")
                print(f"MCP Version: {result['result']['mcp_version']}")
                return True
            else:
                print("\n❌ Response doesn't contain expected MCP initialize fields")
                return False
        else:
            print(f"\n❌ Error: HTTP Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"\n❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    test_mcp_server()
