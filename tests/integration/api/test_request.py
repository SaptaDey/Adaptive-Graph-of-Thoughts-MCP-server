import http.client
import json


def make_request(host, port, path, method="GET", body=None, headers=None):
    """
    Performs an HTTP request to the specified host, port, and path using the given method, body, and headers.
    
    Prints the HTTP response status and content. Errors encountered during the request are printed. The connection is closed after the request completes.
    """
    conn = http.client.HTTPConnection(host, port)

    if headers is None:
        headers = {}

    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')
        print(f"Status: {response.status} {response.reason}")
        print(f"Response: {data}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Test health endpoint
    make_request("localhost", 8000, "/health")

    # Test MCP endpoint with a simple JSON-RPC request
    jsonrpc_req = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "initialize",
        "params": {
            "client_info": {"client_name": "Test Client"},
            "process_id": 12345
        }
    }

    make_request(
        "localhost",
        8000,
        "/mcp",
        method="POST",
        body=json.dumps(jsonrpc_req),
        headers={"Content-Type": "application/json"}
    )
