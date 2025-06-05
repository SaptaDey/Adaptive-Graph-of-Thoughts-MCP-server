"""
Very simple test script using low-level socket connections to test the MCP server.
"""
import json
import socket


def send_tcp_request(host='localhost', port=8000, path='/health', method='GET'):
    """Send a raw TCP request to a server and print the response."""
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout

        # Connect to server
        print(f"Connecting to {host}:{port}...")
        sock.connect((host, port))

        # Create HTTP request
        request = f"{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"

        # Send request
        print(f"Sending request: {request.strip()}")
        sock.sendall(request.encode('utf-8'))

        # Receive all data
        print("Waiting for response...")
        response = b''
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data

        # Close socket
        sock.close()

        # Decode and print response
        response_str = response.decode('utf-8')
        print("\n--- Response received ---")
        print(response_str)
        print("------------------------\n")

        return response_str
    except socket.timeout:
        print("Connection timed out")
        return None
    except ConnectionRefusedError:
        print("Connection refused. Server might not be running.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def send_jsonrpc_request(host='localhost', port=8000, path='/mcp', method='initialize'):
    """Send a JSON-RPC request to the MCP server."""
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout

        # Connect to server
        print(f"Connecting to {host}:{port}...")
        sock.connect((host, port))

        # Create JSON-RPC payload
        payload = {
            "jsonrpc": "2.0",
            "id": "test-init-1",
            "method": method,
            "params": {
                "client_info": {
                    "client_name": "Socket Test Client",
                    "client_version": "1.0.0"
                },
                "process_id": 12345
            }
        }

        payload_json = json.dumps(payload)
        content_length = len(payload_json)

        # Create HTTP request
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {content_length}\r\n"
            f"Connection: close\r\n\r\n"
            f"{payload_json}"
        )

        # Send request
        print(f"Sending request to {path}...")
        print(f"Payload: {payload_json}")
        sock.sendall(request.encode('utf-8'))

        # Receive all data
        print("Waiting for response...")
        response = b''
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data

        # Close socket
        sock.close()

        # Decode and print response
        response_str = response.decode('utf-8')
        print("\n--- Response received ---")
        print(response_str)
        print("------------------------\n")

        # Try to extract and parse JSON from the HTTP response
        if "Content-Type: application/json" in response_str:
            try:
                body_start = response_str.find('\r\n\r\n') + 4
                json_str = response_str[body_start:]
                json_data = json.loads(json_str)
                print("Parsed JSON response:")
                print(json.dumps(json_data, indent=2))
            except json.JSONDecodeError:
                print("Could not parse JSON from response")
            except Exception as e:
                print(f"Error processing JSON: {e}")

        return response_str
    except socket.timeout:
        print("Connection timed out")
        return None
    except ConnectionRefusedError:
        print("Connection refused. Server might not be running.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Test the health endpoint
    print("=== Testing Health Endpoint ===")
    health_response = send_tcp_request(path='/health')

    if health_response and "200 OK" in health_response:
        print("Health check successful!")

        # Test the MCP endpoint
        print("\n=== Testing MCP Initialize Endpoint ===")
        mcp_response = send_jsonrpc_request(path='/mcp', method='initialize')

        if mcp_response and "200 OK" in mcp_response:
            print("MCP initialize request successful!")
        else:
            print("MCP initialize request failed.")
    else:
        print("Health check failed, not testing MCP endpoint.")
