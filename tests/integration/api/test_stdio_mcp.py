import json
import os
import time
import subprocess
import select
import pytest
from pathlib import Path

from adaptive_graph_of_thoughts.config import settings

@pytest.fixture(scope="module")
def stdio_process():
    """Start the MCP STDIO server as a subprocess."""
    cmd = [
        "python",
        "-m",
        "adaptive_graph_of_thoughts.main_stdio"
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # Allow server to initialize
    time.sleep(2)
    yield proc
    proc.terminate()
    proc.wait()

def test_stdio_initialize(stdio_process):
    """Test MCP initialize via STDIO transport."""
    request = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": {
            "client_info": {
                "client_name": "pytest-client",
                "client_version": "1.0.0"
            },
            "process_id": os.getpid()
        }
    }
    # Send request
    line = json.dumps(request) + "\n"
    response_line = None
    response = None
    try:
        stdio_process.stdin.write(line)
        stdio_process.stdin.flush()
        # Wait up to 5 seconds for a response line
        ready, _, _ = select.select([stdio_process.stdout], [], [], 5)
        if ready:
            response_line = stdio_process.stdout.readline()
        else:
            pytest.fail("Timed out waiting for STDIO response")
        if not response_line: # Handle empty readline if process exited
            stderr_output = stdio_process.stderr.read()
            pytest.fail(f"STDIO process exited prematurely. stderr:\n{stderr_output}")
        response = json.loads(response_line)
    except BrokenPipeError as e:
        stderr_output = stdio_process.stderr.read()
        pytest.fail(f"BrokenPipeError encountered. Subprocess stderr:\n{stderr_output}\nOriginal error: {e}")
    except Exception as e:
        # Catch other potential errors like json.JSONDecodeError if response_line is not valid JSON
        stderr_output = stdio_process.stderr.read()
        pytest.fail(f"An unexpected error occurred. Subprocess stderr:\n{stderr_output}\nOriginal error: {e}\nResponse line was: '{response_line}'")

    # Assertions
    assert response is not None, "Response was not successfully parsed."
    assert response.get("id") == "init-1"
    assert "result" in response
    result_data = response.get("result", {})
    assert "server_name" in result_data and isinstance(result_data["server_name"], str)
    assert "server_version" in result_data and isinstance(result_data["server_version"], str)
    assert "mcp_version" in result_data and isinstance(result_data["mcp_version"], str)

@pytest.mark.parametrize("query", ["test question"])
def test_stdio_call_tool(stdio_process, query):
    """Test calling the asr_got_query tool over STDIO."""
    request = {
        "jsonrpc": "2.0",
        "id": "tool-1",
        "method": "callTool",
        "params": {
            "name": "asr_got_query",
            "arguments": {"query": query},
            "client_info": {}
        }
    }
    response_line = None
    response = None
    try:
        stdio_process.stdin.write(json.dumps(request) + "\n")
        stdio_process.stdin.flush()
        ready, _, _ = select.select([stdio_process.stdout], [], [], 5)
        if ready:
            response_line = stdio_process.stdout.readline()
        else:
            pytest.fail("Timed out waiting for STDIO response")
        if not response_line: # Handle empty readline if process exited
            stderr_output = stdio_process.stderr.read()
            pytest.fail(f"STDIO process exited prematurely. stderr:\n{stderr_output}")
        response = json.loads(response_line)
    except BrokenPipeError as e:
        stderr_output = stdio_process.stderr.read()
        pytest.fail(f"BrokenPipeError encountered. Subprocess stderr:\n{stderr_output}\nOriginal error: {e}")
    except Exception as e:
        # Catch other potential errors like json.JSONDecodeError if response_line is not valid JSON
        stderr_output = stdio_process.stderr.read()
        pytest.fail(f"An unexpected error occurred. Subprocess stderr:\n{stderr_output}\nOriginal error: {e}\nResponse line was: '{response_line}'")

    assert response is not None, "Response was not successfully parsed."
    assert response.get("id") == "tool-1"
    assert "result" in response or "error" in response