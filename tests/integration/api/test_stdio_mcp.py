import json
import os
import time
import subprocess
import pytest
from pathlib import Path

from asr_got_reimagined.config import settings

@pytest.fixture(scope="module")
def stdio_process():
    """Start the MCP STDIO server as a subprocess."""
    cmd = [
        "python",
        "-m",
        "asr_got_reimagined.main_stdio"
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
        # Read and parse one line of response
        response_line = stdio_process.stdout.readline()
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
    server_info = response["result"].get("server_info", {})
    assert "name" in server_info and isinstance(server_info["name"], str)
    assert "version" in server_info and isinstance(server_info["version"], str)

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
        response_line = stdio_process.stdout.readline()
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