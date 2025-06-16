import os
import subprocess
import pytest

# Find project root and construct path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "run_mcp_inspector.sh")

def test_inspector_http():
    # Ensure the script is executable
    os.chmod(SCRIPT_PATH, 0o755)
    result = subprocess.run(
        [SCRIPT_PATH, "http"], # Execute shell script directly
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"HTTP test failed: {result.stderr}"
    assert "MCP Inspector test for http transport passed." in result.stdout

@pytest.mark.skip(reason="STDIO transport requires interactive Inspector UI")
def test_inspector_stdio():
    os.chmod(SCRIPT_PATH, 0o755)
    result = subprocess.run(
        [SCRIPT_PATH, "stdio"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"STDIO test failed: {result.stderr}"
    assert "MCP Inspector started successfully" in result.stdout
