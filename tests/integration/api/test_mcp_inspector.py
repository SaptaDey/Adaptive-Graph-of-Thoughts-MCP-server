import os
import subprocess
import sys
import time
import signal
import threading
from pathlib import Path
import pytest
import requests
from unittest.mock import patch

class ProcessManager:
    """Manages background processes for HTTP server testing."""
    
    def __init__(self):
        self.processes = []
    
    def start_http_server(self, timeout=30):
        """Start HTTP server in background and wait for it to be ready."""
        print("Starting HTTP server for testing...")
        
        # Change to project root directory
        project_root = Path(__file__).parent.parent.parent
        
        # Start server process
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "src.adaptive_graph_of_thoughts.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group
            cwd=project_root
        )
        self.processes.append(process)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                r = requests.get("http://localhost:8000/health", timeout=5)
                if r.status_code == 200:
                    print("HTTP server is ready!")
                    return process
            except requests.exceptions.RequestException:
                pass

            if process.poll() is not None:
                out, err = process.communicate()
                raise RuntimeError(f"HTTP server failed to start: {err.decode()}")
            time.sleep(2)

        raise TimeoutError("HTTP server failed to start within timeout")
    
    def cleanup(self):
        """Clean up all managed processes."""
        for proc in self.processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

def run_with_retry(command, max_retries=2, timeout=180):
    """Run command with retry logic and proper timeout handling."""
    for attempt in range(max_retries + 1):
        try:
            print(f"Attempt {attempt + 1}/{max_retries + 1}: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                print("✅ MCP Inspector validation passed!")
                print("Output:", result.stdout)
                return True
            print(f"❌ Exit {result.returncode}. Stdout: {result.stdout} Stderr: {result.stderr}")
            if attempt < max_retries:
                wait = 5 * (attempt + 1)
                print(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)
        except subprocess.TimeoutExpired:
            print(f"⏰ Timed out after {timeout}s")
            if attempt < max_retries:
                print("🔄 Retrying...")
                time.sleep(5)
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            if attempt < max_retries:
                print("🔄 Retrying...")
                time.sleep(5)
    return False

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment and ensure dependencies are available."""
    try:
        res = subprocess.run(
            ["mcp-inspector", "--version"],
            capture_output=True,
            timeout=10
        )
        if res.returncode != 0:
            pytest.skip("mcp-inspector not available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("mcp-inspector not installed or not in PATH")

    original_cwd = os.getcwd()
    os.chdir(Path(__file__).parent.parent.parent)
    yield
    os.chdir(original_cwd)

def test_inspector_http():
    """Test MCP Inspector with HTTP transport using improved process management."""
    pm = ProcessManager()
    try:
        pm.start_http_server(timeout=30)
        cmd = ["mcp-inspector", "-v", "validate", "http://localhost:8000/mcp"]
        assert run_with_retry(cmd, max_retries=2, timeout=120), \
            "MCP Inspector HTTP validation failed after retries"
    except Exception as e:
        pytest.fail(f"HTTP server setup failed: {e}")
    finally:
        print("🧹 Cleaning up HTTP server...")
        pm.cleanup()

@pytest.mark.parametrize("transport_mode", ["stdio"])
def test_inspector_stdio(transport_mode):
    """Test MCP Inspector with STDIO transport using improved protocol handling."""
    os.chdir(Path(__file__).parent.parent.parent)
    cmd = [
        "mcp-inspector",
        "-v",
        "validate",
        "stdio",
        "--program",
        sys.executable,
        "src/adaptive_graph_of_thoughts/main_stdio.py",
    ]
    assert run_with_retry(cmd, max_retries=2, timeout=180), \
        f"MCP Inspector {transport_mode} validation failed after retries"

def test_inspector_both_transports():
    """Test MCP Inspector with both HTTP and STDIO transports comprehensively."""
    pm = ProcessManager()
    results = {}
    try:
        print("\n" + "="*50 + "\nTesting STDIO transport\n" + "="*50)
        stdio_cmd = [
            "mcp-inspector", "-v", "validate", "stdio",
            "--program", sys.executable, "src/adaptive_graph_of_thoughts/main_stdio.py",
        ]
        results['stdio'] = run_with_retry(stdio_cmd, max_retries=2, timeout=180)

        print("\n" + "="*50 + "\nTesting HTTP transport\n" + "="*50)
        pm.start_http_server(timeout=30)
        http_cmd = ["mcp-inspector", "-v", "validate", "http://localhost:8000/mcp"]
        results['http'] = run_with_retry(http_cmd, max_retries=2, timeout=120)
    except Exception as e:
        pytest.fail(f"Transport testing failed: {e}")
    finally:
        pm.cleanup()

    print(f"\n📊 Test Results:\nSTDIO: {'✅ PASS' if results.get('stdio') else '❌ FAIL'}\n" +
          f"HTTP: {'✅ PASS' if results.get('http') else '❌ FAIL'}")
    assert results.get('stdio'), "STDIO transport validation failed"
    assert results.get('http'), "HTTP transport validation failed"

def test_mcp_configuration_validity():
    """Test that MCP configuration is valid and includes necessary timeout settings."""
    cfg = Path("config/settings.yaml")
    assert cfg.exists(), "settings.yaml configuration file not found"
    content = cfg.read_text()
    for key in ("mcp_stdio_timeout", "mcp_http_timeout", "mcp_inspector_retries"):
        assert key in content, f"Missing {key} setting"
    print("✅ MCP configuration includes required timeout and retry settings")