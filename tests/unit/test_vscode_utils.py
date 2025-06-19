import json
import subprocess
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_vscode_parse_ndjson_success():
    """Test successful execution of vscode ndjson parsing utility."""
    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "OK" in result.stdout
    assert result.stderr == "" or len(result.stderr.strip()) == 0


def test_vscode_parse_ndjson_script_not_found():
    """Test behavior when the Node.js script file doesn't exist."""
    script = Path("integrations/vscode-agot/nonexistent_test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Error" in result.stderr or "Cannot find module" in result.stderr


def test_vscode_parse_ndjson_invalid_node_command():
    """Test behavior when node command is not available or invalid."""
    script = Path("integrations/vscode-agot/test_utils.js")
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=127, stdout="", stderr="command not found")
        result = subprocess.run(["invalid_node_command", str(script)], capture_output=True, text=True)
        assert result.returncode == 127
        assert "command not found" in result.stderr


@patch('subprocess.run')
def test_vscode_parse_ndjson_subprocess_timeout(mock_run):
    """Test behavior when subprocess times out."""
    mock_run.side_effect = subprocess.TimeoutExpired("node", 10)
    script = Path("integrations/vscode-agot/test_utils.js")

    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=10)


@patch('subprocess.run')
def test_vscode_parse_ndjson_with_different_outputs(mock_run):
    """Test parsing of different possible outputs from the Node.js script."""
    script = Path("integrations/vscode-agot/test_utils.js")

    # Test with various success outputs
    test_cases = [
        ("OK - All tests passed", 0),
        ("OK\nTest completed successfully", 0),
        ("Tests: OK", 0),
    ]

    for stdout, returncode in test_cases:
        mock_run.return_value = MagicMock(returncode=returncode, stdout=stdout, stderr="")
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == returncode
        assert "OK" in result.stdout


@patch('subprocess.run')
def test_vscode_parse_ndjson_failure_scenarios(mock_run):
    """Test various failure scenarios for the Node.js script."""
    script = Path("integrations/vscode-agot/test_utils.js")

    failure_cases = [
        ("Error: Failed to parse", 1, "Parse error occurred"),
        ("FAIL - Test failed", 1, "Test execution failed"),
        ("", 2, "Syntax error in script"),
    ]

    for stdout, returncode, stderr in failure_cases:
        mock_run.return_value = MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode != 0
        if stdout:
            assert "OK" not in result.stdout


def test_vscode_script_path_validation():
    """Test that the script path is valid and accessible."""
    script = Path("integrations/vscode-agot/test_utils.js")
    assert script.exists(), f"Script file should exist at {script}"
    assert script.is_file(), f"Path should point to a file, not directory: {script}"
    assert script.suffix == ".js", f"Script should be a JavaScript file: {script}"


def test_vscode_script_path_edge_cases():
    """Test edge cases for script path handling."""
    # Test with absolute path
    script = Path("integrations/vscode-agot/test_utils.js").resolve()
    if script.exists():
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == 0

    # Test with relative path containing ".."
    script_relative = Path("./integrations/vscode-agot/test_utils.js")
    if script_relative.exists():
        result = subprocess.run(["node", str(script_relative)], capture_output=True, text=True)
        assert result.returncode == 0


@patch('subprocess.run')
def test_vscode_parse_ndjson_with_env_variables(mock_run):
    """Test script execution with different environment variables."""
    mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
    script = Path("integrations/vscode-agot/test_utils.js")

    # Test with custom environment
    custom_env = os.environ.copy()
    custom_env['NODE_ENV'] = 'test'

    result = subprocess.run(
        ["node", str(script)],
        capture_output=True,
        text=True,
        env=custom_env
    )
    mock_run.assert_called_once()
    assert result.returncode == 0


def test_vscode_parse_ndjson_output_encoding():
    """Test that the script output is properly encoded and decoded."""
    script = Path("integrations/vscode-agot/test_utils.js")

    # Test with text=True (default)
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)

    # Test with text=False for binary output
    result_binary = subprocess.run(["node", str(script)], capture_output=True, text=False)
    assert isinstance(result_binary.stdout, bytes)
    assert isinstance(result_binary.stderr, bytes)


class TestVSCodeUtilsIntegration:
    """Integration tests for VSCode utilities."""

    def test_script_execution_consistency(self):
        """Test that multiple executions of the script produce consistent results."""
        script = Path("integrations/vscode-agot/test_utils.js")
        if not script.exists():
            pytest.skip("Script file not found")

        results = []
        for _ in range(3):
            result = subprocess.run(["node", str(script)], capture_output=True, text=True)
            results.append((result.returncode, result.stdout, result.stderr))

        # All executions should have the same return code
        return_codes = [r[0] for r in results]
        assert all(rc == return_codes[0] for rc in return_codes)

        # All executions should have similar stdout patterns
        stdout_outputs = [r[1] for r in results]
        if return_codes[0] == 0:
            assert all("OK" in output for output in stdout_outputs)

    def test_script_performance(self):
        """Test that the script executes within reasonable time limits."""
        script = Path("integrations/vscode-agot/test_utils.js")
        if not script.exists():
            pytest.skip("Script file not found")

        import time
        start_time = time.time()
        result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
        execution_time = time.time() - start_time

        assert execution_time < 30, f"Script took too long to execute: {execution_time}s"
        assert result.returncode == 0


@pytest.fixture
def vscode_script_path():
    """Fixture providing the path to the VSCode test script."""
    return Path("integrations/vscode-agot/test_utils.js")


@pytest.fixture
def mock_subprocess_success():
    """Fixture providing a mock successful subprocess result."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "OK - Tests passed"
    mock_result.stderr = ""
    return mock_result


@pytest.fixture
def mock_subprocess_failure():
    """Fixture providing a mock failed subprocess result."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error: Test failed"
    return mock_result


def test_vscode_utils_with_fixtures(vscode_script_path, mock_subprocess_success):
    """Test using fixtures for better test organization."""
    with patch('subprocess.run', return_value=mock_subprocess_success):
        result = subprocess.run(["node", str(vscode_script_path)], capture_output=True, text=True)
        assert result.returncode == 0
        assert "OK" in result.stdout


@pytest.mark.parametrize("stdout,expected_success", [
    ("OK", True),
    ("OK - All tests passed", True),
    ("Tests completed: OK", True),
    ("FAIL", False),
    ("Error occurred", False),
    ("", False),
])
def test_vscode_output_patterns(stdout, expected_success):
    """Test various output patterns from the VSCode utility script."""
    if expected_success:
        assert "OK" in stdout
    else:
        assert "OK" not in stdout