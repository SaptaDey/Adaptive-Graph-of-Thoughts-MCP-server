import json
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest
import tempfile
import os


def test_vscode_parse_ndjson_integration():
    """Integration test that runs the actual JavaScript test file."""
    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "OK" in result.stdout


@patch("subprocess.run")
def test_vscode_parse_ndjson_success_mock(mock_run):
    """Test successful execution of vscode test utils with mocked subprocess."""
    # Mock successful execution
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Test passed: OK"
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    # Verify subprocess was called correctly
    mock_run.assert_called_once_with(
        ["node", str(script)], capture_output=True, text=True
    )

    # Verify results
    assert result.returncode == 0
    assert "OK" in result.stdout


@patch("subprocess.run")
def test_vscode_parse_ndjson_failure_returncode(mock_run):
    """Test handling of non-zero return code from JavaScript test."""
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stdout = "Test failed"
    mock_result.stderr = "Error: Test execution failed"
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "OK" not in result.stdout


@patch("subprocess.run")
def test_vscode_parse_ndjson_missing_ok_in_output(mock_run):
    """Test when subprocess succeeds but doesn't contain expected 'OK' output."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Test completed without OK marker"
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "OK" not in result.stdout


@patch("subprocess.run")
def test_vscode_parse_ndjson_subprocess_exception(mock_run):
    """Test handling of subprocess execution exceptions."""
    mock_run.side_effect = FileNotFoundError("node command not found")

    script = Path("integrations/vscode-agot/test_utils.js")

    with pytest.raises(FileNotFoundError):
        subprocess.run(["node", str(script)], capture_output=True, text=True)


@patch("subprocess.run")
def test_vscode_parse_ndjson_timeout_exception(mock_run):
    """Test handling of subprocess timeout."""
    mock_run.side_effect = subprocess.TimeoutExpired("node", 30)

    script = Path("integrations/vscode-agot/test_utils.js")

    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ["node", str(script)], capture_output=True, text=True, timeout=30
        )


def test_vscode_test_script_path_exists():
    """Test that the JavaScript test file exists."""
    script = Path("integrations/vscode-agot/test_utils.js")
    assert script.exists(), f"Test script {script} should exist"
    assert script.is_file(), f"Test script {script} should be a file"


def test_vscode_test_script_path_handling():
    """Test various path handling scenarios."""
    # Test with string path
    script_str = "integrations/vscode-agot/test_utils.js"
    script_path = Path(script_str)
    assert str(script_path) == script_str

    # Test with Path object
    script_path = Path("integrations/vscode-agot/test_utils.js")
    assert isinstance(script_path, Path)

    # Test path resolution
    resolved_path = script_path.resolve()
    assert resolved_path.is_absolute()


@patch("subprocess.run")
def test_vscode_parse_ndjson_with_stderr_output(mock_run):
    """Test handling when subprocess has stderr output but still succeeds."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Processing... OK"
    mock_result.stderr = "Warning: Some non-critical issue"
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "OK" in result.stdout
    assert "Warning" in result.stderr


@patch("subprocess.run")
def test_vscode_parse_ndjson_empty_output(mock_run):
    """Test handling of empty stdout from subprocess."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "OK" not in result.stdout


@patch("subprocess.run")
def test_vscode_parse_ndjson_case_sensitive_ok_check(mock_run):
    """Test that OK check is case sensitive."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Test completed: ok"  # lowercase
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "OK" not in result.stdout  # Should fail because "ok" != "OK"
    assert "ok" in result.stdout


@patch("subprocess.run")
def test_vscode_parse_ndjson_multiple_ok_in_output(mock_run):
    """Test when output contains multiple instances of OK."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Test 1: OK\nTest 2: OK\nAll tests: OK"
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "OK" in result.stdout
    assert result.stdout.count("OK") == 3


class TestVSCodeTestUtilsEdgeCases:
    """Test class for edge cases and boundary conditions."""

    @patch("subprocess.run")
    def test_very_long_output(self, mock_run):
        """Test handling of very long output from subprocess."""
        long_output = "A" * 10000 + " OK " + "B" * 10000
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = long_output
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        script = Path("integrations/vscode-agot/test_utils.js")
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)

        assert result.returncode == 0
        assert "OK" in result.stdout
        assert len(result.stdout) == 20005  # 10000 + 4 (" OK ") + 10000 + 1

    @patch("subprocess.run")
    def test_unicode_characters_in_output(self, mock_run):
        """Test handling of unicode characters in subprocess output."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Test résultat: ✓ OK 🎉"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        script = Path("integrations/vscode-agot/test_utils.js")
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)

        assert result.returncode == 0
        assert "OK" in result.stdout
        assert "✓" in result.stdout
        assert "résultat" in result.stdout


@pytest.fixture
def temp_js_file():
    """Fixture to create temporary JavaScript test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write('console.log("Test OK");')
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    os.unlink(temp_path)


def test_vscode_parse_ndjson_with_temp_file(temp_js_file):
    """Test with a temporary JavaScript file that we know will work."""
    result = subprocess.run(["node", str(temp_js_file)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "OK" in result.stdout


@pytest.mark.parametrize(
    "return_code,stdout,stderr,should_have_ok",
    [
        (0, "Test passed OK", "", True),
        (0, "Test passed ok", "", False),  # Case sensitive
        (0, "EVERYTHING IS OK", "", True),
        (1, "Test failed OK", "Error occurred", True),  # OK in stdout even with failure
        (0, "", "", False),  # Empty output
        (0, "No success marker", "", False),
        (127, "Command not found", "", False),
    ],
)
@patch("subprocess.run")
def test_vscode_parse_ndjson_parametrized(
    mock_run, return_code, stdout, stderr, should_have_ok
):
    """Parametrized test for various subprocess outcomes."""
    mock_result = Mock()
    mock_result.returncode = return_code
    mock_result.stdout = stdout
    mock_result.stderr = stderr
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == return_code
    assert ("OK" in result.stdout) == should_have_ok
    assert result.stderr == stderr
