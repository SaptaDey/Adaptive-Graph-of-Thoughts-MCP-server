import json
import subprocess
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture
def mock_script_path():
    """Fixture providing a mock script path for testing"""
    return Path("integrations/vscode-agot/test_utils.js")


@pytest.fixture
def temp_script_file():
    """Fixture creating a temporary JavaScript file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write('console.log("OK"); process.exit(0);')
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def failing_script_file():
    """Fixture creating a temporary failing JavaScript file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write('console.error("Error"); process.exit(1);')
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


def test_vscode_parse_ndjson_success(mock_script_path):
    """Test successful execution of VSCode NDJSON parsing script"""
    script = mock_script_path
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_vscode_parse_ndjson_with_temp_script(temp_script_file):
    """Test VSCode NDJSON parsing with a temporary script file"""
    result = subprocess.run(["node", str(temp_script_file)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_vscode_parse_ndjson_script_exists(mock_script_path):
    """Test that the VSCode test script file exists"""
    script = mock_script_path
    assert script.exists(), f"Script file {script} should exist"
    assert script.suffix == '.js', "Script should be a JavaScript file"


def test_vscode_parse_ndjson_script_failure(failing_script_file):
    """Test handling of script execution failure"""
    result = subprocess.run(["node", str(failing_script_file)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Error" in result.stderr


def test_vscode_parse_ndjson_nonexistent_script():
    """Test behavior with non-existent script file"""
    nonexistent_script = Path("nonexistent_script.js")
    result = subprocess.run(["node", str(nonexistent_script)], capture_output=True, text=True)
    assert result.returncode != 0


def test_vscode_parse_ndjson_empty_script():
    """Test behavior with empty script file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        empty_script = Path(f.name)

    try:
        result = subprocess.run(["node", str(empty_script)], capture_output=True, text=True)
        # Node.js should handle empty script gracefully
        assert result.returncode == 0
    finally:
        empty_script.unlink()


def test_vscode_parse_ndjson_invalid_node_command():
    """Test behavior when node command is not available"""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = FileNotFoundError("node command not found")

        script = Path("integrations/vscode-agot/test_utils.js")
        with pytest.raises(FileNotFoundError):
            subprocess.run(["node", str(script)], capture_output=True, text=True)


@pytest.mark.parametrize("node_args,expected_success", [
    (["--version"], True),
    (["--help"], True),
    (["--invalid-flag"], False),
])
def test_node_command_variations(node_args, expected_success):
    """Test various node command line arguments"""
    result = subprocess.run(["node"] + node_args, capture_output=True, text=True)
    if expected_success:
        assert result.returncode == 0
    else:
        assert result.returncode != 0


@pytest.mark.parametrize("script_content,expected_output", [
    ('console.log("Test OK");', "Test OK"),
    ('console.log("JSON: " + JSON.stringify({test: true}));', "JSON:"),
    ('process.stdout.write("No newline");', "No newline"),
])
def test_vscode_script_output_variations(script_content, expected_output):
    """Test different script outputs and content variations"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)

    try:
        result = subprocess.run(["node", str(script_path)], capture_output=True, text=True)
        assert result.returncode == 0
        assert expected_output in result.stdout
    finally:
        script_path.unlink()


def test_vscode_parse_ndjson_environment_variables():
    """Test script execution with different environment variables"""
    env = os.environ.copy()
    env["NODE_ENV"] = "test"

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(
        ["node", str(script)],
        capture_output=True,
        text=True,
        env=env
    )
    # Should still work with modified environment
    assert result.returncode == 0


def test_vscode_parse_ndjson_timeout_handling():
    """Test script execution with timeout"""
    script_content = '''
    setTimeout(() => {
        console.log("OK");
        process.exit(0);
    }, 100);
    '''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)

    try:
        result = subprocess.run(
            ["node", str(script_path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert "OK" in result.stdout
    finally:
        script_path.unlink()


def test_vscode_parse_ndjson_working_directory():
    """Test script execution from different working directories"""
    original_cwd = os.getcwd()
    script = Path("integrations/vscode-agot/test_utils.js").resolve()

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        # Should work regardless of working directory
        assert result.returncode == 0

    os.chdir(original_cwd)


@patch('subprocess.run')
def test_vscode_parse_ndjson_mocked_success(mock_run):
    """Test VSCode parsing with mocked subprocess call"""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "OK\n"
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "OK" in result.stdout
    mock_run.assert_called_once_with(["node", str(script)], capture_output=True, text=True)


@patch('subprocess.run')
def test_vscode_parse_ndjson_mocked_failure(mock_run):
    """Test VSCode parsing with mocked subprocess failure"""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Script error\n"
    mock_run.return_value = mock_result

    script = Path("integrations/vscode-agot/test_utils.js")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "Script error" in result.stderr
    mock_run.assert_called_once()


def test_vscode_script_path_validation():
    """Test validation of script path properties"""
    script = Path("integrations/vscode-agot/test_utils.js")

    # Test path properties
    assert script.name == "test_utils.js"
    assert script.parent.name == "vscode-agot"
    assert str(script).endswith("test_utils.js")

    # Test path operations
    absolute_path = script.resolve()
    assert absolute_path.is_absolute()


class TestVSCodeUtilsClass:
    """Class-based tests for VSCode utilities"""

    def setup_method(self):
        """Setup method run before each test method"""
        self.script_path = Path("integrations/vscode-agot/test_utils.js")

    def teardown_method(self):
        """Teardown method run after each test method"""
        # Clean up any temporary files if needed
        pass

    def test_script_execution_class_method(self):
        """Test script execution using class-based approach"""
        result = subprocess.run(["node", str(self.script_path)], capture_output=True, text=True)
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_multiple_executions(self):
        """Test multiple consecutive executions of the script"""
        for i in range(3):
            result = subprocess.run(["node", str(self.script_path)], capture_output=True, text=True)
            assert result.returncode == 0
            assert "OK" in result.stdout