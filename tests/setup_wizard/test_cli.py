import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
from typer.testing import CliRunner

import agt_setup

@pytest.fixture
def mock_successful_connection(monkeypatch):
    """Fixture to mock successful connection test"""
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)

@pytest.fixture
def mock_failed_connection(monkeypatch):
    """Fixture to mock failed connection test"""
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: False)

@pytest.fixture
def setup_temp_dir(tmp_path, monkeypatch):
    """Fixture to setup temporary directory and change to it"""
    monkeypatch.chdir(tmp_path)
    return tmp_path

def test_cli_creates_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)

    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://local:7687\nneo4j\npass\nneo4j\n"
    )

    assert result.exit_code == 0
    content = env_file.read_text()
    assert "NEO4J_URI='bolt://local:7687'" in content
    assert env_file.stat().st_mode & 0o777 == 0o600

def test_cli_fails_on_bad_connection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_a, **_k: False)
    runner = CliRunner()
    result = runner.invoke(agt_setup.app, input="x\nx\nx\nx\n")
    assert result.exit_code != 0
    assert not (tmp_path / ".env").exists()

def test_cli_handles_existing_env_file(tmp_path, monkeypatch):
    """Test that CLI handles existing .env file appropriately"""
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_VAR=value\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nneo4j\npassword\nneo4j\n"
    )
    
    assert result.exit_code == 0
    content = env_file.read_text()
    assert "NEO4J_URI='bolt://localhost:7687'" in content

def test_cli_with_invalid_uri_format(tmp_path, monkeypatch):
    """Test CLI behavior with malformed URI input"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="invalid-uri\nneo4j\npassword\nneo4j\n"
    )
    
    # Should still proceed if connection test passes
    assert result.exit_code == 0 or "invalid" in result.output.lower()

def test_cli_empty_input_handling(tmp_path, monkeypatch):
    """Test CLI behavior with empty inputs"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="\n\n\n\n"
    )
    
    # Should handle empty inputs gracefully or prompt for required fields
    output = result.output.lower()
    assert result.exit_code == 0 or "required" in output or "empty" in output

def test_cli_permission_denied_env_file(tmp_path, monkeypatch):
    """Test CLI behavior when unable to write .env file due to permissions"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    # Create a directory where .env should be to simulate permission issue
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nneo4j\npassword\nneo4j\n"
    )
    
    # Should fail gracefully when can't write file
    assert result.exit_code != 0

def test_cli_connection_test_with_timeout(tmp_path, monkeypatch):
    """Test CLI behavior when connection test times out"""
    monkeypatch.chdir(tmp_path)
    
    def mock_test_connection(*args, **kwargs):
        raise TimeoutError("Connection timeout")
    
    monkeypatch.setattr(agt_setup, "_test_connection", mock_test_connection)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nneo4j\npassword\nneo4j\n"
    )
    
    assert result.exit_code != 0
    assert not (tmp_path / ".env").exists()

def test_cli_connection_test_with_exception(tmp_path, monkeypatch):
    """Test CLI behavior when connection test raises unexpected exception"""
    monkeypatch.chdir(tmp_path)
    
    def mock_test_connection(*args, **kwargs):
        raise Exception("Unexpected error")
    
    monkeypatch.setattr(agt_setup, "_test_connection", mock_test_connection)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nneo4j\npassword\nneo4j\n"
    )
    
    assert result.exit_code != 0
    assert not (tmp_path / ".env").exists()

def test_cli_special_characters_in_credentials(tmp_path, monkeypatch):
    """Test CLI with special characters in username/password"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nuser@domain\npa$$w0rd!\nneo4j\n"
    )
    
    if result.exit_code == 0:
        env_file = tmp_path / ".env"
        content = env_file.read_text()
        assert "user@domain" in content
        assert "pa$$w0rd!" in content

def test_cli_multiple_runs_same_directory(tmp_path, monkeypatch):
    """Test running CLI multiple times in same directory"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    
    # First run
    result1 = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nneo4j\npassword1\nneo4j\n"
    )
    assert result1.exit_code == 0
    
    # Second run - should handle existing file
    result2 = runner.invoke(
        agt_setup.app, input="bolt://localhost:7688\nneo4j\npassword2\nneo4j\n"
    )
    assert result2.exit_code == 0
    
    env_file = tmp_path / ".env"
    content = env_file.read_text()
    # Should contain the latest configuration
    assert "7688" in content

def test_env_file_permissions_after_creation(tmp_path, monkeypatch):
    """Test that .env file has correct permissions after creation"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nneo4j\npassword\nneo4j\n"
    )
    
    assert result.exit_code == 0
    env_file = tmp_path / ".env"
    assert env_file.exists()
    
    # Check file permissions are restrictive (600)
    file_mode = env_file.stat().st_mode & 0o777
    assert file_mode == 0o600

@pytest.mark.parametrize("uri,expected_in_file", [
    ("bolt://localhost:7687", "bolt://localhost:7687"),
    ("neo4j://remote:7687", "neo4j://remote:7687"),
    ("bolt+s://secure:7687", "bolt+s://secure:7687"),
    ("neo4j+s://secure-remote:7687", "neo4j+s://secure-remote:7687"),
])
def test_cli_various_uri_schemes(tmp_path, monkeypatch, uri, expected_in_file):
    """Test CLI with various valid URI schemes"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input=f"{uri}\nneo4j\npassword\nneo4j\n"
    )
    
    assert result.exit_code == 0
    env_file = tmp_path / ".env"
    content = env_file.read_text()
    assert expected_in_file in content

@pytest.mark.parametrize("username,password,database", [
    ("simple_user", "simple_pass", "simple_db"),
    ("user.with.dots", "pass-with-dashes", "db_with_underscores"),
    ("user123", "pass456", "db789"),
    ("UPPERCASE_USER", "MixedCasePass", "lowercase_db"),
])
def test_cli_various_credential_formats(tmp_path, monkeypatch, username, password, database):
    """Test CLI with various credential formats"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input=f"bolt://localhost:7687\n{username}\n{password}\n{database}\n"
    )
    
    assert result.exit_code == 0
    env_file = tmp_path / ".env"
    content = env_file.read_text()
    assert username in content
    assert password in content
    assert database in content

def test_env_file_complete_structure(tmp_path, monkeypatch):
    """Test that .env file contains all expected variables with correct format"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    test_uri = "bolt://testhost:7687"
    test_username = "testuser"
    test_password = "testpass"
    test_database = "testdb"
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input=f"{test_uri}\n{test_username}\n{test_password}\n{test_database}\n"
    )
    
    assert result.exit_code == 0
    env_file = tmp_path / ".env"
    assert env_file.exists()
    
    content = env_file.read_text()
    
    # Verify all expected variables are present
    assert f"NEO4J_URI='{test_uri}'" in content
    assert f"NEO4J_USERNAME='{test_username}'" in content
    assert f"NEO4J_PASSWORD='{test_password}'" in content
    assert f"NEO4J_DATABASE='{test_database}'" in content
    
    # Verify no extra whitespace or formatting issues
    lines = content.strip().split('\n')
    for line in lines:
        if line.strip():
            assert '=' in line
            assert line.startswith('NEO4J_')

def test_cli_output_messages(tmp_path, monkeypatch):
    """Test that CLI provides appropriate user feedback messages"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
    
    runner = CliRunner()
    result = runner.invoke(
        agt_setup.app, input="bolt://localhost:7687\nneo4j\npassword\nneo4j\n"
    )
    
    assert result.exit_code == 0
    output = result.output.lower()
    assert any(keyword in output for keyword in ["neo4j", "connection", "setup", "uri", "database"])

def test_cli_help_command():
    """Test that CLI provides help information"""
    runner = CliRunner()
    result = runner.invoke(agt_setup.app, ["--help"])
    
    assert result.exit_code == 0
    output = result.output.lower()
    assert any(keyword in output for keyword in ["help", "usage", "options", "commands"])

class TestAgtSetupCLIEdgeCases:
    """Test class for edge cases and error conditions in agt_setup CLI"""
    
    def test_cli_with_unicode_characters(self, tmp_path, monkeypatch):
        """Test CLI handling of unicode characters in inputs"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
        
        runner = CliRunner()
        result = runner.invoke(
            agt_setup.app, input="bolt://localhost:7687\nusér\npássword\nneo4j\n"
        )
        
        # Should handle unicode characters appropriately
        if result.exit_code == 0:
            env_file = tmp_path / ".env"
            content = env_file.read_text()
            assert "usér" in content
    
    def test_cli_very_long_inputs(self, tmp_path, monkeypatch):
        """Test CLI with very long input strings"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
        
        long_string = "a" * 100  # Reasonable length for testing
        runner = CliRunner()
        result = runner.invoke(
            agt_setup.app, input=f"bolt://localhost:7687\n{long_string}\npassword\nneo4j\n"
        )
        
        # Should handle long inputs gracefully
        assert result.exit_code == 0 or "too long" in result.output.lower()
    
    def test_cli_whitespace_handling(self, tmp_path, monkeypatch):
        """Test CLI handling of inputs with whitespace"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(agt_setup, "_test_connection", lambda *_args, **_kw: True)
        
        runner = CliRunner()
        result = runner.invoke(
            agt_setup.app, input="  bolt://localhost:7687  \n  neo4j  \n  password  \n  neo4j  \n"
        )
        
        if result.exit_code == 0:
            env_file = tmp_path / ".env"
            content = env_file.read_text()
            # Should handle whitespace appropriately (either strip or preserve)
            assert "bolt://localhost:7687" in content