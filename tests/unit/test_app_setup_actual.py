"""Comprehensive unit tests for the actual app_setup module functions.

This module tests the real functionality present in app_setup.py including
FastAPI app creation, authentication, Neo4j connection testing, and configuration management.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import sys
import tempfile
import shutil
from pathlib import Path
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient
import yaml
import json

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "git" / "src"))

from adaptive_graph_of_thoughts.app_setup import (
    get_basic_auth, _ask_llm, _test_conn, create_app, 
    _read_settings, _write_settings
)


class TestBasicAuthentication:
    """Test suite for HTTP Basic authentication functionality."""

    def test_get_basic_auth_with_valid_credentials_succeeds(self):
        """Test that valid credentials pass authentication."""
        credentials = HTTPBasicCredentials(username="testuser", password="testpass")
        
        with patch.dict(os.environ, {"BASIC_AUTH_USER": "testuser", "BASIC_AUTH_PASS": "testpass"}):
            result = get_basic_auth(credentials)
            assert result is True

    def test_get_basic_auth_with_invalid_username_raises_401(self):
        """Test that invalid username raises HTTP 401."""
        credentials = HTTPBasicCredentials(username="wronguser", password="testpass")
        
        with patch.dict(os.environ, {"BASIC_AUTH_USER": "testuser", "BASIC_AUTH_PASS": "testpass"}):
            with pytest.raises(HTTPException) as exc_info:
                get_basic_auth(credentials)
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Unauthorized"

    def test_get_basic_auth_with_invalid_password_raises_401(self):
        """Test that invalid password raises HTTP 401."""
        credentials = HTTPBasicCredentials(username="testuser", password="wrongpass")
        
        with patch.dict(os.environ, {"BASIC_AUTH_USER": "testuser", "BASIC_AUTH_PASS": "testpass"}):
            with pytest.raises(HTTPException) as exc_info:
                get_basic_auth(credentials)
            assert exc_info.value.status_code == 401

    def test_get_basic_auth_with_no_env_vars_allows_access(self):
        """Test that missing environment variables allow access."""
        credentials = HTTPBasicCredentials(username="anyuser", password="anypass")
        
        with patch.dict(os.environ, {}, clear=True):
            result = get_basic_auth(credentials)
            assert result is True

    def test_get_basic_auth_with_partial_env_vars_allows_access(self):
        """Test that partial environment variables allow access."""
        credentials = HTTPBasicCredentials(username="testuser", password="testpass")
        
        # Only username set, no password
        with patch.dict(os.environ, {"BASIC_AUTH_USER": "testuser"}, clear=True):
            result = get_basic_auth(credentials)
            assert result is True

    @pytest.mark.parametrize("username,password,env_user,env_pass,expected", [
        ("admin", "secret", "admin", "secret", True),
        ("admin", "wrong", "admin", "secret", False),
        ("wrong", "secret", "admin", "secret", False),
        ("", "", "admin", "secret", False),
        ("admin", "", "admin", "secret", False),
        ("", "secret", "admin", "secret", False),
    ])
    def test_get_basic_auth_parametrized_scenarios(self, username, password, env_user, env_pass, expected):
        """Test various authentication scenarios with parameterized inputs."""
        credentials = HTTPBasicCredentials(username=username, password=password)
        
        with patch.dict(os.environ, {"BASIC_AUTH_USER": env_user, "BASIC_AUTH_PASS": env_pass}):
            if expected:
                result = get_basic_auth(credentials)
                assert result is True
            else:
                with pytest.raises(HTTPException):
                    get_basic_auth(credentials)

    def test_get_basic_auth_with_special_characters_in_credentials(self):
        """Test authentication with special characters in credentials."""
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        credentials = HTTPBasicCredentials(username=f"user{special_chars}", password=f"pass{special_chars}")
        
        with patch.dict(os.environ, {
            "BASIC_AUTH_USER": f"user{special_chars}", 
            "BASIC_AUTH_PASS": f"pass{special_chars}"
        }):
            result = get_basic_auth(credentials)
            assert result is True

    def test_get_basic_auth_with_unicode_credentials(self):
        """Test authentication with unicode characters in credentials."""
        credentials = HTTPBasicCredentials(username="üser_tëst", password="pässwörd_🔐")
        
        with patch.dict(os.environ, {"BASIC_AUTH_USER": "üser_tëst", "BASIC_AUTH_PASS": "pässwörd_🔐"}):
            result = get_basic_auth(credentials)
            assert result is True


class TestLLMForwarding:
    """Test suite for LLM forwarding functionality."""

    @patch("adaptive_graph_of_thoughts.app_setup.ask_llm")
    def test_ask_llm_forwards_prompt_correctly(self, mock_ask_llm):
        """Test that _ask_llm correctly forwards prompts to the LLM service."""
        mock_ask_llm.return_value = "Test response"
        
        result = _ask_llm("Test prompt")
        
        mock_ask_llm.assert_called_once_with("Test prompt")
        assert result == "Test response"

    @patch("adaptive_graph_of_thoughts.app_setup.ask_llm")
    def test_ask_llm_with_empty_prompt(self, mock_ask_llm):
        """Test _ask_llm with empty prompt."""
        mock_ask_llm.return_value = "Empty response"
        
        result = _ask_llm("")
        
        mock_ask_llm.assert_called_once_with("")
        assert result == "Empty response"

    @patch("adaptive_graph_of_thoughts.app_setup.ask_llm")
    def test_ask_llm_with_large_prompt(self, mock_ask_llm):
        """Test _ask_llm with very large prompt."""
        large_prompt = "x" * 100000  # 100KB prompt
        mock_ask_llm.return_value = "Large response"
        
        result = _ask_llm(large_prompt)
        
        mock_ask_llm.assert_called_once_with(large_prompt)
        assert result == "Large response"

    @patch("adaptive_graph_of_thoughts.app_setup.ask_llm")
    def test_ask_llm_with_unicode_prompt(self, mock_ask_llm):
        """Test _ask_llm with unicode characters in prompt."""
        unicode_prompt = "Explain quantum computing in Français: 量子计算 🚀"
        mock_ask_llm.return_value = "Unicode response: 回答"
        
        result = _ask_llm(unicode_prompt)
        
        mock_ask_llm.assert_called_once_with(unicode_prompt)
        assert result == "Unicode response: 回答"

    @patch("adaptive_graph_of_thoughts.app_setup.ask_llm")
    def test_ask_llm_exception_propagation(self, mock_ask_llm):
        """Test that exceptions from ask_llm are properly propagated."""
        mock_ask_llm.side_effect = ConnectionError("LLM service unavailable")
        
        with pytest.raises(ConnectionError, match="LLM service unavailable"):
            _ask_llm("Test prompt")


class TestNeo4jConnectionTesting:
    """Test suite for Neo4j connection testing functionality."""

    @patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver")
    def test_test_conn_with_valid_connection_returns_true(self, mock_driver_class):
        """Test successful Neo4j connection."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver_class.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        result = _test_conn("bolt://localhost:7687", "neo4j", "password", "neo4j")
        
        assert result is True
        mock_driver.close.assert_called_once()
        mock_session.run.assert_called_once_with("RETURN 1")

    @patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver")
    def test_test_conn_with_service_unavailable_returns_false(self, mock_driver_class):
        """Test Neo4j connection when service is unavailable."""
        from neo4j.exceptions import ServiceUnavailable
        mock_driver_class.side_effect = ServiceUnavailable("Service unavailable")
        
        result = _test_conn("bolt://localhost:7687", "neo4j", "password", "neo4j")
        
        assert result is False

    @patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver")
    def test_test_conn_with_auth_error_returns_false(self, mock_driver_class):
        """Test Neo4j connection with authentication error."""
        from neo4j.exceptions import AuthError
        mock_driver_class.side_effect = AuthError("Authentication failed")
        
        result = _test_conn("bolt://localhost:7687", "neo4j", "wrongpass", "neo4j")
        
        assert result is False

    @patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver")
    def test_test_conn_with_neo4j_error_returns_false(self, mock_driver_class):
        """Test Neo4j connection with general Neo4j error."""
        from neo4j.exceptions import Neo4jError
        mock_driver_class.side_effect = Neo4jError("General Neo4j error")
        
        result = _test_conn("bolt://localhost:7687", "neo4j", "password", "neo4j")
        
        assert result is False

    @patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver")
    def test_test_conn_with_unexpected_error_returns_false(self, mock_driver_class):
        """Test Neo4j connection with unexpected error."""
        mock_driver_class.side_effect = Exception("Unexpected error")
        
        result = _test_conn("bolt://localhost:7687", "neo4j", "password", "neo4j")
        
        assert result is False

    @patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver")
    def test_test_conn_with_session_error_returns_false(self, mock_driver_class):
        """Test Neo4j connection when session operation fails."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver_class.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Session error")
        
        result = _test_conn("bolt://localhost:7687", "neo4j", "password", "neo4j")
        
        assert result is False
        mock_driver.close.assert_called_once()

    @pytest.mark.parametrize("uri,user,password,database", [
        ("bolt://localhost:7687", "neo4j", "password", "neo4j"),
        ("neo4j://remote.db:7687", "admin", "secret123", "mydb"),
        ("bolt+s://secure.db:7687", "user", "p@ssw0rd!", "production"),
        ("neo4j+ssc://cluster.db:7687", "readonly", "view_only", "analytics"),
    ])
    @patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver")
    def test_test_conn_with_various_connection_strings(self, mock_driver_class, uri, user, password, database):
        """Test Neo4j connection with various connection string formats."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver_class.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        result = _test_conn(uri, user, password, database)
        
        assert result is True
        mock_driver_class.assert_called_once_with(uri, auth=(user, password))
        mock_driver.session.assert_called_once_with(database=database)


class TestYAMLConfigurationManagement:
    """Test suite for YAML configuration reading and writing."""

    def test_read_settings_from_valid_yaml_file(self):
        """Test reading settings from a valid YAML file."""
        test_data = {
            "app": {
                "name": "Test App",
                "version": "1.0.0",
                "debug": True,
                "log_level": "INFO"
            }
        }
        
        with patch("adaptive_graph_of_thoughts.app_setup.yaml_path") as mock_path:
            mock_path.read_text.return_value = yaml.safe_dump(test_data)
            with patch("builtins.open", mock_open(read_data=yaml.safe_dump(test_data))):
                result = _read_settings()
        
        expected = {"name": "Test App", "version": "1.0.0", "debug": True, "log_level": "INFO"}
        assert result == expected

    def test_read_settings_from_empty_yaml_file(self):
        """Test reading settings from an empty YAML file."""
        with patch("builtins.open", mock_open(read_data="")):
            result = _read_settings()
        
        assert result == {}

    def test_read_settings_from_yaml_without_app_section(self):
        """Test reading settings from YAML file without app section."""
        test_data = {"database": {"url": "sqlite:///test.db"}}
        
        with patch("builtins.open", mock_open(read_data=yaml.safe_dump(test_data))):
            result = _read_settings()
        
        assert result == {}

    def test_write_settings_to_yaml_file(self):
        """Test writing settings to YAML file."""
        mock_file = mock_open()
        test_data = {"name": "Updated App", "version": "2.0.0"}
        existing_data = {"app": {"old_key": "old_value"}, "other": {"key": "value"}}
        
        with patch("builtins.open", mock_file):
            with patch("adaptive_graph_of_thoughts.app_setup.yaml.safe_load", return_value=existing_data):
                with patch("adaptive_graph_of_thoughts.app_setup.fcntl.flock"):
                    _write_settings(test_data)
        
        # Verify file operations
        mock_file.assert_called_once()
        handle = mock_file()
        handle.seek.assert_called()
        handle.truncate.assert_called_once()

    def test_write_settings_creates_app_section_if_missing(self):
        """Test that write_settings creates app section if it doesn't exist."""
        mock_file = mock_open()
        test_data = {"name": "New App"}
        existing_data = {"other_section": {"key": "value"}}
        
        with patch("builtins.open", mock_file):
            with patch("adaptive_graph_of_thoughts.app_setup.yaml.safe_load", return_value=existing_data):
                with patch("adaptive_graph_of_thoughts.app_setup.fcntl.flock"):
                    with patch("adaptive_graph_of_thoughts.app_setup.yaml.safe_dump") as mock_dump:
                        _write_settings(test_data)
        
        # Verify that app section was added
        mock_dump.assert_called_once()
        written_data = mock_dump.call_args[0][0]
        assert "app" in written_data
        assert written_data["app"]["name"] == "New App"

    def test_write_settings_with_file_lock_error(self):
        """Test write_settings behavior when file locking fails."""
        test_data = {"name": "Test App"}
        
        with patch("builtins.open", side_effect=IOError("File lock failed")):
            with pytest.raises(IOError):
                _write_settings(test_data)

    def test_read_settings_with_malformed_yaml(self):
        """Test reading settings from malformed YAML file."""
        malformed_yaml = "invalid: yaml: content: ["
        
        with patch("builtins.open", mock_open(read_data=malformed_yaml)):
            with pytest.raises(yaml.YAMLError):
                _read_settings()

    def test_write_settings_preserves_existing_data(self):
        """Test that write_settings preserves existing non-app data."""
        mock_file = mock_open()
        test_data = {"name": "Updated App"}
        existing_data = {
            "app": {"old_name": "Old App", "version": "1.0.0"},
            "database": {"url": "sqlite:///test.db"},
            "logging": {"level": "DEBUG"}
        }
        
        captured_data = None
        def capture_dump(data, file_handle):
            nonlocal captured_data
            captured_data = data
        
        with patch("builtins.open", mock_file):
            with patch("adaptive_graph_of_thoughts.app_setup.yaml.safe_load", return_value=existing_data):
                with patch("adaptive_graph_of_thoughts.app_setup.fcntl.flock"):
                    with patch("adaptive_graph_of_thoughts.app_setup.yaml.safe_dump", side_effect=capture_dump):
                        _write_settings(test_data)
        
        # Verify existing sections are preserved
        assert captured_data["database"]["url"] == "sqlite:///test.db"
        assert captured_data["logging"]["level"] == "DEBUG"
        assert captured_data["app"]["name"] == "Updated App"
        assert captured_data["app"]["version"] == "1.0.0"  # Should be preserved


class TestFastAPIAppCreation:
    """Test suite for FastAPI application creation and configuration."""

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_create_app_returns_fastapi_instance(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test that create_app returns a properly configured FastAPI instance."""
        from fastapi import FastAPI
        
        # Mock settings
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "*"
        
        app = create_app()
        
        assert isinstance(app, FastAPI)
        assert app.title == "Test App"
        assert app.version == "1.0.0"

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_create_app_configures_cors_with_wildcard(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test CORS configuration with wildcard origin."""
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "*"
        
        app = create_app()
        
        # Verify CORS middleware is added
        cors_middleware = None
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware.cls):
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_create_app_configures_cors_with_specific_origins(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test CORS configuration with specific origins."""
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "http://localhost:3000,https://example.com"
        
        app = create_app()
        
        # The app should be created successfully with specific origins
        assert isinstance(app, type(create_app()))

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_create_app_attaches_got_processor_to_state(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test that GoTProcessor is attached to app state."""
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "*"
        
        mock_processor_instance = Mock()
        mock_got_processor.return_value = mock_processor_instance
        
        app = create_app()
        
        assert hasattr(app.state, 'got_processor')
        assert app.state.got_processor == mock_processor_instance

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_create_app_includes_all_routers(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test that all expected routers are included in the app."""
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "*"
        
        app = create_app()
        
        # Check that routes exist (this is a basic check)
        route_paths = [route.path for route in app.routes]
        
        # Should have various endpoints
        assert any("/mcp" in path for path in route_paths)
        assert any("/admin/mcp" in path for path in route_paths)

    def test_create_app_with_test_client(self):
        """Test creating app and using it with TestClient."""
        with patch("adaptive_graph_of_thoughts.app_setup.settings") as mock_settings:
            with patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor"):
                with patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor"):
                    mock_settings.app.name = "Test App"
                    mock_settings.app.version = "1.0.0"
                    mock_settings.app.log_level = "INFO"
                    mock_settings.app.cors_allowed_origins_str = "*"
                    
                    app = create_app()
                    client = TestClient(app)
                    
                    # Test basic functionality
                    assert client is not None


class TestAppEndpointsIntegration:
    """Integration tests for app endpoints created by create_app."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_health_endpoint_without_neo4j(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test health endpoint when Neo4j is unavailable."""
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "*"
        
        with patch("adaptive_graph_of_thoughts.app_setup.runtime_settings") as mock_runtime:
            mock_runtime.neo4j.uri = "bolt://localhost:7687"
            mock_runtime.neo4j.user = "neo4j"
            mock_runtime.neo4j.password = "password"
            mock_runtime.neo4j.database = "neo4j"
            
            with patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver") as mock_driver:
                mock_driver.side_effect = Exception("Connection failed")
                
                app = create_app()
                client = TestClient(app)
                
                # Health endpoint should return 500 when Neo4j is down
                with patch.dict(os.environ, {"BASIC_AUTH_USER": "test", "BASIC_AUTH_PASS": "test"}):
                    response = client.get("/health", auth=("test", "test"))
                    assert response.status_code == 500
                    assert response.json()["status"] == "unhealthy"

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_health_endpoint_with_working_neo4j(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test health endpoint when Neo4j is working."""
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "*"
        
        with patch("adaptive_graph_of_thoughts.app_setup.runtime_settings") as mock_runtime:
            mock_runtime.neo4j.uri = "bolt://localhost:7687"
            mock_runtime.neo4j.user = "neo4j"
            mock_runtime.neo4j.password = "password"
            mock_runtime.neo4j.database = "neo4j"
            
            with patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver") as mock_driver:
                mock_driver_instance = Mock()
                mock_session = Mock()
                mock_driver.return_value = mock_driver_instance
                mock_driver_instance.session.return_value.__enter__.return_value = mock_session
                
                app = create_app()
                client = TestClient(app)
                
                # Health endpoint should return 200 when Neo4j is working
                with patch.dict(os.environ, {"BASIC_AUTH_USER": "test", "BASIC_AUTH_PASS": "test"}):
                    response = client.get("/health", auth=("test", "test"))
                    assert response.status_code == 200
                    assert response.json()["status"] == "ok"
                    assert response.json()["neo4j"] == "up"


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling scenarios."""

    def test_ask_llm_with_none_input(self):
        """Test _ask_llm with None input."""
        with patch("adaptive_graph_of_thoughts.app_setup.ask_llm") as mock_ask_llm:
            mock_ask_llm.return_value = "None response"
            
            result = _ask_llm(None)
            
            mock_ask_llm.assert_called_once_with(None)
            assert result == "None response"

    def test_test_conn_with_none_parameters(self):
        """Test _test_conn with None parameters."""
        result = _test_conn(None, None, None, None)
        assert result is False

    def test_test_conn_with_empty_string_parameters(self):
        """Test _test_conn with empty string parameters."""
        result = _test_conn("", "", "", "")
        assert result is False

    def test_get_basic_auth_with_none_credentials(self):
        """Test get_basic_auth with None credentials."""
        with pytest.raises(AttributeError):
            get_basic_auth(None)

    def test_read_settings_with_permission_error(self):
        """Test _read_settings when file permission is denied."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                _read_settings()

    def test_write_settings_with_readonly_file(self):
        """Test _write_settings when file is read-only."""
        test_data = {"name": "Test"}
        
        with patch("builtins.open", side_effect=PermissionError("Read-only file")):
            with pytest.raises(PermissionError):
                _write_settings(test_data)

    @patch("adaptive_graph_of_thoughts.app_setup.settings")
    @patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor")
    @patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor")
    def test_create_app_with_invalid_cors_config(self, mock_resource_monitor, mock_got_processor, mock_settings):
        """Test create_app with invalid CORS configuration."""
        mock_settings.app.name = "Test App"
        mock_settings.app.version = "1.0.0"
        mock_settings.app.log_level = "INFO"
        mock_settings.app.cors_allowed_origins_str = "   ,  ,   "  # Only whitespace and commas
        
        app = create_app()
        
        # Should handle invalid CORS config gracefully
        assert isinstance(app, type(create_app()))


class TestConcurrencyAndThreadSafety:
    """Test suite for concurrency and thread safety."""

    def test_multiple_create_app_calls_thread_safety(self):
        """Test that multiple create_app calls are thread-safe."""
        import threading
        import time
        
        apps = []
        errors = []
        
        def create_app_thread():
            try:
                with patch("adaptive_graph_of_thoughts.app_setup.settings") as mock_settings:
                    with patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor"):
                        with patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor"):
                            mock_settings.app.name = "Test App"
                            mock_settings.app.version = "1.0.0"
                            mock_settings.app.log_level = "INFO"
                            mock_settings.app.cors_allowed_origins_str = "*"
                            
                            app = create_app()
                            apps.append(app)
                            time.sleep(0.01)  # Small delay to increase chances of race conditions
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=create_app_thread) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(apps) == 5

    def test_concurrent_yaml_read_write_operations(self):
        """Test concurrent YAML read/write operations."""
        import threading
        
        results = []
        errors = []
        
        def read_settings():
            try:
                with patch("builtins.open", mock_open(read_data="app:\n  name: Test")):
                    result = _read_settings()
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        def write_settings():
            try:
                with patch("builtins.open", mock_open()):
                    with patch("adaptive_graph_of_thoughts.app_setup.yaml.safe_load", return_value={}):
                        with patch("adaptive_graph_of_thoughts.app_setup.fcntl.flock"):
                            _write_settings({"name": "Updated"})
            except Exception as e:
                errors.append(e)
        
        # Mix of read and write operations
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=read_settings))
            threads.append(threading.Thread(target=write_settings))
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"


# Additional helper functions and fixtures for comprehensive testing
@pytest.fixture
def mock_neo4j_driver():
    """Provide a mock Neo4j driver for testing."""
    with patch("adaptive_graph_of_thoughts.app_setup.GraphDatabase.driver") as mock_driver:
        mock_instance = Mock()
        mock_session = Mock()
        mock_driver.return_value = mock_instance
        mock_instance.session.return_value.__enter__.return_value = mock_session
        yield mock_driver

@pytest.fixture
def app_config():
    """Provide standard app configuration for testing."""
    return {
        "name": "Test Application",
        "version": "1.0.0",
        "debug": True,
        "log_level": "INFO",
        "cors_allowed_origins_str": "*"
    }

@pytest.fixture
def authenticated_client():
    """Provide an authenticated test client."""
    with patch("adaptive_graph_of_thoughts.app_setup.settings") as mock_settings:
        with patch("adaptive_graph_of_thoughts.app_setup.GoTProcessor"):
            with patch("adaptive_graph_of_thoughts.app_setup.ResourceMonitor"):
                mock_settings.app.name = "Test App"
                mock_settings.app.version = "1.0.0" 
                mock_settings.app.log_level = "INFO"
                mock_settings.app.cors_allowed_origins_str = "*"
                
                app = create_app()
                client = TestClient(app)
                
                with patch.dict(os.environ, {"BASIC_AUTH_USER": "test", "BASIC_AUTH_PASS": "test"}):
                    yield client

def assert_yaml_file_structure(file_content, expected_sections):
    """Helper function to assert YAML file structure."""
    data = yaml.safe_load(file_content)
    for section in expected_sections:
        assert section in data, f"Expected section '{section}' not found in YAML"

def create_temp_yaml_file(content_dict, temp_dir):
    """Helper function to create temporary YAML files for testing."""
    file_path = os.path.join(temp_dir, "test_config.yaml")
    with open(file_path, "w") as f:
        yaml.safe_dump(content_dict, f)
    return file_path