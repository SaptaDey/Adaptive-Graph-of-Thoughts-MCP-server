"""Comprehensive unit tests for the app_setup module.

This module tests all functionality related to application setup, configuration,
initialization, and teardown processes.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import os
import sys
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Optional

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "git" / "src"))

from adaptive_graph_of_thoughts.app_setup import *

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration object for testing."""
    return {
        "database_url": "sqlite:///test.db",
        "debug": True,
        "log_level": "INFO",
        "api_key": "test_api_key",
        "timeout": 30,
        "max_retries": 3
    }

@pytest.fixture
def invalid_config() -> Dict[str, Any]:
    """Provide an invalid configuration for error testing."""
    return {
        "database_url": "",
        "debug": "not_a_boolean",
        "log_level": "INVALID_LEVEL",
        "timeout": -1,
        "max_retries": "not_a_number"
    }

@pytest.fixture
def temp_directory() -> str:
    """Create a temporary directory for file-based tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing logging functionality."""
    with patch("adaptive_graph_of_thoughts.app_setup.logger") as mock_log:
        yield mock_log

class TestAppSetup:
    """Test suite for core application setup functionality."""

    def test_initialize_app_with_valid_config_succeeds(self, mock_config, mock_logger):
        initialize_app(mock_config)
        mock_logger.info.assert_called()

    def test_initialize_app_with_invalid_config_raises_error(self, invalid_config, mock_logger):
        with pytest.raises(ValueError, match="Invalid configuration"):
            initialize_app(invalid_config)

    def test_initialize_app_with_missing_config_uses_defaults(self, mock_logger):
        initialize_app({})
        mock_logger.info.assert_called()

    @pytest.mark.parametrize("config_key,invalid_value,expected_error", [
        ("database_url", None, "Database URL cannot be None"),
        ("database_url", "", "Database URL cannot be empty"),
        ("timeout", -1, "Timeout must be positive"),
        ("timeout", "invalid", "Timeout must be a number"),
        ("max_retries", -1, "Max retries must be non-negative"),
        ("log_level", "INVALID", "Invalid log level"),
    ])
    def test_validate_config_with_invalid_values(self, mock_config, config_key, invalid_value, expected_error):
        config = mock_config.copy()
        config[config_key] = invalid_value
        with pytest.raises(ValueError, match=expected_error):
            validate_config(config)

class TestDatabaseSetup:
    """Test suite for database setup and connection management."""

    @patch("adaptive_graph_of_thoughts.app_setup.create_engine")
    def test_setup_database_with_valid_url_creates_engine(self, mock_create_engine, mock_config):
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        engine = setup_database(mock_config["database_url"], mock_config["max_retries"])
        assert engine is mock_engine

    @patch("adaptive_graph_of_thoughts.app_setup.create_engine")
    def test_setup_database_with_invalid_url_raises_error(self, mock_create_engine):
        mock_create_engine.side_effect = Exception("Invalid database URL")
        with pytest.raises(Exception, match="Invalid database URL"):
            setup_database("invalid_url", 1)

    def test_database_connection_retry_logic(self, mock_config):
        with patch("adaptive_graph_of_thoughts.app_setup.create_engine") as mock_create:
            mock_create.side_effect = [Exception("Connection failed"), Exception("Connection failed"), Mock()]
            engine = setup_database(mock_config["database_url"], 3)
            assert engine is not None

    def test_database_connection_max_retries_exceeded(self, mock_config):
        with patch("adaptive_graph_of_thoughts.app_setup.create_engine") as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            with pytest.raises(Exception, match="Max retries exceeded"):
                setup_database(mock_config["database_url"], mock_config["max_retries"])

class TestLoggingSetup:
    """Test suite for logging configuration and setup."""

    @patch("adaptive_graph_of_thoughts.app_setup.logging.getLogger")
    def test_setup_logging_with_valid_config(self, mock_get_logger, mock_config):
        mock_logger_instance = Mock()
        mock_get_logger.return_value = mock_logger_instance
        setup_logging(mock_config)
        mock_get_logger.assert_called_with("app")

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_setup_logging_with_different_levels(self, log_level, mock_config):
        config = mock_config.copy()
        config["log_level"] = log_level
        with patch("adaptive_graph_of_thoughts.app_setup.logging") as mock_logging:
            setup_logging(config)
            mock_logging.getLogger.return_value.setLevel.assert_called_with(log_level)

    def test_setup_logging_with_file_handler(self, mock_config, temp_directory):
        log_file = os.path.join(temp_directory, "test.log")
        config = mock_config.copy()
        config["log_file"] = log_file
        setup_logging(config)
        assert os.path.exists(log_file)

    def test_setup_logging_with_invalid_log_file_path_falls_back_to_console(self, mock_config):
        config = mock_config.copy()
        config["log_file"] = "/invalid/path/test.log"
        with patch("adaptive_graph_of_thoughts.app_setup.logging") as mock_logging:
            setup_logging(config)
            mock_logging.getLogger.return_value.addHandler.assert_called()

class TestConfigurationManagement:
    """Test suite for configuration management and environment handling."""

    def test_load_config_from_environment_variables(self):
        env_vars = {
            "APP_DATABASE_URL": "postgresql://test:test@localhost/testdb",
            "APP_DEBUG": "true",
            "APP_LOG_LEVEL": "DEBUG",
            "APP_TIMEOUT": "60"
        }
        with patch.dict(os.environ, env_vars):
            config = load_config()
            assert config["database_url"] == env_vars["APP_DATABASE_URL"]
            assert config["debug"] is True
            assert config["log_level"] == "DEBUG"
            assert config["timeout"] == 60

    def test_load_config_from_file(self, temp_directory):
        config_file = os.path.join(temp_directory, "config.json")
        data = {"database_url": "sqlite:///file.db", "debug": False, "log_level": "WARNING"}
        with open(config_file, "w") as f:
            import json
            json.dump(data, f)
        config = load_config(config_file)
        assert config["database_url"] == data["database_url"]
        assert config["log_level"] == "WARNING"

    def test_config_precedence_environment_over_file(self, temp_directory):
        config_file = os.path.join(temp_directory, "config.json")
        file_data = {"debug": False, "timeout": 30}
        with open(config_file, "w") as f:
            import json
            json.dump(file_data, f)
        env_vars = {"APP_DEBUG": "true", "APP_TIMEOUT": "60"}
        with patch.dict(os.environ, env_vars):
            config = load_config(config_file)
            assert config["debug"] is True
            assert config["timeout"] == 60

    @pytest.mark.parametrize("missing_key", ["database_url", "log_level", "timeout"])
    def test_config_with_missing_required_keys_uses_defaults(self, missing_key, mock_config):
        config = mock_config.copy()
        del config[missing_key]
        result = load_config(config)
        assert missing_key in result

class TestApplicationLifecycle:
    """Test suite for application lifecycle management."""

    def test_app_startup_sequence(self, mock_config):
        with patch.multiple(
            "adaptive_graph_of_thoughts.app_setup",
            setup_database=Mock(),
            setup_logging=Mock(),
            setup_monitoring=Mock()
        ) as mocks:
            initialize_app(mock_config)
            mocks["setup_database"].assert_called_once()
            mocks["setup_logging"].assert_called_once()
            mocks["setup_monitoring"].assert_called_once()

    def test_app_shutdown_sequence(self, mock_config):
        with patch.multiple(
            "adaptive_graph_of_thoughts.app_setup",
            cleanup_database=Mock(),
            cleanup_logging=Mock(),
            cleanup_monitoring=Mock()
        ) as mocks:
            shutdown_app()
            mocks["cleanup_database"].assert_called_once()
            mocks["cleanup_logging"].assert_called_once()
            mocks["cleanup_monitoring"].assert_called_once()

    def test_graceful_shutdown_on_signal(self, mock_config):
        with patch("signal.signal") as mock_signal:
            initialize_app(mock_config)
            mock_signal.assert_called()

    def test_cleanup_on_unexpected_error(self, mock_config):
        with patch("adaptive_graph_of_thoughts.app_setup.setup_database") as mock_setup:
            mock_setup.side_effect = Exception("Unexpected error")
            with pytest.raises(Exception):
                initialize_app(mock_config)
            assert_cleanup_completed(mock_logger())

    def test_resource_cleanup_idempotent(self, mock_config):
        shutdown_app()
        shutdown_app()

class TestConcurrencyAndPerformance:
    """Test suite for concurrency and performance aspects."""

    def test_concurrent_initialization_thread_safety(self, mock_config):
        import threading
        results, errors = [], []
        def target():
            try:
                initialize_app(mock_config)
                results.append(True)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=target) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert len(results) == 10

    def test_initialization_performance_within_limits(self, mock_config):
        import time
        start = time.time()
        initialize_app(mock_config)
        assert time.time() - start < 5.0

    def test_memory_usage_during_initialization(self, mock_config):
        import psutil, os
        proc = psutil.Process(os.getpid())
        before = proc.memory_info().rss
        initialize_app(mock_config)
        after = proc.memory_info().rss
        assert after - before < 100 * 1024 * 1024

class TestComponentIntegration:
    """Test suite for integration between app_setup components."""

    def test_database_and_logging_integration(self, mock_config, mock_logger):
        with patch("adaptive_graph_of_thoughts.app_setup.create_engine") as mock_create:
            mock_create.return_value = Mock()
            initialize_app(mock_config)
            mock_logger.info.assert_any_call("Database engine created")

    def test_config_validation_and_error_logging_integration(self, invalid_config, mock_logger):
        with pytest.raises(ValueError):
            initialize_app(invalid_config)
        mock_logger.error.assert_called()

    def test_monitoring_and_database_health_checks_integration(self, mock_config):
        with patch.multiple(
            "adaptive_graph_of_thoughts.app_setup",
            setup_database=Mock(),
            setup_monitoring=Mock()
        ) as mocks:
            initialize_app(mock_config)
            mocks["setup_monitoring"].assert_called()

    def test_configuration_change_propagation(self, mock_config):
        new_config = mock_config.copy()
        new_config["debug"] = False
        update_config(new_config)
        assert get_current_config()["debug"] is False

class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and comprehensive error handling."""

    def test_setup_with_minimal_config(self):
        initialize_app({})

    def test_setup_with_malformed_config_file(self, temp_directory):
        config_file = os.path.join(temp_directory, "malformed.json")
        with open(config_file, "w") as f:
            f.write('{"invalid": json}')
        with pytest.raises(ValueError, match="Malformed configuration"):
            load_config(config_file)

    def test_setup_with_insufficient_permissions(self, mock_config, temp_directory):
        restricted = os.path.join(temp_directory, "restricted")
        os.makedirs(restricted, mode=0o000)
        try:
            mock_config["log_file"] = os.path.join(restricted, "app.log")
            initialize_app(mock_config)
        finally:
            os.chmod(restricted, 0o755)

    def test_setup_with_network_unavailable(self, mock_config):
        mock_config["database_url"] = "postgresql://unreachable:5432/db"
        with patch("adaptive_graph_of_thoughts.app_setup.create_engine") as mock_create:
            mock_create.side_effect = ConnectionError("Network unreachable")
            initialize_app(mock_config)

    def test_cleanup_with_partial_initialization(self, mock_config):
        initialize_app(mock_config)
        shutdown_app()

    def test_repeated_initialization_calls(self, mock_config):
        initialize_app(mock_config)
        initialize_app(mock_config)

    def test_initialization_with_corrupted_state(self, mock_config):
        corrupt_state()
        initialize_app(mock_config)

# Additional utility functions for test helpers
def create_test_database_url(temp_directory: str) -> str:
    """Create a test database URL pointing to a temporary database."""
    db_path = os.path.join(temp_directory, "test.db")
    return f"sqlite:///{db_path}"

def assert_cleanup_completed(mock_logger):
    """Assert that cleanup operations completed successfully."""
    cleanup_calls = [
        c for c in mock_logger.info.call_args_list
        if "cleanup" in str(c).lower()
    ]
    assert cleanup_calls

def assert_initialization_order(mock_calls, expected_order):
    """Assert that initialization steps occurred in the expected order."""
    actual_order = [c[0] for c in mock_calls]
    assert actual_order == expected_order