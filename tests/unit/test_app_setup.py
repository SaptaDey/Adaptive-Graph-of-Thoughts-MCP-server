import pytest
import unittest.mock as mock
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = {
        'debug': False,
        'host': 'localhost',
        'port': 8000,
        'database_url': 'sqlite:///test.db',
        'secret_key': 'test-secret-key'
    }
    return config

@pytest.fixture
def app_instance():
    """Create a test application instance."""
    with patch('app_setup.create_app') as mock_create:
        mock_app = Mock()
        mock_create.return_value = mock_app
        yield mock_app

class TestAppSetup:
    """Test suite for application setup functionality."""

    def test_create_app_with_default_config(self, mock_config):
        """Test application creation with default configuration."""
        with patch('app_setup.load_config') as mock_load:
            mock_load.return_value = mock_config

            with patch('app_setup.Flask') as mock_flask:
                mock_app = Mock()
                mock_flask.return_value = mock_app

                from app_setup import create_app
                result = create_app()

                assert result == mock_app
                mock_flask.assert_called_once()
                mock_load.assert_called_once()

    def test_create_app_with_custom_config(self):
        """Test application creation with custom configuration."""
        custom_config = {
            'debug': True,
            'host': '0.0.0.0',
            'port': 9000,
            'database_url': 'postgresql://localhost/test'
        }

        with patch('app_setup.Flask') as mock_flask:
            mock_app = Mock()
            mock_flask.return_value = mock_app

            from app_setup import create_app
            result = create_app(config=custom_config)

            assert result == mock_app
            mock_app.config.update.assert_called_with(custom_config)

    def test_create_app_with_missing_config(self):
        """Test application creation when configuration is missing."""
        with patch('app_setup.load_config') as mock_load:
            mock_load.side_effect = FileNotFoundError("Config file not found")

            with pytest.raises(FileNotFoundError):
                from app_setup import create_app
                create_app()

    def test_create_app_with_invalid_config(self):
        """Test application creation with invalid configuration."""
        invalid_config = {"invalid": "config"}

        with patch('app_setup.load_config') as mock_load:
            mock_load.return_value = invalid_config

            with patch('app_setup.validate_config') as mock_validate:
                mock_validate.side_effect = ValueError("Invalid configuration")

                with pytest.raises(ValueError):
                    from app_setup import create_app
                    create_app()

    def test_create_app_with_none_config(self):
        """Test application creation when config is None."""
        with patch('app_setup.Flask') as mock_flask:
            mock_app = Mock()
            mock_flask.return_value = mock_app

            from app_setup import create_app
            result = create_app(config=None)

            assert result == mock_app
            # Should use default configuration
            mock_app.config.update.assert_called()

    def test_load_config_from_file(self, temp_dir):
        """Test loading configuration from file."""
        config_file = Path(temp_dir) / "config.json"
        config_data = {
            "debug": True,
            "host": "localhost",
            "port": 5000
        }

        with open(config_file, 'w') as f:
            import json
            json.dump(config_data, f)

        with patch('app_setup.CONFIG_PATH', str(config_file)):
            from app_setup import load_config
            result = load_config()

            assert result == config_data

    def test_load_config_from_environment(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            'APP_DEBUG': 'true',
            'APP_HOST': '0.0.0.0',
            'APP_PORT': '8080'
        }

        with patch.dict(os.environ, env_vars):
            from app_setup import load_config_from_env
            result = load_config_from_env()

            assert result['debug'] is True
            assert result['host'] == '0.0.0.0'
            assert result['port'] == 8080

    def test_validate_config_success(self, mock_config):
        """Test successful configuration validation."""
        from app_setup import validate_config

        # Should not raise any exceptions
        validate_config(mock_config)

    def test_validate_config_missing_required_fields(self):
        """Test configuration validation with missing required fields."""
        incomplete_config = {"debug": True}

        from app_setup import validate_config

        with pytest.raises(ValueError, match="Missing required configuration"):
            validate_config(incomplete_config)

    def test_initialize_database_success(self, app_instance):
        """Test successful database initialization."""
        with patch('app_setup.db') as mock_db:
            from app_setup import initialize_database

            initialize_database(app_instance)

            mock_db.create_all.assert_called_once()

    def test_initialize_database_failure(self, app_instance):
        """Test database initialization failure."""
        with patch('app_setup.db') as mock_db:
            mock_db.create_all.side_effect = Exception("Database connection failed")

            from app_setup import initialize_database

            with pytest.raises(Exception, match="Database connection failed"):
                initialize_database(app_instance)

    def test_setup_database_migrations(self, app_instance):
        """Test database migration setup."""
        with patch('app_setup.Migrate') as mock_migrate:
            from app_setup import setup_migrations

            setup_migrations(app_instance)

            mock_migrate.assert_called_once()

    def test_setup_middleware(self, app_instance):
        """Test middleware setup."""
        with patch('app_setup.CORS') as mock_cors:
            from app_setup import setup_middleware

            setup_middleware(app_instance)

            mock_cors.assert_called_once_with(app_instance)

    def test_setup_extensions(self, app_instance):
        """Test extensions setup."""
        with patch('app_setup.db') as mock_db, \
             patch('app_setup.migrate') as mock_migrate, \
             patch('app_setup.login_manager') as mock_login:

            from app_setup import setup_extensions

            setup_extensions(app_instance)

            mock_db.init_app.assert_called_once_with(app_instance)
            mock_migrate.init_app.assert_called_once_with(app_instance, mock_db)
            mock_login.init_app.assert_called_once_with(app_instance)

    def test_setup_logging(self, app_instance, temp_dir):
        """Test logging configuration setup."""
        from app_setup import setup_logging
        log_file = Path(temp_dir) / "app.log"

        with patch('app_setup.logging') as mock_logging:
            setup_logging(app_instance, log_file=str(log_file))

            mock_logging.basicConfig.assert_called_once()

    @pytest.mark.parametrize("debug,expected", [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
    ])
    def test_parse_debug_flag(self, debug, expected):
        """Test parsing of debug flag from different formats."""
        from app_setup import parse_debug_flag

        result = parse_debug_flag(debug)
        assert result == expected

    @pytest.mark.parametrize("port,expected", [
        ("8000", 8000),
        (8000, 8000),
        ("invalid", ValueError),
        (-1, ValueError),
        (70000, ValueError),
    ])
    def test_parse_port_number(self, port, expected):
        """Test parsing of port number with various inputs."""
        from app_setup import parse_port_number

        if expected == ValueError:
            with pytest.raises(ValueError):
                parse_port_number(port)
        else:
            result = parse_port_number(port)
            assert result == expected

    def test_full_app_setup_integration(self, temp_dir, mock_config):
        """Test complete application setup integration."""
        config_file = Path(temp_dir) / "config.json"

        with open(config_file, 'w') as f:
            import json
            json.dump(mock_config, f)

        with patch('app_setup.CONFIG_PATH', str(config_file)), \
             patch('app_setup.Flask') as mock_flask, \
             patch('app_setup.db') as mock_db, \
             patch('app_setup.CORS') as mock_cors:

            mock_app = Mock()
            mock_flask.return_value = mock_app

            from app_setup import create_app
            result = create_app()

            # Verify all setup steps were called
            assert result == mock_app
            mock_flask.assert_called_once()
            mock_app.config.update.assert_called()
            mock_db.init_app.assert_called_with(mock_app)
            mock_cors.assert_called_with(mock_app)

    def test_app_teardown(self, app_instance):
        """Test application teardown process."""
        with patch('app_setup.db') as mock_db:
            from app_setup import teardown_app

            teardown_app(app_instance)

            mock_db.session.remove.assert_called_once()

    def test_handle_startup_errors(self):
        """Test error handling during application startup."""
        with patch('app_setup.Flask') as mock_flask:
            mock_flask.side_effect = ImportError("Failed to import Flask")

            from app_setup import create_app

            with pytest.raises(ImportError):
                create_app()

    def test_cleanup_on_failure(self, app_instance):
        """Test cleanup when application setup fails."""
        with patch('app_setup.setup_extensions') as mock_setup:
            mock_setup.side_effect = Exception("Setup failed")

            with patch('app_setup.cleanup_on_error') as mock_cleanup:
                from app_setup import create_app

                with pytest.raises(Exception):
                    create_app()

                mock_cleanup.assert_called_once()

    def test_configuration_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        edge_cases = [
            {},  # empty config
            {"debug": "invalid"},  # invalid debug value
            {"port": "not_a_number"},  # invalid port
            {"host": ""},  # empty host
        ]

        from app_setup import validate_config

        for config in edge_cases:
            with pytest.raises((ValueError, TypeError)):
                validate_config(config)