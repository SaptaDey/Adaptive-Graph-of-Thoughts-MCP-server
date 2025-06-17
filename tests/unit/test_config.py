import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from adaptive_graph_of_thoughts.config import (
    Config,
    GraphConfig,
    LoggingConfig,
    ModelConfig,
    _config,
    get_config,
    load_config,
    set_config,
)


@pytest.fixture
def sample_model_config():
    """Sample ModelConfig for testing."""
    return {
        "name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 30,
        "api_key": "test_key",
        "base_url": "https://api.test.com",
    }


@pytest.fixture
def sample_graph_config():
    """Sample GraphConfig for testing."""
    return {
        "max_depth": 5,
        "max_breadth": 3,
        "pruning_threshold": 0.1,
        "enable_caching": True,
        "cache_size": 1000,
    }


@pytest.fixture
def sample_logging_config():
    """Sample LoggingConfig for testing."""
    return {
        "level": "INFO",
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        "file_path": "/tmp/test.log",
        "enable_console": True,
    }


@pytest.fixture
def sample_full_config(sample_model_config, sample_graph_config, sample_logging_config):
    """Sample complete configuration for testing."""
    return {
        "model": sample_model_config,
        "graph": sample_graph_config,
        "logging": sample_logging_config,
    }


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 30
        assert config.api_key is None
        assert config.base_url is None

    def test_model_config_custom_values(self, sample_model_config):
        """Test ModelConfig with custom values."""
        config = ModelConfig(**sample_model_config)
        assert config.name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 30
        assert config.api_key == "test_key"
        assert config.base_url == "https://api.test.com"

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_model_config_valid_temperature_range(self, temperature):
        """Test ModelConfig with valid temperature values."""
        config = ModelConfig(temperature=temperature)
        assert config.temperature == temperature

    @pytest.mark.parametrize("max_tokens", [1, 100, 1000, 4096, 8192])
    def test_model_config_valid_max_tokens(self, max_tokens):
        """Test ModelConfig with valid max_tokens values."""
        config = ModelConfig(max_tokens=max_tokens)
        assert config.max_tokens == max_tokens

    @pytest.mark.parametrize("timeout", [1, 10, 30, 60, 120])
    def test_model_config_valid_timeout(self, timeout):
        """Test ModelConfig with valid timeout values."""
        config = ModelConfig(timeout=timeout)
        assert config.timeout == timeout


class TestGraphConfig:
    """Test cases for GraphConfig class."""

    def test_graph_config_defaults(self):
        """Test GraphConfig default values."""
        config = GraphConfig()
        assert config.max_depth == 5
        assert config.max_breadth == 3
        assert config.pruning_threshold == 0.1
        assert config.enable_caching is True
        assert config.cache_size == 1000

    def test_graph_config_custom_values(self, sample_graph_config):
        """Test GraphConfig with custom values."""
        config = GraphConfig(**sample_graph_config)
        assert config.max_depth == 5
        assert config.max_breadth == 3
        assert config.pruning_threshold == 0.1
        assert config.enable_caching is True
        assert config.cache_size == 1000

    @pytest.mark.parametrize("depth", [1, 3, 5, 10, 20])
    def test_graph_config_valid_max_depth(self, depth):
        """Test GraphConfig with valid max_depth values."""
        config = GraphConfig(max_depth=depth)
        assert config.max_depth == depth

    @pytest.mark.parametrize("breadth", [1, 2, 5, 10])
    def test_graph_config_valid_max_breadth(self, breadth):
        """Test GraphConfig with valid max_breadth values."""
        config = GraphConfig(max_breadth=breadth)
        assert config.max_breadth == breadth

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_graph_config_valid_pruning_threshold(self, threshold):
        """Test GraphConfig with valid pruning_threshold values."""
        config = GraphConfig(pruning_threshold=threshold)
        assert config.pruning_threshold == threshold

    @pytest.mark.parametrize("caching", [True, False])
    def test_graph_config_enable_caching(self, caching):
        """Test GraphConfig with different caching settings."""
        config = GraphConfig(enable_caching=caching)
        assert config.enable_caching is caching

    @pytest.mark.parametrize("cache_size", [1, 100, 1000, 5000])
    def test_graph_config_valid_cache_size(self, cache_size):
        """Test GraphConfig with valid cache_size values."""
        config = GraphConfig(cache_size=cache_size)
        assert config.cache_size == cache_size


class TestLoggingConfig:
    """Test cases for LoggingConfig class."""

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        assert config.file_path is None
        assert config.enable_console is True

    def test_logging_config_custom_values(self, sample_logging_config):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(**sample_logging_config)
        assert config.level == "INFO"
        assert config.format == "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        assert config.file_path == "/tmp/test.log"
        assert config.enable_console is True

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_logging_config_valid_levels(self, level):
        """Test LoggingConfig with valid log levels."""
        config = LoggingConfig(level=level)
        assert config.level == level

    @pytest.mark.parametrize("enable_console", [True, False])
    def test_logging_config_console_setting(self, enable_console):
        """Test LoggingConfig with different console settings."""
        config = LoggingConfig(enable_console=enable_console)
        assert config.enable_console is enable_console

    def test_logging_config_custom_format(self):
        """Test LoggingConfig with custom format string."""
        custom_format = "{time} - {level} - {message}"
        config = LoggingConfig(format=custom_format)
        assert config.format == custom_format

    def test_logging_config_file_path(self):
        """Test LoggingConfig with file path."""
        file_path = "/var/log/app.log"
        config = LoggingConfig(file_path=file_path)
        assert config.file_path == file_path


class TestConfig:
    """Test cases for Config class."""

    def test_config_defaults(self):
        """Test Config with default values."""
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.model.name == "gpt-4"
        assert config.graph.max_depth == 5
        assert config.logging.level == "INFO"

    def test_config_from_dict_empty(self):
        """Test Config.from_dict with empty dictionary."""
        config = Config.from_dict({})
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_config_from_dict_partial(self, sample_model_config):
        """Test Config.from_dict with partial configuration."""
        data = {"model": sample_model_config}
        config = Config.from_dict(data)
        assert config.model.name == "gpt-4"
        assert config.model.api_key == "test_key"
        assert config.graph.max_depth == 5  # default
        assert config.logging.level == "INFO"  # default

    def test_config_from_dict_complete(self, sample_full_config):
        """Test Config.from_dict with complete configuration."""
        config = Config.from_dict(sample_full_config)
        assert config.model.name == "gpt-4"
        assert config.model.api_key == "test_key"
        assert config.graph.max_depth == 5
        assert config.graph.cache_size == 1000
        assert config.logging.level == "INFO"
        assert config.logging.file_path == "/tmp/test.log"

    def test_config_to_dict(self, sample_full_config):
        """Test Config.to_dict method."""
        config = Config.from_dict(sample_full_config)
        result = config.to_dict()
        assert isinstance(result, dict)
        assert "model" in result
        assert "graph" in result
        assert "logging" in result
        assert result["model"]["name"] == "gpt-4"
        assert result["graph"]["max_depth"] == 5
        assert result["logging"]["level"] == "INFO"

    def test_config_to_json(self, sample_full_config):
        """Test Config.to_json method."""
        config = Config.from_dict(sample_full_config)
        json_str = config.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "model" in parsed
        assert "graph" in parsed
        assert "logging" in parsed

    def test_config_to_yaml(self, sample_full_config):
        """Test Config.to_yaml method."""
        config = Config.from_dict(sample_full_config)
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)
        parsed = yaml.safe_load(yaml_str)
        assert "model" in parsed
        assert "graph" in parsed
        assert "logging" in parsed


class TestConfigFileOperations:
    """Test cases for Config file operations."""

    def test_config_from_file_json_success(self, sample_full_config, tmp_path):
        """Test successful loading from JSON file."""
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(sample_full_config, f)

        config = Config.from_file(config_file)
        assert config.model.name == "gpt-4"
        assert config.model.api_key == "test_key"
        assert config.graph.max_depth == 5
        assert config.logging.level == "INFO"

    def test_config_from_file_yaml_success(self, sample_full_config, tmp_path):
        """Test successful loading from YAML file."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_full_config, f)

        config = Config.from_file(config_file)
        assert config.model.name == "gpt-4"
        assert config.model.api_key == "test_key"
        assert config.graph.max_depth == 5
        assert config.logging.level == "INFO"

    def test_config_from_file_yml_extension(self, sample_full_config, tmp_path):
        """Test loading from .yml file extension."""
        config_file = tmp_path / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(sample_full_config, f)

        config = Config.from_file(config_file)
        assert config.model.name == "gpt-4"

    def test_config_from_file_not_found(self, tmp_path):
        """Test loading from non-existent file."""
        config_file = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config.from_file(config_file)

    def test_config_from_file_unsupported_format(self, tmp_path):
        """Test loading from unsupported file format."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            Config.from_file(config_file)

    def test_config_from_file_invalid_json(self, tmp_path):
        """Test loading from invalid JSON file."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json}")
        with pytest.raises(ValueError, match="Failed to parse configuration file"):
            Config.from_file(config_file)

    def test_config_from_file_invalid_yaml(self, tmp_path):
        """Test loading from invalid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        with pytest.raises(ValueError, match="Failed to parse configuration file"):
            Config.from_file(config_file)

    def test_config_from_file_empty_file(self, tmp_path):
        """Test loading from empty file."""
        config_file = tmp_path / "config.json"
        config_file.write_text("")
        with pytest.raises(ValueError, match="Failed to parse configuration file"):
            Config.from_file(config_file)

    def test_config_from_file_null_content(self, tmp_path):
        """Test loading from file with null content."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("null")
        config = Config.from_file(config_file)
        assert isinstance(config, Config)
        assert config.model.name == "gpt-4"


class TestConfigEnvironmentLoading:
    """Test cases for loading configuration from environment variables."""

    @patch.dict(
        os.environ,
        {
            "AGOT_MODEL_NAME": "gpt-3.5-turbo",
            "AGOT_MODEL_TEMPERATURE": "0.5",
            "AGOT_MODEL_MAX_TOKENS": "1024",
            "AGOT_MODEL_TIMEOUT": "60",
            "AGOT_MODEL_API_KEY": "env_key",
            "AGOT_MODEL_BASE_URL": "https://env.api.com",
        },
    )
    def test_config_from_env_model_complete(self):
        config = Config.from_env()
        assert config.model.name == "gpt-3.5-turbo"
        assert config.model.temperature == 0.5
        assert config.model.max_tokens == 1024
        assert config.model.timeout == 60
        assert config.model.api_key == "env_key"
        assert config.model.base_url == "https://env.api.com"

    @patch.dict(
        os.environ,
        {
            "AGOT_GRAPH_MAX_DEPTH": "10",
            "AGOT_GRAPH_MAX_BREADTH": "5",
            "AGOT_GRAPH_PRUNING_THRESHOLD": "0.2",
            "AGOT_GRAPH_ENABLE_CACHING": "false",
            "AGOT_GRAPH_CACHE_SIZE": "2000",
        },
    )
    def test_config_from_env_graph_complete(self):
        config = Config.from_env()
        assert config.graph.max_depth == 10
        assert config.graph.max_breadth == 5
        assert config.graph.pruning_threshold == 0.2
        assert config.graph.enable_caching is False
        assert config.graph.cache_size == 2000

    @patch.dict(
        os.environ,
        {
            "AGOT_LOGGING_LEVEL": "DEBUG",
            "AGOT_LOGGING_FORMAT": "custom format",
            "AGOT_LOGGING_FILE_PATH": "/env/log/path",
            "AGOT_LOGGING_ENABLE_CONSOLE": "false",
        },
    )
    def test_config_from_env_logging_complete(self):
        config = Config.from_env()
        assert config.logging.level == "DEBUG"
        assert config.logging.format == "custom format"
        assert config.logging.file_path == "/env/log/path"
        assert config.logging.enable_console is False

    @patch.dict(os.environ, {"AGOT_MODEL_NAME": "claude"})
    def test_config_from_env_partial(self):
        config = Config.from_env()
        assert config.model.name == "claude"
        assert config.model.temperature == 0.7
        assert config.graph.max_depth == 5

    @patch.dict(
        os.environ, {"CUSTOM_MODEL_NAME": "custom-model", "CUSTOM_GRAPH_MAX_DEPTH": "8"}
    )
    def test_config_from_env_custom_prefix(self):
        config = Config.from_env(prefix="CUSTOM_")
        assert config.model.name == "custom-model"
        assert config.graph.max_depth == 8

    def test_config_from_env_no_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            assert config.model.name == "gpt-4"
            assert config.graph.max_depth == 5
            assert config.logging.level == "INFO"

    @patch.dict(
        os.environ,
        {"AGOT_GRAPH_ENABLE_CACHING": "TRUE", "AGOT_LOGGING_ENABLE_CONSOLE": "False"},
    )
    def test_config_from_env_boolean_variations(self):
        config = Config.from_env()
        assert config.graph.enable_caching is True
        assert config.logging.enable_console is False

    @patch.dict(
        os.environ,
        {
            "AGOT_MODEL_TEMPERATURE": "invalid_float",
            "AGOT_GRAPH_MAX_DEPTH": "invalid_int",
        },
    )
    def test_config_from_env_invalid_values(self):
        with pytest.raises(ValueError):
            Config.from_env()


class TestConfigFileSaving:
    """Test cases for Config file saving operations."""

    def test_save_to_file_json(self, sample_full_config, tmp_path):
        config = Config.from_dict(sample_full_config)
        config_file = tmp_path / "output.json"
        config.save_to_file(config_file)
        assert config_file.exists()
        with open(config_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data["model"]["name"] == "gpt-4"

    def test_save_to_file_yaml(self, sample_full_config, tmp_path):
        config = Config.from_dict(sample_full_config)
        config_file = tmp_path / "output.yaml"
        config.save_to_file(config_file)
        assert config_file.exists()
        with open(config_file, "r") as f:
            saved_data = yaml.safe_load(f)
        assert saved_data["model"]["name"] == "gpt-4"

    def test_save_to_file_unsupported_format(self, tmp_path):
        config = Config()
        config_file = tmp_path / "output.txt"
        with pytest.raises(ValueError, match="Unsupported file format"):
            config.save_to_file(config_file)

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_save_to_file_permission_error(self, mock_open):
        config = Config()
        with pytest.raises(RuntimeError, match="Failed to save configuration"):
            config.save_to_file("test.json")

    def test_save_to_file_roundtrip_json(self, sample_full_config, tmp_path):
        original_config = Config.from_dict(sample_full_config)
        config_file = tmp_path / "roundtrip.json"
        original_config.save_to_file(config_file)
        loaded_config = Config.from_file(config_file)
        assert loaded_config.to_dict() == original_config.to_dict()

    def test_save_to_file_roundtrip_yaml(self, sample_full_config, tmp_path):
        original_config = Config.from_dict(sample_full_config)
        config_file = tmp_path / "roundtrip.yaml"
        original_config.save_to_file(config_file)
        loaded_config = Config.from_file(config_file)
        assert loaded_config.to_dict() == original_config.to_dict()


class TestConfigValidation:
    """Test cases for configuration validation."""

    def test_validate_success(self, sample_full_config):
        config = Config.from_dict(sample_full_config)
        config.validate()

    @pytest.mark.parametrize("temperature", [-0.1, -1.0, 2.1, 3.0, float("inf")])
    def test_validate_invalid_temperature(self, temperature):
        config = Config()
        config.model.temperature = temperature
        with pytest.raises(
            ValueError, match="Model temperature must be between 0.0 and 2.0"
        ):
            config.validate()

    @pytest.mark.parametrize("max_tokens", [0, -1, -100])
    def test_validate_invalid_max_tokens(self, max_tokens):
        config = Config()
        config.model.max_tokens = max_tokens
        with pytest.raises(ValueError, match="Model max_tokens must be positive"):
            config.validate()

    @pytest.mark.parametrize("timeout", [0, -1, -30])
    def test_validate_invalid_timeout(self, timeout):
        config = Config()
        config.model.timeout = timeout
        with pytest.raises(ValueError, match="Model timeout must be positive"):
            config.validate()

    @pytest.mark.parametrize("max_depth", [0, -1, -5])
    def test_validate_invalid_max_depth(self, max_depth):
        config = Config()
        config.graph.max_depth = max_depth
        with pytest.raises(ValueError, match="Graph max_depth must be positive"):
            config.validate()

    @pytest.mark.parametrize("max_breadth", [0, -1, -3])
    def test_validate_invalid_max_breadth(self, max_breadth):
        config = Config()
        config.graph.max_breadth = max_breadth
        with pytest.raises(ValueError, match="Graph max_breadth must be positive"):
            config.validate()

    @pytest.mark.parametrize("threshold", [-0.1, -1.0, 1.1, 2.0])
    def test_validate_invalid_pruning_threshold(self, threshold):
        config = Config()
        config.graph.pruning_threshold = threshold
        with pytest.raises(
            ValueError, match="Graph pruning_threshold must be between 0.0 and 1.0"
        ):
            config.validate()

    @pytest.mark.parametrize("cache_size", [0, -1, -1000])
    def test_validate_invalid_cache_size(self, cache_size):
        config = Config()
        config.graph.cache_size = cache_size
        with pytest.raises(ValueError, match="Graph cache_size must be positive"):
            config.validate()

    @pytest.mark.parametrize("level", ["INVALID", "debug", "info", "TRACE", ""])
    def test_validate_invalid_logging_level(self, level):
        config = Config()
        config.logging.level = level
        with pytest.raises(ValueError, match="Logging level must be one of"):
            config.validate()

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_validate_valid_logging_levels(self, level):
        config = Config()
        config.logging.level = level
        config.validate()


class TestConfigUpdate:
    """Test cases for configuration update functionality."""

    def test_update_model_config(self):
        config = Config()
        config.update(model={"name": "updated-model", "temperature": 0.9})
        assert config.model.name == "updated-model"
        assert config.model.temperature == 0.9
        assert config.model.max_tokens == 2048

    def test_update_graph_config(self):
        config = Config()
        config.update(graph={"max_depth": 10, "enable_caching": False})
        assert config.graph.max_depth == 10
        assert config.graph.enable_caching is False
        assert config.graph.max_breadth == 3

    def test_update_logging_config(self):
        config = Config()
        config.update(logging={"level": "DEBUG", "file_path": "/new/path"})
        assert config.logging.level == "DEBUG"
        assert config.logging.file_path == "/new/path"
        assert config.logging.enable_console is True

    def test_update_unknown_key_warning(self, caplog):
        config = Config()
        config.update(unknown_key="value")
        assert "Unknown config key: unknown_key" in caplog.text

    def test_update_unknown_nested_key_warning(self, caplog):
        config = Config()
        config.update(model={"unknown_nested": "value"})
        assert "Unknown nested config key: model.unknown_nested" in caplog.text

    def test_update_multiple_sections(self):
        config = Config()
        config.update(
            model={"name": "new-model"},
            graph={"max_depth": 8},
            logging={"level": "ERROR"},
        )
        assert config.model.name == "new-model"
        assert config.graph.max_depth == 8
        assert config.logging.level == "ERROR"


class TestGlobalConfigManagement:
    """Test cases for global configuration management."""

    def setup_method(self):
        global _config
        _config = None

    def test_get_config_default(self):
        config = get_config()
        assert isinstance(config, Config)
        assert config.model.name == "gpt-4"

    def test_set_config_valid(self):
        new_config = Config()
        new_config.model.name = "custom-model"
        set_config(new_config)
        assert get_config().model.name == "custom-model"

    def test_set_config_invalid(self):
        invalid_config = Config()
        invalid_config.model.temperature = -1.0
        with pytest.raises(ValueError):
            set_config(invalid_config)

    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2


class TestLoadConfig:
    """Test cases for load_config function."""

    def setup_method(self):
        global _config
        _config = None

    def test_load_config_no_sources(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config()
            assert isinstance(cfg, Config)
            assert cfg is get_config()

    def test_load_config_file_only(self, sample_full_config, tmp_path):
        file = tmp_path / "cfg.json"
        with open(file, "w") as f:
            json.dump(sample_full_config, f)
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config(file_path=file)
            assert cfg.model.api_key == "test_key"
            assert cfg.graph.cache_size == 1000

    @patch.dict(
        os.environ, {"AGOT_MODEL_NAME": "env-model", "AGOT_GRAPH_MAX_DEPTH": "15"}
    )
    def test_load_config_env_only(self):
        cfg = load_config()
        assert cfg.model.name == "env-model"
        assert cfg.graph.max_depth == 15

    def test_load_config_file_and_env_override(self, sample_full_config, tmp_path):
        file = tmp_path / "cfg.json"
        with open(file, "w") as f:
            json.dump(sample_full_config, f)
        with patch.dict(
            os.environ,
            {"AGOT_MODEL_NAME": "env-override", "AGOT_GRAPH_MAX_DEPTH": "20"},
        ):
            cfg = load_config(file_path=file, env_prefix="AGOT_")
            assert cfg.model.name == "env-override"
            assert cfg.graph.max_depth == 20
            assert cfg.model.api_key == "test_key"
            assert cfg.logging.file_path == "/tmp/test.log"

    def test_load_config_invalid_file_warning(self, tmp_path, caplog):
        file = tmp_path / "bad.json"
        file.write_text("{bad}")
        cfg = load_config(file_path=file)
        assert "Failed to load config from file" in caplog.text
        assert isinstance(cfg, Config)

    def test_load_config_nonexistent_file_warning(self, caplog):
        cfg = load_config(file_path="nope.json")
        assert "Failed to load config from file" in caplog.text
        assert isinstance(cfg, Config)

    @patch.dict(os.environ, {"AGOT_MODEL_TEMPERATURE": "invalid"})
    def test_load_config_invalid_env_warning(self, caplog):
        cfg = load_config()
        assert "Failed to load config from environment" in caplog.text
        assert isinstance(cfg, Config)

    def test_load_config_custom_env_prefix(self, tmp_path):
        with patch.dict(
            os.environ, {"CUSTOM_MODEL_NAME": "custom", "CUSTOM_GRAPH_MAX_DEPTH": "12"}
        ):
            cfg = load_config(env_prefix="CUSTOM_")
            assert cfg.model.name == "custom"
            assert cfg.graph.max_depth == 12

    def test_load_config_sets_global(self, sample_full_config, tmp_path):
        file = tmp_path / "cfg.json"
        with open(file, "w") as f:
            json.dump(sample_full_config, f)
        loaded = load_config(file_path=file)
        assert loaded is get_config()
        assert get_config().model.api_key == "test_key"


class TestConfigEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def test_config_with_none_values(self):
        data = {
            "model": {"name": None, "api_key": None},
            "graph": {"enable_caching": None},
            "logging": {"file_path": None},
        }
        cfg = Config.from_dict(data)
        assert cfg.model.name is None
        assert cfg.model.api_key is None
        assert cfg.graph.enable_caching is None
        assert cfg.logging.file_path is None

    def test_config_serialization_with_special_characters(self):
        cfg = Config()
        cfg.model.api_key = "key_with_特殊字符_and_🚀"
        cfg.logging.file_path = "/path/含空格_和_unicode.log"
        j = cfg.to_json()
        parsed_j = json.loads(j)
        assert "特殊字符" in parsed_j["model"]["api_key"]
        y = cfg.to_yaml()
        parsed_y = yaml.safe_load(y)
        assert "特殊字符" in parsed_y["model"]["api_key"]

    @pytest.mark.parametrize("file_size_kb", [1, 10, 100, 1000])
    def test_config_large_file_handling(self, file_size_kb, tmp_path):
        large = "x" * (file_size_kb * 1024)
        data = {"model": {"api_key": large}, "logging": {"format": large}}
        file = tmp_path / f"large_{file_size_kb}kb.json"
        with open(file, "w") as f:
            json.dump(data, f)
        cfg = Config.from_file(file)
        assert len(cfg.model.api_key) == file_size_kb * 1024
        assert len(cfg.logging.format) == file_size_kb * 1024
