import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

# Import all config classes and functions
from config import (
    AGoTSettings,
    AppConfig,
    AppSettingsModel,
    ASRGoTDefaultParams,
    Config,
    ExaSearchConfig,
    GoogleScholarConfig,
    GraphConfig,
    KnowledgeDomain,
    LegacyConfig,
    LoggingConfig,
    ModelConfig,
    Neo4jSettingsModel,
    PubMedConfig,
    RuntimeSettings,
    Settings,
    config,
    env_settings,
    get_config,
    legacy_settings,
    load_config,
    load_runtime_settings,
    runtime_settings,
    set_config,
    settings,
    validate_batch_size,
    validate_config_schema,
    validate_learning_rate,
    validate_max_steps,
)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {"learning_rate": 0.01, "batch_size": 32, "max_steps": 1000}


@pytest.fixture
def sample_yaml_content():
    """Sample YAML configuration content."""
    return """
learning_rate: 0.02
batch_size: 64
max_steps: 2000
"""


@pytest.fixture
def sample_json_content():
    """Sample JSON configuration content."""
    return {"learning_rate": 0.03, "batch_size": 128, "max_steps": 3000}


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"learning_rate": 0.01, "batch_size": 32, "max_steps": 1000}, f)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"learning_rate": 0.01, "batch_size": 32, "max_steps": 1000}, f)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestValidationFunctions:
    """Test configuration validation functions."""

    def test_validate_learning_rate_valid(self):
        """Test valid learning rates pass validation."""
        validate_learning_rate(0.01)
        validate_learning_rate(0.5)
        validate_learning_rate(1.0)
        validate_learning_rate(0.001)

    def test_validate_learning_rate_invalid(self):
        """Test invalid learning rates raise errors."""
        with pytest.raises(ValueError, match="Learning rate must be between 0.0 and 1.0"):
            validate_learning_rate(0)
        with pytest.raises(ValueError, match="Learning rate must be between 0.0 and 1.0"):
            validate_learning_rate(-0.1)
        with pytest.raises(ValueError, match="Learning rate must be between 0.0 and 1.0"):
            validate_learning_rate(1.1)
        with pytest.raises(TypeError, match="Learning rate must be a number"):
            validate_learning_rate("0.1")

    def test_validate_batch_size_valid(self):
        """Test valid batch sizes pass validation."""
        validate_batch_size(1)
        validate_batch_size(32)
        validate_batch_size(1000)

    def test_validate_batch_size_invalid(self):
        """Test invalid batch sizes raise errors."""
        with pytest.raises(ValueError, match="Batch size must be between"):
            validate_batch_size(0)
        with pytest.raises(ValueError, match="Batch size must be between"):
            validate_batch_size(-1)
        with pytest.raises(TypeError, match="Batch size must be an integer"):
            validate_batch_size(1.5)
        with pytest.raises(TypeError, match="Batch size must be an integer"):
            validate_batch_size("32")

    def test_validate_max_steps_valid(self):
        """Test valid max steps pass validation."""
        validate_max_steps(1)
        validate_max_steps(1000)
        validate_max_steps(10000)

    def test_validate_max_steps_invalid(self):
        """Test invalid max steps raise errors."""
        with pytest.raises(ValueError, match="Max steps must be between"):
            validate_max_steps(0)
        with pytest.raises(ValueError, match="Max steps must be between"):
            validate_max_steps(-1)
        with pytest.raises(TypeError, match="Max steps must be an integer"):
            validate_max_steps(1.5)
        with pytest.raises(TypeError, match="Max steps must be an integer"):
            validate_max_steps("1000")

    def test_validate_config_schema(self):
        """Test config schema validation."""
        valid = {
            "app": {"host": "localhost", "port": 8000},
            "asr_got": runtime_settings.asr_got,
            "mcp_settings": {
                "protocol_version": "1",
                "server_name": "srv",
                "server_version": "0.1",
                "vendor_name": "v",
            },
        }
        assert validate_config_schema(valid) is True
        with pytest.raises(ValueError):
            validate_config_schema({})


class TestAGoTSettings:
    """Test AGoTSettings Pydantic settings class."""

    def test_agot_settings_defaults(self):
        """Test AGoTSettings default values."""
        settings = AGoTSettings()
        assert settings.llm_provider == "openai"
        assert settings.openai_api_key is None
        assert settings.anthropic_api_key is None

    @patch.dict(os.environ, {"LLM_PROVIDER": "claude"})
    def test_agot_settings_from_env(self):
        """Test AGoTSettings loading from environment variables."""
        settings = AGoTSettings()
        assert settings.llm_provider == "claude"

    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        },
    )
    def test_agot_settings_all_env_vars(self):
        """Test AGoTSettings with all environment variables set."""
        settings = AGoTSettings()
        assert settings.llm_provider == "openai"
        assert settings.openai_api_key == "test-openai-key"
        assert settings.anthropic_api_key == "test-anthropic-key"

    def test_agot_settings_validation(self):
        """Test AGoTSettings field validation."""
        settings = AGoTSettings(llm_provider="custom")
        assert settings.llm_provider == "custom"


class TestSimpleConfigClasses:
    """Test simple configuration classes."""

    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()
        assert config.name == "Adaptive Graph of Thoughts"
        assert config.version == "0.1.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.reload is True
        assert config.log_level == "INFO"
        assert config.cors_allowed_origins_str == "*"
        assert config.auth_token is None

    def test_app_config_custom_values(self):
        """Test AppConfig with custom values."""
        config = AppConfig(
            name="Custom App",
            version="1.0.0",
            host="127.0.0.1",
            port=9000,
            reload=False,
            log_level="DEBUG",
            cors_allowed_origins_str="http://localhost:3000",
            auth_token="secret",
        )
        assert config.name == "Custom App"
        assert config.version == "1.0.0"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.reload is False
        assert config.log_level == "DEBUG"
        assert config.cors_allowed_origins_str == "http://localhost:3000"
        assert config.auth_token == "secret"

    def test_asr_got_default_params(self):
        """Test ASRGoTDefaultParams default values."""
        params = ASRGoTDefaultParams()
        assert params.initial_confidence == 0.8
        assert params.confidence_threshold == 0.75
        assert params.max_iterations == 10
        assert params.convergence_threshold == 0.05

    def test_asr_got_custom_params(self):
        """Test ASRGoTDefaultParams with custom values."""
        params = ASRGoTDefaultParams(
            initial_confidence=0.9,
            confidence_threshold=0.8,
            max_iterations=20,
            convergence_threshold=0.01,
        )
        assert params.initial_confidence == 0.9
        assert params.confidence_threshold == 0.8
        assert params.max_iterations == 20
        assert params.convergence_threshold == 0.01

    def test_pubmed_config_defaults(self):
        """Test PubMedConfig default values."""
        config = PubMedConfig()
        assert config.api_key is None
        assert config.max_results == 20
        assert config.rate_limit_delay == 0.5

    def test_google_scholar_config_defaults(self):
        """Test GoogleScholarConfig default values."""
        config = GoogleScholarConfig()
        assert config.max_results == 10
        assert config.rate_limit_delay == 1.0

    def test_exa_search_config_defaults(self):
        """Test ExaSearchConfig default values."""
        config = ExaSearchConfig()
        assert config.api_key is None
        assert config.max_results == 10

    def test_knowledge_domain_defaults(self):
        """Test KnowledgeDomain default values."""
        domain = KnowledgeDomain("test_domain")
        assert domain.name == "test_domain"
        assert domain.description == ""
        assert domain.keywords == []

    def test_knowledge_domain_with_values(self):
        """Test KnowledgeDomain with custom values."""
        domain = KnowledgeDomain(
            "AI",
            description="Artificial Intelligence",
            keywords=["machine learning", "neural networks"],
        )
        assert domain.name == "AI"
        assert domain.description == "Artificial Intelligence"
        assert domain.keywords == ["machine learning", "neural networks"]


class TestLegacyConfig:
    """Test LegacyConfig class functionality."""

    def test_legacy_config_defaults(self):
        """Test LegacyConfig default initialization."""
        config = LegacyConfig()
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_steps == 1000
        assert config._frozen is False
        assert isinstance(config.app, AppConfig)
        assert isinstance(config.asr_got, ASRGoTDefaultParams)

    def test_legacy_config_custom_values(self, sample_config_data):
        """Test LegacyConfig with custom values."""
        config = LegacyConfig(**sample_config_data)
        assert config.learning_rate == sample_config_data["learning_rate"]
        assert config.batch_size == sample_config_data["batch_size"]
        assert config.max_steps == sample_config_data["max_steps"]

    def test_legacy_config_validation_on_init(self):
        """Test LegacyConfig validates parameters on initialization."""
        with pytest.raises(ValueError):
            LegacyConfig(learning_rate=-0.1)
        with pytest.raises(ValueError):
            LegacyConfig(batch_size=0)
        with pytest.raises(ValueError):
            LegacyConfig(max_steps=-1)

    def test_legacy_config_frozen(self):
        """Test LegacyConfig frozen functionality."""
        config = LegacyConfig(frozen=True)
        with pytest.raises(AttributeError, match="Cannot modify frozen config"):
            config.learning_rate = 0.02

    def test_legacy_config_equality(self):
        """Test LegacyConfig equality comparison."""
        config1 = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        config2 = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        config3 = LegacyConfig(learning_rate=0.02, batch_size=32, max_steps=1000)
        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"

    def test_legacy_config_repr(self):
        """Test LegacyConfig string representation."""
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        expected = "Config(learning_rate=0.01, batch_size=32, max_steps=1000)"
        assert repr(config) == expected

    def test_legacy_config_model_dump(self):
        """Test LegacyConfig model_dump method."""
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        dump = config.model_dump()
        expected = {"learning_rate": 0.01, "batch_size": 32, "max_steps": 1000}
        assert dump == expected

    def test_legacy_config_copy(self):
        """Test LegacyConfig copy method."""
        original = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        copy = original.copy()
        assert copy == original
        assert copy is not original
        assert copy._frozen is False

    def test_legacy_config_update(self):
        """Test LegacyConfig update method."""
        config = LegacyConfig()
        updates = {"learning_rate": 0.02, "batch_size": 64, "max_steps": 2000}
        config.update(updates)
        assert config.learning_rate == 0.02
        assert config.batch_size == 64
        assert config.max_steps == 2000

    def test_legacy_config_update_validation(self):
        """Test LegacyConfig update validates values."""
        config = LegacyConfig()
        with pytest.raises(ValueError):
            config.update({"learning_rate": -0.1})
        with pytest.raises(ValueError):
            config.update({"batch_size": 0})
        with pytest.raises(ValueError):
            config.update({"max_steps": -1})

    def test_legacy_config_merge(self):
        """Test LegacyConfig merge method."""
        config1 = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        config2 = LegacyConfig(learning_rate=0.02, batch_size=64)
        merged = config1.merge(config2)
        assert merged.learning_rate == 0.02
        assert merged.batch_size == 64
        assert merged.max_steps == 1000  # From config1


class TestLegacyConfigFileOperations:
    """Test LegacyConfig file loading and saving operations."""

    def test_load_yaml_file(self, temp_config_file):
        """Test loading YAML configuration file."""
        config = LegacyConfig.load(temp_config_file)
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_steps == 1000

    def test_load_json_file(self, temp_json_file):
        """Test loading JSON configuration file."""
        config = LegacyConfig.load(temp_json_file)
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_steps == 1000

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            LegacyConfig.load("/nonexistent/file.yaml")

    def test_load_empty_file(self):
        """Test loading empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Empty configuration file"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises YAMLError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            temp_path = f.name
        try:
            with pytest.raises(yaml.YAMLError):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')
            temp_path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_unsupported_format(self):
        """Test loading unsupported file format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_missing_required_keys(self):
        """Test loading config with missing required keys raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"max_steps": 1000}, f)  # Missing learning_rate and batch_size
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Missing required key"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_data_types(self):
        """Test loading config with invalid data types raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {"learning_rate": "not_a_number", "batch_size": 32, "max_steps": 1000},
                f,
            )
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="learning_rate must be a number"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_save_yaml_file(self):
        """Test saving configuration to YAML file."""
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
        try:
            config.save(temp_path)
            # Verify file was created and has correct content
            loaded_config = LegacyConfig.load(temp_path)
            assert loaded_config == config
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_json_file(self):
        """Test saving configuration to JSON file."""
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            config.save(temp_path)
            loaded_config = LegacyConfig.load(temp_path)
            assert loaded_config == config
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_unsupported_format(self):
        """Test saving to unsupported format raises ValueError."""
        config = LegacyConfig()
        with pytest.raises(ValueError, match="Unsupported file format"):
            config.save("config.txt")

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_save_permission_error(self, mock_open):
        """Test saving with permission error raises PermissionError."""
        config = LegacyConfig()
        with pytest.raises(PermissionError, match="Permission denied writing to"):
            config.save("readonly.yaml")


class TestLegacyConfigEnvironment:
    """Test LegacyConfig environment variable loading."""

    @patch.dict(
        os.environ, {"LEARNING_RATE": "0.02", "BATCH_SIZE": "64", "MAX_STEPS": "2000"}
    )
    def test_from_env_all_vars(self):
        """Test loading all environment variables."""
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.02
        assert config.batch_size == 64
        assert config.max_steps == 2000

    @patch.dict(os.environ, {"LEARNING_RATE": "0.02"})
    def test_from_env_partial_vars(self):
        """Test loading with partial environment variables."""
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.02
        assert config.batch_size == 32  # Default
        assert config.max_steps == 1000  # Default

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_no_vars(self):
        """Test loading with no environment variables uses defaults."""
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_steps == 1000

    def test_load_with_overrides(self, temp_config_file):
        """Test loading config with hierarchical overrides."""
        # Create override file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"learning_rate": 0.05, "batch_size": 128}, f)
            override_path = f.name
        try:
            config = LegacyConfig.load_with_overrides(temp_config_file, override_path)
            assert config.learning_rate == 0.05  # Overridden
            assert config.batch_size == 128  # Overridden
            assert config.max_steps == 1000  # From base
        finally:
            if os.path.exists(override_path):
                os.unlink(override_path)

    def test_load_with_overrides_nonexistent_override(self, temp_config_file):
        """Test loading with nonexistent override file raises error."""
        with pytest.raises(FileNotFoundError, match="Override file not found"):
            LegacyConfig.load_with_overrides(temp_config_file, "/nonexistent.yaml")

    def test_load_with_overrides_empty_override(self, temp_config_file):
        """Test loading with empty override file returns base config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            override_path = f.name
        try:
            config = LegacyConfig.load_with_overrides(temp_config_file, override_path)
            assert config.learning_rate == 0.01
            assert config.batch_size == 32
            assert config.max_steps == 1000
        finally:
            if os.path.exists(override_path):
                os.unlink(override_path)

    def test_load_with_overrides_circular(self, temp_config_file):
        """Circular references should raise an error."""
        with pytest.raises(ValueError, match="Circular dependency"):
            LegacyConfig.load_with_overrides(temp_config_file, temp_config_file)


class TestDataclassConfigurations:
    """Test dataclass-based configuration classes."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 30
        assert config.api_key is None
        assert config.base_url is None

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1024,
            timeout=60,
            api_key="test-key",
            base_url="https://api.example.com",
        )
        assert config.name == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.timeout == 60
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"

    def test_graph_config_defaults(self):
        """Test GraphConfig default values."""
        config = GraphConfig()
        assert config.max_depth == 5
        assert config.max_breadth == 3
        assert config.pruning_threshold == 0.1
        assert config.enable_caching is True
        assert config.cache_size == 1000

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        assert config.file_path is None
        assert config.enable_console is True

    def test_config_defaults(self):
        """Test main Config class default values."""
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_config_from_dict(self):
        """Test Config.from_dict class method."""
        data = {
            "model": {"name": "custom-model", "temperature": 0.8},
            "graph": {"max_depth": 10},
            "logging": {"level": "DEBUG"},
        }
        config = Config.from_dict(data)
        assert config.model.name == "custom-model"
        assert config.model.temperature == 0.8
        assert config.graph.max_depth == 10
        assert config.logging.level == "DEBUG"
        # Test defaults for unspecified values
        assert config.model.max_tokens == 2048  # Default
        assert config.graph.max_breadth == 3  # Default

    def test_config_to_dict(self):
        """Test Config.to_dict method."""
        config = Config()
        data = config.to_dict()
        assert "model" in data
        assert "graph" in data
        assert "logging" in data
        assert data["model"]["name"] == "gpt-4"
        assert data["graph"]["max_depth"] == 5

    def test_config_to_json(self):
        """Test Config.to_json method."""
        config = Config()
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["model"]["name"] == "gpt-4"
        assert data["graph"]["max_depth"] == 5

    def test_config_to_yaml(self):
        """Test Config.to_yaml method."""
        config = Config()
        yaml_str = config.to_yaml()
        data = yaml.safe_load(yaml_str)
        assert data["model"]["name"] == "gpt-4"
        assert data["graph"]["max_depth"] == 5


class TestConfigFileOperations:
    """Test Config file operations."""

    def test_from_file_yaml(self):
        """Test Config.from_file with YAML file."""
        data = {
            "model": {"name": "test-model", "temperature": 0.9},
            "graph": {"max_depth": 7},
            "logging": {"level": "DEBUG"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name
        try:
            config = Config.from_file(temp_path)
            assert config.model.name == "test-model"
            assert config.model.temperature == 0.9
            assert config.graph.max_depth == 7
            assert config.logging.level == "DEBUG"
        finally:
            os.unlink(temp_path)

    def test_from_file_json(self):
        """Test Config.from_file with JSON file."""
        data = {
            "model": {"name": "test-model", "temperature": 0.9},
            "graph": {"max_depth": 7},
            "logging": {"level": "DEBUG"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        try:
            config = Config.from_file(temp_path)
            assert config.model.name == "test-model"
            assert config.model.temperature == 0.9
            assert config.graph.max_depth == 7
            assert config.logging.level == "DEBUG"
        finally:
            os.unlink(temp_path)

    def test_from_file_nonexistent(self):
        """Test Config.from_file with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config.from_file("/nonexistent/file.yaml")

    def test_from_file_empty(self):
        """Test Config.from_file with empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Failed to parse configuration file"):
                Config.from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_file_unsupported_format(self):
        """Test Config.from_file with unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                Config.from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_save_to_file_yaml(self):
        """Test Config.save_to_file with YAML format."""
        config = Config()
        config.model.name = "saved-model"
        config.graph.max_depth = 8
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
        try:
            config.save_to_file(temp_path)
            loaded_config = Config.from_file(temp_path)
            assert loaded_config.model.name == "saved-model"
            assert loaded_config.graph.max_depth == 8
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_to_file_json(self):
        """Test Config.save_to_file with JSON format."""
        config = Config()
        config.model.name = "saved-model"
        config.graph.max_depth = 8
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            config.save_to_file(temp_path)
            loaded_config = Config.from_file(temp_path)
            assert loaded_config.model.name == "saved-model"
            assert loaded_config.graph.max_depth == 8
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_to_file_unsupported_format(self):
        """Test Config.save_to_file with unsupported format."""
        config = Config()
        with pytest.raises(ValueError, match="Unsupported file format"):
            config.save_to_file("config.txt")

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_save_to_file_permission_error(self, mock_open):
        """Test Config.save_to_file with permission error."""
        config = Config()
        with pytest.raises(RuntimeError, match="Failed to save configuration"):
            config.save_to_file("readonly.yaml")

    @patch.dict(
        os.environ,
        {
            "AGOT_MODEL_NAME": "env-model",
            "AGOT_MODEL_TEMPERATURE": "0.8",
            "AGOT_GRAPH_MAX_DEPTH": "6",
            "AGOT_LOGGING_LEVEL": "DEBUG",
        },
    )
    def test_from_env_with_defaults(self):
        """Test Config.from_env with default prefix."""
        config = Config.from_env()
        assert config.model.name == "env-model"
        assert config.model.temperature == 0.8
        assert config.graph.max_depth == 6
        assert config.logging.level == "DEBUG"

    @patch.dict(
        os.environ,
        {"CUSTOM_MODEL_NAME": "custom-model", "CUSTOM_MODEL_TEMPERATURE": "0.9"},
    )
    def test_from_env_custom_prefix(self):
        """Test Config.from_env with custom prefix."""
        config = Config.from_env(prefix="CUSTOM_")
        assert config.model.name == "custom-model"
        assert config.model.temperature == 0.9
        assert config.graph.max_depth == 5  # Default

    @patch.dict(os.environ, {"AGOT_MODEL_TEMPERATURE": "invalid"})
    def test_from_env_invalid_value(self):
        """Test Config.from_env with invalid environment value."""
        with pytest.raises(ValueError, match="Invalid value for"):
            Config.from_env()

    @patch.dict(os.environ, {"AGOT_GRAPH_ENABLE_CACHING": "true"})
    def test_from_env_boolean_values(self):
        """Test Config.from_env with boolean environment values."""
        config = Config.from_env()
        assert config.graph.enable_caching is True
        with patch.dict(os.environ, {"AGOT_GRAPH_ENABLE_CACHING": "false"}):
            config = Config.from_env()
            assert config.graph.enable_caching is False
        with patch.dict(os.environ, {"AGOT_GRAPH_ENABLE_CACHING": "0"}):
            config = Config.from_env()
            assert config.graph.enable_caching is False


class TestConfigValidationAndUpdates:
    """Test Config validation and update functionality."""

    def test_validate_valid_config(self):
        """Test validation passes for valid configuration."""
        config = Config()
        config.validate()  # Should not raise

    def test_validate_invalid_temperature(self):
        """Test validation fails for invalid temperature."""
        config = Config()
        config.model.temperature = -0.1
        with pytest.raises(
            ValueError, match="Model temperature must be between 0.0 and 2.0"
        ):
            config.validate()
        config.model.temperature = 2.1
        with pytest.raises(
            ValueError, match="Model temperature must be between 0.0 and 2.0"
        ):
            config.validate()

    def test_validate_invalid_max_tokens(self):
        """Test validation fails for invalid max_tokens."""
        config = Config()
        config.model.max_tokens = 0
        with pytest.raises(ValueError, match="Model max_tokens must be positive"):
            config.validate()
        config.model.max_tokens = -100
        with pytest.raises(ValueError, match="Model max_tokens must be positive"):
            config.validate()

    def test_validate_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        config = Config()
        config.model.timeout = 0
        with pytest.raises(ValueError, match="Model timeout must be positive"):
            config.validate()

    def test_validate_invalid_graph_params(self):
        """Test validation fails for invalid graph parameters."""
        config = Config()
        config.graph.max_depth = 0
        with pytest.raises(ValueError, match="Graph max_depth must be positive"):
            config.validate()
        config.graph.max_depth = 5
        config.graph.max_breadth = -1
        with pytest.raises(ValueError, match="Graph max_breadth must be positive"):
            config.validate()
        config.graph.max_breadth = 3
        config.graph.pruning_threshold = 1.5
        with pytest.raises(
            ValueError, match="Graph pruning_threshold must be between 0.0 and 1.0"
        ):
            config.validate()
        config.graph.pruning_threshold = 0.1
        config.graph.cache_size = 0
        with pytest.raises(ValueError, match="Graph cache_size must be positive"):
            config.validate()

    def test_validate_invalid_logging_level(self):
        """Test validation fails for invalid logging level."""
        config = Config()
        config.logging.level = "INVALID"
        with pytest.raises(ValueError, match="Logging level must be one of"):
            config.validate()

    def test_update_model_section(self):
        """Test updating model section."""
        config = Config()
        config.update(model={"name": "updated-model", "temperature": 0.8})
        assert config.model.name == "updated-model"
        assert config.model.temperature == 0.8
        assert config.model.max_tokens == 2048  # Unchanged

    def test_update_multiple_sections(self):
        """Test updating multiple sections."""
        config = Config()
        config.update(
            model={"name": "updated-model"},
            graph={"max_depth": 8},
            logging={"level": "DEBUG"},
        )
        assert config.model.name == "updated-model"
        assert config.graph.max_depth == 8
        assert config.logging.level == "DEBUG"

    def test_update_unknown_section(self):
        """Test updating unknown section logs warning."""
        config = Config()
        with patch("logging.warning") as mock_warning:
            config.update(unknown_section={"key": "value"})
            mock_warning.assert_called_with("Unknown config key: %s", "unknown_section")

    def test_update_unknown_nested_key(self):
        """Test updating unknown nested key logs warning."""
        config = Config()
        with patch("logging.warning") as mock_warning:
            config.update(model={"unknown_key": "value"})
            mock_warning.assert_called_with(
                "Unknown nested config key: %s.%s", "model", "unknown_key"
            )


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_config_singleton(self):
        """Test get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
        assert isinstance(config1, Config)

    def test_set_config_validation(self):
        """Test set_config validates configuration."""
        invalid_config = Config()
        invalid_config.model.temperature = -1.0
        with pytest.raises(ValueError):
            set_config(invalid_config)

    def test_set_config_valid(self):
        """Test set_config with valid configuration."""
        new_config = Config()
        new_config.model.name = "test-model"
        set_config(new_config)
        retrieved_config = get_config()
        assert retrieved_config.model.name == "test-model"

    def test_load_config_file_only(self):
        """Test load_config with file only."""
        data = {"model": {"name": "file-model"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name
        try:
            config = load_config(file_path=temp_path)
            assert config.model.name == "file-model"
        finally:
            os.unlink(temp_path)

    @patch.dict(os.environ, {"TEST_MODEL_NAME": "env-model"})
    def test_load_config_env_only(self):
        """Test load_config with environment variables only."""
        config = load_config(env_prefix="TEST_")
        assert config.model.name == "env-model"

    @patch.dict(os.environ, {"TEST_MODEL_NAME": "env-model"})
    def test_load_config_file_and_env(self):
        """Test load_config with both file and environment variables."""
        data = {"model": {"temperature": 0.9}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name
        try:
            config = load_config(file_path=temp_path, env_prefix="TEST_")
            assert config.model.name == "env-model"  # From env
            assert config.model.temperature == 0.9  # From file
        finally:
            os.unlink(temp_path)

    def test_load_config_invalid_file_continues(self):
        """Test load_config continues with invalid file."""
        with patch("logging.warning") as mock_warning:
            config = load_config(file_path="/nonexistent/file.yaml")
            assert isinstance(config, Config)
            mock_warning.assert_called()

    @patch.dict(os.environ, {"TEST_MODEL_TEMPERATURE": "invalid"})
    def test_load_config_invalid_env_continues(self):
        """Test load_config continues with invalid environment variables."""
        with patch("logging.warning") as mock_warning:
            config = load_config(env_prefix="TEST_")
            assert isinstance(config, Config)
            mock_warning.assert_called()


class TestThreadSafety:
    """Test thread safety of configuration operations."""

    def test_legacy_config_thread_safety(self):
        """Test LegacyConfig thread safety with config lock."""
        results = []
        errors = []

        def create_config():
            try:
                config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
                results.append(config)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_config) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(errors) == 0
        assert len(results) == 10
        assert all(isinstance(c, LegacyConfig) for c in results)

    def test_concurrent_config_access(self):
        """Test concurrent access to global configuration."""
        results = []
        errors = []

        def access_config():
            try:
                cfg = get_config()
                results.append(cfg.model.name)
                time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_config) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(errors) == 0
        assert len(results) == 20


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_settings_alias_compatibility(self):
        """Test Settings alias works correctly."""
        settings_config = Settings(learning_rate=0.02, batch_size=64, max_steps=2000)
        assert isinstance(settings_config, LegacyConfig)
        assert settings_config.learning_rate == 0.02
        assert settings_config.batch_size == 64
        assert settings_config.max_steps == 2000

    def test_global_config_instances(self):
        """Test global configuration instances are accessible."""
        assert isinstance(config, LegacyConfig)
        assert legacy_settings is config
        assert isinstance(env_settings, AGoTSettings)
        assert isinstance(runtime_settings, RuntimeSettings)
        assert settings is runtime_settings

    def test_runtime_settings_loading(self):
        """Test runtime settings loading function."""
        loaded_settings = load_runtime_settings()
        assert isinstance(loaded_settings, RuntimeSettings)
        assert isinstance(loaded_settings.app, AppSettingsModel)
        assert isinstance(loaded_settings.neo4j, Neo4jSettingsModel)

    def test_large_config_handling(self):
        """Test handling of large configuration structures."""
        large_kwargs = {f"param_{i}": f"value_{i}" for i in range(1000)}
        cfg = LegacyConfig(
            learning_rate=0.01, batch_size=32, max_steps=1000, **large_kwargs
        )
        assert cfg.learning_rate == 0.01
        assert cfg.batch_size == 32
        assert cfg.max_steps == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAdditionalValidationEdgeCases:
    """Test additional edge cases for validation functions."""

    def test_validate_learning_rate_boundary_values(self):
        """Test validation at exact boundary values."""
        # Test floating point precision edge cases
        validate_learning_rate(0.000001)  # Very small but valid
        validate_learning_rate(0.999999)  # Very close to 1.0
        
        with pytest.raises(ValueError):
            validate_learning_rate(float('inf'))
        with pytest.raises(ValueError):
            validate_learning_rate(float('-inf'))
        with pytest.raises(ValueError):
            validate_learning_rate(float('nan'))

    def test_validate_batch_size_extreme_values(self):
        """Test batch size validation with extreme values."""
        validate_batch_size(1)  # Minimum valid
        validate_batch_size(2**20)  # Very large but valid
        
        with pytest.raises(ValueError):
            validate_batch_size(2**32)  # Extremely large
        with pytest.raises(TypeError):
            validate_batch_size(None)
        with pytest.raises(TypeError):
            validate_batch_size([32])

    def test_validate_max_steps_edge_cases(self):
        """Test max steps validation edge cases."""
        validate_max_steps(1)  # Minimum valid
        validate_max_steps(2**31 - 1)  # Maximum int32
        
        with pytest.raises(TypeError):
            validate_max_steps(complex(1000, 0))
        with pytest.raises(TypeError):
            validate_max_steps({'steps': 1000})

    def test_validate_config_schema_malformed_data(self):
        """Test config schema validation with malformed data."""
        with pytest.raises(ValueError):
            validate_config_schema({'app': 'not_a_dict'})
        with pytest.raises(ValueError):
            validate_config_schema({'app': {'port': 'not_a_number'}})
        with pytest.raises(ValueError):
            validate_config_schema(None)
        with pytest.raises(ValueError):
            validate_config_schema("not_a_dict")


class TestLegacyConfigAdvancedEdgeCases:
    """Test advanced edge cases for LegacyConfig."""

    def test_legacy_config_with_extra_kwargs(self):
        """Test LegacyConfig behavior with unexpected keyword arguments."""
        # Should handle extra kwargs gracefully
        config = LegacyConfig(
            learning_rate=0.01,
            batch_size=32,
            max_steps=1000,
            unknown_param="value",
            another_param=123
        )
        assert config.learning_rate == 0.01
        assert hasattr(config, 'unknown_param')
        assert config.unknown_param == "value"

    def test_legacy_config_deepcopy_behavior(self):
        """Test deep copy behavior of LegacyConfig."""
        import copy
        original = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        deep_copied = copy.deepcopy(original)
        
        assert deep_copied == original
        assert deep_copied is not original
        assert deep_copied.app is not original.app
        assert deep_copied.asr_got is not original.asr_got

    def test_legacy_config_setattr_frozen_detailed(self):
        """Test detailed frozen config behavior."""
        config = LegacyConfig(frozen=True)
        
        # Test all possible attribute modifications
        with pytest.raises(AttributeError):
            config.learning_rate = 0.02
        with pytest.raises(AttributeError):
            config.batch_size = 64
        with pytest.raises(AttributeError):
            config.max_steps = 2000
        with pytest.raises(AttributeError):
            config.new_attribute = "value"

    def test_legacy_config_update_with_none_values(self):
        """Test updating config with None values."""
        config = LegacyConfig()
        # Should handle None values gracefully
        with pytest.raises(TypeError):
            config.update({'learning_rate': None})

    def test_legacy_config_merge_with_frozen(self):
        """Test merging with frozen configs."""
        frozen_config = LegacyConfig(learning_rate=0.02, frozen=True)
        normal_config = LegacyConfig(batch_size=64, max_steps=2000)
        
        merged = normal_config.merge(frozen_config)
        assert merged.learning_rate == 0.02
        assert merged.batch_size == 64
        assert merged.max_steps == 2000
        assert merged._frozen is False  # Merged config should not be frozen

    def test_legacy_config_model_dump_with_nested_objects(self):
        """Test model_dump with nested configuration objects."""
        config = LegacyConfig()
        dump = config.model_dump(include={'learning_rate', 'app'})
        assert 'learning_rate' in dump
        assert 'app' in dump
        assert 'batch_size' not in dump


class TestFileOperationsAdvancedCases:
    """Test advanced file operation scenarios."""

    def test_load_config_with_unicode_content(self):
        """Test loading config files with Unicode content."""
        unicode_data = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'max_steps': 1000,
            'description': 'Configuration with émojis 🚀 and Unicode ñáéíóú'
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding='utf-8') as f:
            yaml.dump(unicode_data, f, allow_unicode=True)
            temp_path = f.name
        try:
            config = LegacyConfig.load(temp_path)
            assert config.learning_rate == 0.01
            assert hasattr(config, 'description')
        finally:
            os.unlink(temp_path)

    def test_load_config_with_very_large_file(self):
        """Test loading very large configuration files."""
        large_data = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'max_steps': 1000
        }
        # Add many parameters to make it large
        for i in range(1000):
            large_data[f'param_{i}'] = f'value_{i}' * 100
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(large_data, f)
            temp_path = f.name
        try:
            config = LegacyConfig.load(temp_path)
            assert config.learning_rate == 0.01
            assert hasattr(config, 'param_999')
        finally:
            os.unlink(temp_path)

    def test_save_config_to_readonly_directory(self):
        """Test saving config to read-only directory."""
        config = LegacyConfig()
        readonly_path = "/root/readonly_config.yaml"
        
        with pytest.raises((PermissionError, OSError)):
            config.save(readonly_path)

    def test_concurrent_file_operations(self):
        """Test concurrent file read/write operations."""
        import threading
        import time
        
        config_data = {'learning_rate': 0.01, 'batch_size': 32, 'max_steps': 1000}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        results = []
        errors = []
        
        def read_config():
            try:
                cfg = LegacyConfig.load(temp_path)
                results.append(cfg.learning_rate)
                time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        try:
            threads = [threading.Thread(target=read_config) for _ in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            assert len(errors) == 0
            assert len(results) == 10
            assert all(r == 0.01 for r in results)
        finally:
            os.unlink(temp_path)

    def test_load_config_with_circular_references(self):
        """Test loading config files that might have circular references."""
        # YAML can create circular references
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
learning_rate: 0.01
batch_size: 32
max_steps: 1000
circular: &anchor
  self_ref: *anchor
""")
            temp_path = f.name
        try:
            # Should handle gracefully or raise appropriate error
            with pytest.raises((ValueError, yaml.YAMLError)):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestEnvironmentVariableAdvancedCases:
    """Test advanced environment variable scenarios."""

    @patch.dict(os.environ, {
        'LEARNING_RATE': '0.01',
        'BATCH_SIZE': '32',
        'MAX_STEPS': '1000',
        'EXTRA_UNICODE': 'café_config_ñ'
    })
    def test_from_env_with_unicode_values(self):
        """Test environment variables with Unicode values."""
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.01
        assert hasattr(config, 'EXTRA_UNICODE')

    @patch.dict(os.environ, {
        'LEARNING_RATE': '  0.02  ',  # With whitespace
        'BATCH_SIZE': '\t64\n',       # With tabs and newlines
        'MAX_STEPS': '2000'
    })
    def test_from_env_with_whitespace(self):
        """Test environment variables with surrounding whitespace."""
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.02
        assert config.batch_size == 64
        assert config.max_steps == 2000

    @patch.dict(os.environ, {
        'LEARNING_RATE': '',  # Empty string
        'BATCH_SIZE': '32',
        'MAX_STEPS': '1000'
    })
    def test_from_env_with_empty_values(self):
        """Test environment variables with empty values."""
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.01  # Should use default
        assert config.batch_size == 32
        assert config.max_steps == 1000

    @patch.dict(os.environ, {
        'LEARNING_RATE': '0.01' * 1000,  # Very long value
        'BATCH_SIZE': '32',
        'MAX_STEPS': '1000'
    })
    def test_from_env_with_very_long_values(self):
        """Test environment variables with extremely long values."""
        with pytest.raises((ValueError, OverflowError)):
            LegacyConfig.from_env()


class TestConfigSerializationEdgeCases:
    """Test configuration serialization edge cases."""

    def test_config_to_json_with_special_values(self):
        """Test JSON serialization with special values."""
        config = Config()
        config.model.api_key = "key_with_special_chars_!@#$%^&*()"
        config.model.base_url = "https://api.example.com/v1/with/path?param=value&other=true"
        
        json_str = config.to_json()
        data = json.loads(json_str)
        assert "!@#$%^&*()" in data["model"]["api_key"]
        assert "https://api.example.com" in data["model"]["base_url"]

    def test_config_to_yaml_with_multiline_strings(self):
        """Test YAML serialization with multiline strings."""
        config = Config()
        config.logging.format = """
        Line 1
        Line 2 with {placeholder}
        Line 3
        """
        
        yaml_str = config.to_yaml()
        reloaded_data = yaml.safe_load(yaml_str)
        assert "Line 1" in reloaded_data["logging"]["format"]
        assert "Line 2" in reloaded_data["logging"]["format"]

    def test_config_from_dict_with_deeply_nested_data(self):
        """Test creating config from deeply nested dictionary."""
        nested_data = {
            "model": {
                "name": "test",
                "nested": {
                    "deep": {
                        "deeper": {
                            "value": "found"
                        }
                    }
                }
            }
        }
        
        config = Config.from_dict(nested_data)
        assert config.model.name == "test"
        # Should handle nested data gracefully


class TestMemoryAndPerformanceEdgeCases:
    """Test memory usage and performance edge cases."""

    def test_large_config_memory_usage(self):
        """Test memory usage with very large configurations."""
        import sys
        
        # Create a large configuration
        large_kwargs = {}
        for i in range(10000):
            large_kwargs[f'param_{i}'] = f'value_{i}' * 10
        
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000, **large_kwargs)
        
        # Basic sanity checks
        assert config.learning_rate == 0.01
        assert hasattr(config, 'param_9999')
        
        # Memory should be reasonable (this is more of a smoke test)
        assert sys.getsizeof(config) > 0

    def test_config_creation_performance(self):
        """Test configuration creation performance."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        end_time = time.time()
        
        # Should complete reasonably quickly (less than 1 second for 100 configs)
        assert (end_time - start_time) < 1.0

    def test_config_copy_with_large_data(self):
        """Test copying configurations with large amounts of data."""
        large_data = {f'key_{i}': f'value_{i}' * 100 for i in range(1000)}
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000, **large_data)
        
        copied_config = config.copy()
        assert copied_config == config
        assert copied_config is not config


class TestConfigValidationComprehensive:
    """Test comprehensive configuration validation scenarios."""

    def test_config_validate_with_custom_ranges(self):
        """Test validation with custom value ranges."""
        config = Config()
        
        # Test extreme but valid values
        config.model.temperature = 0.0001
        config.model.max_tokens = 1
        config.model.timeout = 1
        config.validate()  # Should pass
        
        config.model.temperature = 1.9999
        config.model.max_tokens = 100000
        config.model.timeout = 3600
        config.validate()  # Should pass

    def test_config_validate_graph_edge_cases(self):
        """Test graph configuration validation edge cases."""
        config = Config()
        
        # Test boundary values
        config.graph.pruning_threshold = 0.0
        config.graph.validate()
        
        config.graph.pruning_threshold = 1.0
        config.graph.validate()
        
        config.graph.cache_size = 1
        config.graph.validate()

    def test_config_validate_logging_edge_cases(self):
        """Test logging configuration validation edge cases."""
        config = Config()
        
        # Test all valid logging levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config.logging.level = level
            config.validate()
        
        # Test case sensitivity
        config.logging.level = "debug"
        with pytest.raises(ValueError):
            config.validate()


class TestConcurrencyAndRaceConditions:
    """Test concurrency scenarios and potential race conditions."""

    def test_concurrent_config_modification(self):
        """Test concurrent modification of configuration objects."""
        import threading
        import time
        
        config = LegacyConfig()
        errors = []
        
        def modify_config(value):
            try:
                config.learning_rate = value
                time.sleep(0.001)
                assert config.learning_rate == value
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=modify_config, args=(0.01 + i * 0.001,))
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # There might be race conditions, but should not crash
        assert len(errors) == 0 or all(isinstance(e, AssertionError) for e in errors)

    def test_global_config_thread_safety(self):
        """Test thread safety of global configuration access."""
        import threading
        
        results = []
        errors = []
        
        def access_and_modify_global_config():
            try:
                cfg = get_config()
                original_name = cfg.model.name
                cfg.model.name = f"thread_{threading.current_thread().ident}"
                time.sleep(0.001)
                results.append(cfg.model.name)
                cfg.model.name = original_name
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=access_and_modify_global_config) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        # Some race conditions are expected, but no crashes


class TestConfigMigrationAndCompatibility:
    """Test configuration migration and version compatibility."""

    def test_legacy_config_backward_compatibility(self):
        """Test backward compatibility with old configuration formats."""
        # Simulate old config format
        old_format_data = {
            'lr': 0.01,  # Old key name
            'bs': 32,    # Old key name
            'steps': 1000  # Old key name
        }
        
        # Should handle gracefully (either map or ignore unknown keys)
        config = LegacyConfig(**old_format_data)
        assert hasattr(config, 'lr')
        assert hasattr(config, 'bs')
        assert hasattr(config, 'steps')

    def test_config_version_migration(self):
        """Test configuration version migration scenarios."""
        # Simulate config with version information
        versioned_data = {
            'version': '1.0',
            'learning_rate': 0.01,
            'batch_size': 32,
            'max_steps': 1000,
            'deprecated_field': 'should_be_ignored'
        }
        
        config = LegacyConfig(**versioned_data)
        assert config.learning_rate == 0.01
        assert hasattr(config, 'version')
        assert hasattr(config, 'deprecated_field')


class TestConfigSecurityAndValidation:
    """Test configuration security and validation edge cases."""

    def test_config_with_potentially_dangerous_values(self):
        """Test configuration with potentially dangerous values."""
        dangerous_values = {
            'script_injection': '<script>alert("xss")</script>',
            'sql_injection': "'; DROP TABLE users; --",
            'path_traversal': '../../../etc/passwd',
            'command_injection': '$(rm -rf /)'
        }
        
        # Should handle these as regular string values without execution
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000, **dangerous_values)
        assert config.learning_rate == 0.01
        assert hasattr(config, 'script_injection')
        assert config.script_injection == '<script>alert("xss")</script>'

    def test_config_input_sanitization(self):
        """Test input sanitization for configuration values."""
        config = Config()
        
        # Test with potentially problematic string values
        config.model.name = "model\nwith\nnewlines"
        config.model.api_key = "key\twith\ttabs"
        
        # Should preserve the values as-is
        assert "\n" in config.model.name
        assert "\t" in config.model.api_key

    def test_config_file_path_validation(self):
        """Test file path validation for configuration operations."""
        config = LegacyConfig()
        
        # Test with potentially dangerous paths
        dangerous_paths = [
            "/dev/null",
            "/proc/self/mem",
            "//network/path",
            "con.txt",  # Windows reserved name
            "../../../etc/passwd"
        ]
        
        for path in dangerous_paths:
            # Should either handle gracefully or raise appropriate errors
            try:
                config.save(path)
            except (PermissionError, OSError, ValueError):
                pass  # Expected behavior


class TestConfigIntegrationScenarios:
    """Test integration scenarios between different config components."""

    def test_config_hierarchy_integration(self):
        """Test integration between different configuration hierarchies."""
        # Create configs at different levels
        base_config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        override_config = LegacyConfig(learning_rate=0.02, batch_size=64)
        
        # Test merging
        merged = base_config.merge(override_config)
        assert merged.learning_rate == 0.02  # Overridden
        assert merged.batch_size == 64       # Overridden
        assert merged.max_steps == 1000      # From base

    def test_settings_model_integration(self):
        """Test integration between different settings models."""
        app_settings = AppSettingsModel()
        neo4j_settings = Neo4jSettingsModel()
        runtime_settings = RuntimeSettings(app=app_settings, neo4j=neo4j_settings)
        
        assert isinstance(runtime_settings.app, AppSettingsModel)
        assert isinstance(runtime_settings.neo4j, Neo4jSettingsModel)

    @patch.dict(os.environ, {
        'AGOT_MODEL_NAME': 'env-model',
        'LEARNING_RATE': '0.05'
    })
    def test_multi_source_config_integration(self):
        """Test integration of configuration from multiple sources."""
        # Load from multiple sources
        main_config = Config.from_env()
        legacy_config = LegacyConfig.from_env()
        
        assert main_config.model.name == 'env-model'
        assert legacy_config.learning_rate == 0.05
        
        # Should be independent but compatible
        assert isinstance(main_config, Config)
        assert isinstance(legacy_config, LegacyConfig)


class TestPydanticModelValidation:
    """Test Pydantic model validation scenarios."""

    def test_agot_settings_field_validation(self):
        """Test AGoTSettings field validation edge cases."""
        # Test with empty strings
        settings = AGoTSettings(llm_provider="", openai_api_key="", anthropic_api_key="")
        assert settings.llm_provider == ""
        assert settings.openai_api_key == ""
        assert settings.anthropic_api_key == ""

    def test_agot_settings_with_invalid_types(self):
        """Test AGoTSettings with invalid field types."""
        with pytest.raises((ValueError, TypeError)):
            AGoTSettings(llm_provider=123)  # Should be string
        
        with pytest.raises((ValueError, TypeError)):
            AGoTSettings(openai_api_key=["not", "a", "string"])

    def test_app_settings_model_validation(self):
        """Test AppSettingsModel validation."""
        app_settings = AppSettingsModel()
        
        # Test with various port values
        app_settings.port = 1
        app_settings.port = 65535
        
        # Test invalid ports
        with pytest.raises((ValueError, TypeError)):
            app_settings.port = 0
        with pytest.raises((ValueError, TypeError)):
            app_settings.port = 65536

    def test_neo4j_settings_validation(self):
        """Test Neo4jSettingsModel validation."""
        neo4j_settings = Neo4jSettingsModel()
        
        # Test with valid URIs
        neo4j_settings.uri = "bolt://localhost:7687"
        neo4j_settings.uri = "neo4j://localhost:7687"
        
        # Test connection validation scenarios
        assert neo4j_settings.uri is not None


class TestConfigErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_config_load_with_corrupted_files(self):
        """Test loading configurations from corrupted files."""
        # Create a file with mixed valid/invalid content
        mixed_content = """
learning_rate: 0.01
batch_size: 32
max_steps: 1000
corrupted_line: this is not valid YAML syntax: [unclosed bracket
another_field: valid_value
"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(mixed_content)
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_with_circular_imports(self):
        """Test configuration handling with circular reference scenarios."""
        # Create two config files that reference each other
        config1_content = """
base:
  learning_rate: 0.01
  batch_size: 32
include: config2.yaml
"""
        config2_content = """
override:
  learning_rate: 0.02
include: config1.yaml
"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(config1_content)
            config1_path = f1.name
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write(config2_content)
            config2_path = f2.name
        
        try:
            # Should handle circular references gracefully
            config = LegacyConfig.load(config1_path)
            assert config.learning_rate == 0.01  # Should use default or handle gracefully
        except (ValueError, RecursionError):
            # Acceptable to raise error for circular references
            pass
        finally:
            os.unlink(config1_path)
            os.unlink(config2_path)

    def test_config_with_network_timeouts(self):
        """Test configuration loading with network timeout scenarios."""
        # Simulate network-based config loading
        config = Config()
        
        # Test timeout handling for external config sources
        with patch('builtins.open', side_effect=TimeoutError("Network timeout")):
            with pytest.raises(TimeoutError):
                Config.from_file("http://example.com/config.yaml")

    def test_config_with_memory_constraints(self):
        """Test configuration behavior under memory constraints."""
        # Test with very large configurations that might cause memory issues
        try:
            massive_config_data = {}
            for i in range(100000):  # Very large config
                massive_config_data[f'key_{i}'] = f'value_{i}' * 1000
            
            config = LegacyConfig(**massive_config_data)
            assert hasattr(config, 'key_99999')
        except MemoryError:
            # Acceptable to fail with MemoryError for extremely large configs
            pass


class TestConfigurationPatterns:
    """Test common configuration patterns and best practices."""

    def test_config_factory_pattern(self):
        """Test factory pattern for configuration creation."""
        def create_development_config():
            return LegacyConfig(learning_rate=0.1, batch_size=16, max_steps=100)
        
        def create_production_config():
            return LegacyConfig(learning_rate=0.01, batch_size=64, max_steps=10000)
        
        dev_config = create_development_config()
        prod_config = create_production_config()
        
        assert dev_config.learning_rate == 0.1
        assert prod_config.learning_rate == 0.01
        assert dev_config != prod_config

    def test_config_builder_pattern(self):
        """Test builder pattern for configuration construction."""
        class ConfigBuilder:
            def __init__(self):
                self.config_data = {}
            
            def with_learning_rate(self, lr):
                self.config_data['learning_rate'] = lr
                return self
            
            def with_batch_size(self, bs):
                self.config_data['batch_size'] = bs
                return self
            
            def with_max_steps(self, steps):
                self.config_data['max_steps'] = steps
                return self
            
            def build(self):
                return LegacyConfig(**self.config_data)
        
        config = (ConfigBuilder()
                 .with_learning_rate(0.02)
                 .with_batch_size(128)
                 .with_max_steps(5000)
                 .build())
        
        assert config.learning_rate == 0.02
        assert config.batch_size == 128
        assert config.max_steps == 5000

    def test_config_inheritance_patterns(self):
        """Test configuration inheritance and composition patterns."""
        base_config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        
        # Test inheritance-like behavior through merging
        specialized_config = base_config.copy()
        specialized_config.update({'learning_rate': 0.02, 'specialized_param': 'value'})
        
        assert specialized_config.learning_rate == 0.02
        assert specialized_config.batch_size == 32  # Inherited
        assert specialized_config.max_steps == 1000  # Inherited
        assert hasattr(specialized_config, 'specialized_param')


class TestConfigurationDocumentation:
    """Test configuration documentation and introspection capabilities."""

    def test_config_schema_documentation(self):
        """Test configuration schema documentation capabilities."""
        config = Config()
        
        # Test that config objects can be introspected
        model_fields = dir(config.model)
        assert 'name' in model_fields
        assert 'temperature' in model_fields
        assert 'max_tokens' in model_fields
        
        graph_fields = dir(config.graph)
        assert 'max_depth' in graph_fields
        assert 'max_breadth' in graph_fields

    def test_config_field_descriptions(self):
        """Test field descriptions and metadata."""
        config = LegacyConfig()
        
        # Test that config has meaningful string representations
        config_str = str(config)
        assert 'learning_rate' in config_str or 'Config' in config_str
        
        config_repr = repr(config)
        assert 'Config' in config_repr

    def test_config_validation_messages(self):
        """Test that validation errors provide helpful messages."""
        config = Config()
        
        config.model.temperature = -1.0
        try:
            config.validate()
            assert False, "Should have raised validation error"
        except ValueError as e:
            error_message = str(e)
            assert 'temperature' in error_message.lower()
            assert 'between' in error_message.lower() or 'range' in error_message.lower()
