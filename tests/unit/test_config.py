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
    return {
        "learning_rate": 0.01,
        "batch_size": 32,
        "max_steps": 1000
    }

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
    return {
        "learning_rate": 0.03,
        "batch_size": 128,
        "max_steps": 3000
    }

@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            "learning_rate": 0.01,
            "batch_size": 32,
            "max_steps": 1000
        }, f)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def temp_json_file():
    """Create a temporary JSON configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "learning_rate": 0.01,
            "batch_size": 32,
            "max_steps": 1000
        }, f)
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
        """Test invalid learning rates raise ValueError."""
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate(0)
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate(-0.1)
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate(1.1)
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate("0.1")
    
    def test_validate_batch_size_valid(self):
        """Test valid batch sizes pass validation."""
        validate_batch_size(1)
        validate_batch_size(32)
        validate_batch_size(1000)
    
    def test_validate_batch_size_invalid(self):
        """Test invalid batch sizes raise ValueError."""
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size(0)
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size(-1)
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size(1.5)
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size("32")
    
    def test_validate_max_steps_valid(self):
        """Test valid max steps pass validation."""
        validate_max_steps(1)
        validate_max_steps(1000)
        validate_max_steps(10000)
    
    def test_validate_max_steps_invalid(self):
        """Test invalid max steps raise ValueError."""
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps(0)
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps(-1)
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps(1.5)
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps("1000")
    
    def test_validate_config_schema(self):
        """Test config schema validation."""
        assert validate_config_schema({"key": "value"}) is True
        assert validate_config_schema({}) is True

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
    
    @patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key"
    })
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
            auth_token="secret"
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
            convergence_threshold=0.01
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
            keywords=["machine learning", "neural networks"]
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
        expected = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "max_steps": 1000
        }
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
        updates = {
            "learning_rate": 0.02,
            "batch_size": 64,
            "max_steps": 2000
        }
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Empty configuration file"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises YAMLError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            temp_path = f.name
        try:
            with pytest.raises(yaml.YAMLError):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')
            temp_path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_missing_required_keys(self):
        """Test loading config with missing required keys raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"max_steps": 1000}, f)  # Missing learning_rate and batch_size
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Missing required key"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_invalid_data_types(self):
        """Test loading config with invalid data types raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "learning_rate": "not_a_number",
                "batch_size": 32,
                "max_steps": 1000
            }, f)
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="learning_rate must be a number"):
                LegacyConfig.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_yaml_file(self):
        """Test saving configuration to YAML file."""
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
    
    @patch.dict(os.environ, {
        "LEARNING_RATE": "0.02",
        "BATCH_SIZE": "64",
        "MAX_STEPS": "2000"
    })
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "learning_rate": 0.05,
                "batch_size": 128
            }, f)
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
            base_url="https://api.example.com"
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
            "model": {
                "name": "custom-model",
                "temperature": 0.8
            },
            "graph": {
                "max_depth": 10
            },
            "logging": {
                "level": "DEBUG"
            }
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
            "logging": {"level": "DEBUG"}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
            "logging": {"level": "DEBUG"}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Failed to parse configuration file"):
                Config.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_file_unsupported_format(self):
        """Test Config.from_file with unsupported format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
    
    @patch.dict(os.environ, {
        "AGOT_MODEL_NAME": "env-model",
        "AGOT_MODEL_TEMPERATURE": "0.8",
        "AGOT_GRAPH_MAX_DEPTH": "6",
        "AGOT_LOGGING_LEVEL": "DEBUG"
    })
    def test_from_env_with_defaults(self):
        """Test Config.from_env with default prefix."""
        config = Config.from_env()
        assert config.model.name == "env-model"
        assert config.model.temperature == 0.8
        assert config.graph.max_depth == 6
        assert config.logging.level == "DEBUG"
    
    @patch.dict(os.environ, {
        "CUSTOM_MODEL_NAME": "custom-model",
        "CUSTOM_MODEL_TEMPERATURE": "0.9"
    })
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
        with pytest.raises(ValueError, match="Model temperature must be between 0.0 and 2.0"):
            config.validate()
        config.model.temperature = 2.1
        with pytest.raises(ValueError, match="Model temperature must be between 0.0 and 2.0"):
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
        with pytest.raises(ValueError, match="Graph pruning_threshold must be between 0.0 and 1.0"):
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
            logging={"level": "DEBUG"}
        )
        assert config.model.name == "updated-model"
        assert config.graph.max_depth == 8
        assert config.logging.level == "DEBUG"
    
    def test_update_unknown_section(self):
        """Test updating unknown section logs warning."""
        config = Config()
        with patch('logging.warning') as mock_warning:
            config.update(unknown_section={"key": "value"})
            mock_warning.assert_called_with("Unknown config key: %s", "unknown_section")
    
    def test_update_unknown_nested_key(self):
        """Test updating unknown nested key logs warning."""
        config = Config()
        with patch('logging.warning') as mock_warning:
            config.update(model={"unknown_key": "value"})
            mock_warning.assert_called_with("Unknown nested config key: %s.%s", "model", "unknown_key")

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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
        with patch('logging.warning') as mock_warning:
            config = load_config(file_path="/nonexistent/file.yaml")
            assert isinstance(config, Config)
            mock_warning.assert_called()
    
    @patch.dict(os.environ, {"TEST_MODEL_TEMPERATURE": "invalid"})
    def test_load_config_invalid_env_continues(self):
        """Test load_config continues with invalid environment variables."""
        with patch('logging.warning') as mock_warning:
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
        cfg = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000, **large_kwargs)
        assert cfg.learning_rate == 0.01
        assert cfg.batch_size == 32
        assert cfg.max_steps == 1000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])