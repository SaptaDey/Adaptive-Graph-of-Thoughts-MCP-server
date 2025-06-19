import json
import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from threading import Thread
import time

# Import from the actual config module
from git.src.adaptive_graph_of_thoughts.config import (
    AGoTSettings,
    LegacyConfig,
    AppConfig,
    ASRGoTDefaultParams,
    PubMedConfig,
    GoogleScholarConfig,
    ExaSearchConfig,
    KnowledgeDomain,
    ModelConfig,
    GraphConfig,
    LoggingConfig,
    Config,
    Neo4jSettingsModel,
    AppSettingsModel,
    RuntimeSettings,
    validate_learning_rate,
    validate_batch_size,
    validate_max_steps,
    validate_config_schema,
    config,
    settings,
    runtime_settings,
    get_config,
    set_config,
    load_config,
    load_runtime_settings
)

@pytest.fixture
def sample_yaml_config():
    """Sample YAML configuration data for testing."""
    return {
        'learning_rate': 0.01,
        'batch_size': 32,
        'max_steps': 1000
    }

@pytest.fixture
def sample_json_config():
    """Sample JSON configuration data for testing."""
    return {
        'learning_rate': 0.05,
        'batch_size': 64,
        'max_steps': 2000
    }

@pytest.fixture
def temp_yaml_file(sample_yaml_config):
    """Create a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_yaml_config, f)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_json_file(sample_json_config):
    """Create a temporary JSON config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_json_config, f)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def clean_env():
    """Clean environment variables for testing."""
    original_env = os.environ.copy()
    # Remove any config-related env vars
    for key in list(os.environ.keys()):
        if any(prefix in key.upper() for prefix in ['LEARNING_RATE', 'BATCH_SIZE', 'MAX_STEPS', 'OPENAI', 'ANTHROPIC', 'NEO4J', 'AGOT']):
            del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(original_env)

class TestValidationFunctions:
    """Test cases for configuration validation functions."""
    
    def test_validate_learning_rate_valid_values(self):
        """Test learning rate validation with valid values."""
        validate_learning_rate(0.001)
        validate_learning_rate(0.01)
        validate_learning_rate(0.1)
        validate_learning_rate(1.0)
        validate_learning_rate(1)
    
    def test_validate_learning_rate_invalid_values(self):
        """Test learning rate validation with invalid values."""
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate(-0.1)
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate(0)
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate(1.1)
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate("0.1")
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            validate_learning_rate(None)
    
    def test_validate_batch_size_valid_values(self):
        """Test batch size validation with valid values."""
        validate_batch_size(1)
        validate_batch_size(32)
        validate_batch_size(128)
        validate_batch_size(1000)
    
    def test_validate_batch_size_invalid_values(self):
        """Test batch size validation with invalid values."""
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size(0)
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size(-1)
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size(32.5)
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size("32")
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_batch_size(None)
    
    def test_validate_max_steps_valid_values(self):
        """Test max steps validation with valid values."""
        validate_max_steps(1)
        validate_max_steps(100)
        validate_max_steps(10000)
    
    def test_validate_max_steps_invalid_values(self):
        """Test max steps validation with invalid values."""
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps(0)
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps(-1)
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps(100.5)
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            validate_max_steps("100")
    
    def test_validate_config_schema(self):
        """Test config schema validation."""
        assert validate_config_schema({}) is True
        assert validate_config_schema({"key": "value"}) is True

class TestAGoTSettings:
    """Test cases for AGoTSettings pydantic model."""
    
    def test_agot_settings_default_values(self, clean_env):
        settings = AGoTSettings()
        assert settings.llm_provider == "openai"
        assert settings.openai_api_key is None
        assert settings.anthropic_api_key is None
    
    @patch.dict(os.environ, {
        'LLM_PROVIDER': 'claude',
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    })
    def test_agot_settings_from_env(self):
        settings = AGoTSettings()
        assert settings.llm_provider == "claude"
        assert settings.openai_api_key == "test-openai-key"
        assert settings.anthropic_api_key == "test-anthropic-key"
    
    def test_agot_settings_explicit_values(self):
        settings = AGoTSettings(
            llm_provider="claude",
            openai_api_key="explicit-openai-key",
            anthropic_api_key="explicit-anthropic-key"
        )
        assert settings.llm_provider == "claude"
        assert settings.openai_api_key == "explicit-openai-key"
        assert settings.anthropic_api_key == "explicit-anthropic-key"

class TestLegacyConfig:
    """Test cases for LegacyConfig class."""
    
    def test_legacy_config_default_initialization(self):
        config = LegacyConfig()
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_steps == 1000
        assert config._frozen is False
        assert isinstance(config.app, AppConfig)
        assert isinstance(config.asr_got, ASRGoTDefaultParams)
    
    def test_legacy_config_custom_initialization(self):
        config = LegacyConfig(
            learning_rate=0.02,
            batch_size=64,
            max_steps=2000,
            frozen=True
        )
        assert config.learning_rate == 0.02
        assert config.batch_size == 64
        assert config.max_steps == 2000
        assert config._frozen is True
    
    def test_legacy_config_validation_on_init(self):
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            LegacyConfig(learning_rate=-0.1)
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            LegacyConfig(batch_size=0)
        with pytest.raises(ValueError, match="Max steps must be a positive integer"):
            LegacyConfig(max_steps=-1)
    
    def test_legacy_config_frozen_attribute_modification(self):
        config = LegacyConfig(frozen=True)
        with pytest.raises(AttributeError, match="Cannot modify frozen config"):
            config.learning_rate = 0.02
    
    def test_legacy_config_equality(self):
        config1 = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        config2 = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        config3 = LegacyConfig(learning_rate=0.02, batch_size=32, max_steps=1000)
        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"
    
    def test_legacy_config_repr(self):
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        expected = "Config(learning_rate=0.01, batch_size=32, max_steps=1000)"
        assert repr(config) == expected
    
    def test_legacy_config_model_dump(self):
        config = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        dump = config.model_dump()
        expected = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "max_steps": 1000
        }
        assert dump == expected
    
    def test_legacy_config_copy(self):
        original = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000, frozen=True)
        copy = original.copy()
        assert copy == original
        assert copy is not original
        assert copy._frozen is False
    
    def test_legacy_config_update(self):
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
        config = LegacyConfig()
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1.0"):
            config.update({"learning_rate": -0.1})
    
    def test_legacy_config_merge(self):
        config1 = LegacyConfig(learning_rate=0.01, batch_size=32, max_steps=1000)
        config2 = LegacyConfig(learning_rate=0.02, batch_size=64)
        merged = config1.merge(config2)
        assert merged.learning_rate == 0.02
        assert merged.batch_size == 64
        assert merged.max_steps == 1000

class TestLegacyConfigFileOperations:
    """Test cases for LegacyConfig file loading and saving."""
    
    def test_load_yaml_config(self, temp_yaml_file):
        config = LegacyConfig.load(temp_yaml_file)
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_steps == 1000
    
    def test_load_json_config(self, temp_json_file):
        config = LegacyConfig.load(temp_json_file)
        assert config.learning_rate == 0.05
        assert config.batch_size == 64
        assert config.max_steps == 2000
    
    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            LegacyConfig.load("/nonexistent/config.yaml")
    
    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
        try:
            with pytest.raises(ValueError, match="Empty configuration file"):
                LegacyConfig.load(f.name)
        finally:
            os.unlink(f.name)
    
    def test_load_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
        try:
            with pytest.raises(yaml.YAMLError, match="Invalid YAML"):
                LegacyConfig.load(f.name)
        finally:
            os.unlink(f.name)
    
    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')
        try:
            with pytest.raises(json.JSONDecodeError):
                LegacyConfig.load(f.name)
        finally:
            os.unlink(f.name)
    
    def test_load_unsupported_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                LegacyConfig.load(f.name)
        finally:
            os.unlink(f.name)
    
    def test_load_missing_required_keys(self):
        incomplete_config = {"max_steps": 1000}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_config, f)
        try:
            with pytest.raises(ValueError, match="Missing required key"):
                LegacyConfig.load(f.name)
        finally:
            os.unlink(f.name)
    
    def test_load_invalid_data_types(self):
        invalid_config = {
            "learning_rate": "not_a_number",
            "batch_size": 32,
            "max_steps": 1000
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
        try:
            with pytest.raises(ValueError, match="learning_rate must be a number"):
                LegacyConfig.load(f.name)
        finally:
            os.unlink(f.name)
    
    def test_save_yaml_config(self):
        config = LegacyConfig(learning_rate=0.02, batch_size=64, max_steps=2000)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save(f.name)
        try:
            loaded = LegacyConfig.load(f.name)
            assert loaded == config
        finally:
            os.unlink(f.name)
    
    def test_save_json_config(self):
        config = LegacyConfig(learning_rate=0.03, batch_size=128, max_steps=3000)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
        try:
            loaded = LegacyConfig.load(f.name)
            assert loaded == config
        finally:
            os.unlink(f.name)
    
    def test_save_unsupported_format(self):
        config = LegacyConfig()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            try:
                with pytest.raises(ValueError, match="Unsupported file format"):
                    config.save(f.name)
            finally:
                os.unlink(f.name)
    
    def test_save_permission_denied(self):
        config = LegacyConfig()
        with pytest.raises(PermissionError, match="Permission denied writing to"):
            config.save("/root/protected_config.yaml")

class TestLegacyConfigEnvironment:
    """Test cases for LegacyConfig environment variable loading."""
    
    @patch.dict(os.environ, {
        'LEARNING_RATE': '0.02',
        'BATCH_SIZE': '64',
        'MAX_STEPS': '2000'
    })
    def test_from_env_all_variables(self):
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.02
        assert config.batch_size == 64
        assert config.max_steps == 2000
    
    @patch.dict(os.environ, {
        'LEARNING_RATE': '0.03'
    })
    def test_from_env_partial_variables(self):
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.03
        assert config.batch_size == 32
        assert config.max_steps == 1000
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_no_variables(self):
        config = LegacyConfig.from_env()
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_steps == 1000
    
    def test_load_with_overrides(self, temp_yaml_file):
        override = {"learning_rate": 0.05, "batch_size": 128}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(override, f)
        try:
            config = LegacyConfig.load_with_overrides(temp_yaml_file, f.name)
            assert config.learning_rate == 0.05
            assert config.batch_size == 128
            assert config.max_steps == 1000
        finally:
            os.unlink(f.name)
    
    def test_load_with_overrides_nonexistent_override(self, temp_yaml_file):
        with pytest.raises(FileNotFoundError, match="Override file not found"):
            LegacyConfig.load_with_overrides(temp_yaml_file, "/nonexistent/override.json")
    
    def test_load_with_overrides_empty_override(self, temp_yaml_file):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
        try:
            config = LegacyConfig.load_with_overrides(temp_yaml_file, f.name)
            base = LegacyConfig.load(temp_yaml_file)
            assert config == base
        finally:
            os.unlink(f.name)

class TestSimpleConfigClasses:
    """Test cases for simple configuration classes."""
    
    def test_app_config_initialization(self):
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
        config = AppConfig(
            name="Custom App",
            version="1.0.0",
            host="127.0.0.1",
            port=9000,
            reload=False,
            log_level="DEBUG",
            cors_allowed_origins_str="http://localhost:3000",
            auth_token="test-token"
        )
        assert config.name == "Custom App"
        assert config.version == "1.0.0"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.reload is False
        assert config.log_level == "DEBUG"
        assert config.cors_allowed_origins_str == "http://localhost:3000"
        assert config.auth_token == "test-token"
    
    def test_asr_got_default_params(self):
        params = ASRGoTDefaultParams()
        assert params.initial_confidence == 0.8
        assert params.confidence_threshold == 0.75
        assert params.max_iterations == 10
        assert params.convergence_threshold == 0.05
    
    def test_pubmed_config(self):
        config = PubMedConfig()
        assert config.api_key is None
        assert config.max_results == 20
        assert config.rate_limit_delay == 0.5
        custom = PubMedConfig(api_key="test-key", max_results=50, rate_limit_delay=1.0)
        assert custom.api_key == "test-key"
        assert custom.max_results == 50
        assert custom.rate_limit_delay == 1.0
    
    def test_google_scholar_config(self):
        config = GoogleScholarConfig()
        assert config.max_results == 10
        assert config.rate_limit_delay == 1.0
    
    def test_exa_search_config(self):
        config = ExaSearchConfig()
        assert config.api_key is None
        assert config.max_results == 10
    
    def test_knowledge_domain(self):
        domain = KnowledgeDomain("AI")
        assert domain.name == "AI"
        assert domain.description == ""
        assert domain.keywords == []
        custom = KnowledgeDomain(
            "Machine Learning",
            description="ML techniques and algorithms",
            keywords=["neural networks", "deep learning", "algorithms"]
        )
        assert custom.name == "Machine Learning"
        assert custom.description == "ML techniques and algorithms"
        assert custom.keywords == ["neural networks", "deep learning", "algorithms"]

class TestDataclassConfigSystem:
    """Test cases for the dataclass-based configuration system."""
    
    def test_model_config_defaults(self):
        config = ModelConfig()
        assert config.name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 30
        assert config.api_key is None
        assert config.base_url is None
    
    def test_model_config_custom_values(self):
        config = ModelConfig(
            name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1024,
            timeout=60,
            api_key="test-key",
            base_url="https://api.custom.com"
        )
        assert config.name == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.timeout == 60
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.custom.com"
    
    def test_graph_config_defaults(self):
        config = GraphConfig()
        assert config.max_depth == 5
        assert config.max_breadth == 3
        assert config.pruning_threshold == 0.1
        assert config.enable_caching is True
        assert config.cache_size == 1000
    
    def test_logging_config_defaults(self):
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        assert config.file_path is None
        assert config.enable_console is True
    
    def test_config_initialization(self):
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_config_from_dict(self):
        data = {
            "model": {"name": "custom-model", "temperature": 0.8},
            "graph": {"max_depth": 10},
            "logging": {"level": "DEBUG"}
        }
        config = Config.from_dict(data)
        assert config.model.name == "custom-model"
        assert config.model.temperature == 0.8
        assert config.graph.max_depth == 10
        assert config.logging.level == "DEBUG"
    
    def test_config_to_dict(self):
        config = Config()
        data = config.to_dict()
        assert "model" in data
        assert "graph" in data
        assert "logging" in data
        assert isinstance(data["model"], dict)
        assert isinstance(data["graph"], dict)
        assert isinstance(data["logging"], dict)
    
    def test_config_to_json(self):
        config = Config()
        json_str = config.to_json()
        parsed = json.loads(json_str)
        assert "model" in parsed
        assert "graph" in parsed
        assert "logging" in parsed
    
    def test_config_to_yaml(self):
        config = Config()
        yaml_str = config.to_yaml()
        parsed = yaml.safe_load(yaml_str)
        assert "model" in parsed
        assert "graph" in parsed
        assert "logging" in parsed

class TestDataclassConfigFileOperations:
    """Test cases for dataclass Config file operations."""
    
    def test_config_from_file_yaml(self):
        config_data = {
            "model": {"name": "test-model", "temperature": 0.9},
            "graph": {"max_depth": 8},
            "logging": {"level": "WARNING"}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
        try:
            config = Config.from_file(f.name)
            assert config.model.name == "test-model"
            assert config.model.temperature == 0.9
            assert config.graph.max_depth == 8
            assert config.logging.level == "WARNING"
        finally:
            os.unlink(f.name)
    
    def test_config_from_file_json(self):
        config_data = {
            "model": {"name": "test-model", "max_tokens": 512},
            "graph": {"enable_caching": False},
            "logging": {"enable_console": False}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
        try:
            config = Config.from_file(f.name)
            assert config.model.name == "test-model"
            assert config.model.max_tokens == 512
            assert config.graph.enable_caching is False
            assert config.logging.enable_console is False
        finally:
            os.unlink(f.name)
    
    def test_config_from_file_nonexistent(self):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config.from_file("/nonexistent/config.yaml")
    
    def test_config_from_file_empty(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
        try:
            with pytest.raises(ValueError, match="Failed to parse configuration file"):
                Config.from_file(f.name)
        finally:
            os.unlink(f.name)
    
    def test_config_from_file_invalid_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                Config.from_file(f.name)
        finally:
            os.unlink(f.name)
    
    def test_config_save_to_file_yaml(self):
        config = Config()
        config.model.name = "saved-model"
        config.graph.max_depth = 7
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_to_file(f.name)
        try:
            loaded = Config.from_file(f.name)
            assert loaded.model.name == "saved-model"
            assert loaded.graph.max_depth == 7
        finally:
            os.unlink(f.name)
    
    def test_config_save_to_file_json(self):
        config = Config()
        config.model.temperature = 0.3
        config.logging.level = "ERROR"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_to_file(f.name)
        try:
            loaded = Config.from_file(f.name)
            assert loaded.model.temperature == 0.3
            assert loaded.logging.level == "ERROR"
        finally:
            os.unlink(f.name)
    
    @patch.dict(os.environ, {
        'AGOT_MODEL_NAME': 'env-model',
        'AGOT_MODEL_TEMPERATURE': '0.6',
        'AGOT_MODEL_MAX_TOKENS': '1024',
        'AGOT_GRAPH_MAX_DEPTH': '12',
        'AGOT_LOGGING_LEVEL': 'DEBUG'
    })
    def test_config_from_env(self):
        config = Config.from_env()
        assert config.model.name == "env-model"
        assert config.model.temperature == 0.6
        assert config.model.max_tokens == 1024
        assert config.graph.max_depth == 12
        assert config.logging.level == "DEBUG"
    
    @patch.dict(os.environ, {
        'AGOT_MODEL_TEMPERATURE': 'invalid'
    })
    def test_config_from_env_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid value for"):
            Config.from_env()
    
    def test_config_validation_valid(self):
        config = Config()
        config.validate()
    
    def test_config_validation_invalid_temperature(self):
        config = Config()
        config.model.temperature = 3.0
        with pytest.raises(ValueError, match="Model temperature must be between 0.0 and 2.0"):
            config.validate()
    
    def test_config_validation_invalid_max_tokens(self):
        config = Config()
        config.model.max_tokens = -1
        with pytest.raises(ValueError, match="Model max_tokens must be positive"):
            config.validate()
    
    def test_config_validation_invalid_logging_level(self):
        config = Config()
        config.logging.level = "INVALID"
        with pytest.raises(ValueError, match="Logging level must be one of"):
            config.validate()
    
    def test_config_update(self):
        config = Config()
        config.update(
            model={"name": "updated-model", "temperature": 0.4},
            graph={"max_depth": 15},
            logging={"level": "WARNING"}
        )
        assert config.model.name == "updated-model"
        assert config.model.temperature == 0.4
        assert config.graph.max_depth == 15
        assert config.logging.level == "WARNING"

class TestGlobalConfigManagement:
    """Test cases for global configuration management."""
    
    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
    
    def test_set_config(self):
        original = get_config()
        new = Config()
        new.model.name = "test-global-model"
        set_config(new)
        retrieved = get_config()
        assert retrieved is new
        assert retrieved.model.name == "test-global-model"
        set_config(original)
    
    def test_set_config_validates(self):
        invalid = Config()
        invalid.model.temperature = 5.0
        with pytest.raises(ValueError, match="Model temperature must be between 0.0 and 2.0"):
            set_config(invalid)
    
    def test_load_config_with_file(self):
        data = {"model": {"name": "loaded-model"}, "graph": {"max_depth": 20}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
        try:
            cfg = load_config(file_path=f.name)
            assert cfg.model.name == "loaded-model"
            assert cfg.graph.max_depth == 20
        finally:
            os.unlink(f.name)
    
    @patch.dict(os.environ, {
        'AGOT_MODEL_NAME': 'env-loaded-model',
        'AGOT_GRAPH_MAX_BREADTH': '5'
    })
    def test_load_config_with_env_override(self):
        data = {
            "model": {"name": "file-model", "temperature": 0.7},
            "graph": {"max_depth": 3, "max_breadth": 2}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
        try:
            cfg = load_config(file_path=f.name)
            assert cfg.model.name == "env-loaded-model"
            assert cfg.model.temperature == 0.7
            assert cfg.graph.max_breadth == 5
            assert cfg.graph.max_depth == 3
        finally:
            os.unlink(f.name)

class TestThreadSafety:
    """Test cases for thread safety of configuration operations."""
    
    def test_legacy_config_thread_safety(self):
        results = []
        errors = []
        def create_config(tid):
            try:
                cfg = LegacyConfig(
                    learning_rate=0.01 + tid*0.001,
                    batch_size=32 + tid,
                    max_steps=1000 + tid*10
                )
                results.append((tid, cfg))
            except Exception as e:
                errors.append((tid, e))
        threads = [Thread(target=create_config, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert len(results) == 10
        for tid, cfg in results:
            assert cfg.learning_rate == pytest.approx(0.01 + tid*0.001)
            assert cfg.batch_size == 32 + tid
            assert cfg.max_steps == 1000 + tid*10
    
    def test_global_config_concurrent_access(self):
        results = []
        def access_global(tid):
            cfg = get_config()
            time.sleep(0.001)
            results.append((tid, id(cfg)))
        threads = [Thread(target=access_global, args=(i,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        ids = {cid for _, cid in results}
        assert len(ids) == 1
        assert len(results) == 5

class TestEdgeCasesAndErrorConditions:
    """Test cases for edge cases and error conditions."""
    
    def test_config_with_none_values(self):
        cfg = ModelConfig(api_key=None, base_url=None)
        assert cfg.api_key is None
        assert cfg.base_url is None
    
    def test_config_with_extreme_values(self):
        cfg1 = ModelConfig(temperature=0.0, max_tokens=1, timeout=1)
        assert cfg1.temperature == 0.0
        assert cfg1.max_tokens == 1
        assert cfg1.timeout == 1
        cfg2 = ModelConfig(temperature=2.0, max_tokens=100000, timeout=3600)
        assert cfg2.temperature == 2.0
        assert cfg2.max_tokens == 100000
        assert cfg2.timeout == 3600
    
    def test_config_memory_efficiency(self):
        configs = [Config() for _ in range(1000)]
        configs[0].model.name = "model-0"
        configs[-1].model.name = "model-999"
        assert len(configs) == 1000
        assert configs[0].model.name == "model-0"
        assert configs[-1].model.name == "model-999"
    
    def test_config_deep_copy_independence(self):
        original = LegacyConfig(learning_rate=0.01, batch_size=32)
        copy_cfg = original.copy()
        copy_cfg.learning_rate = 0.02
        copy_cfg.batch_size = 64
        assert original.learning_rate == 0.01
        assert original.batch_size == 32
        assert copy_cfg.learning_rate == 0.02
        assert copy_cfg.batch_size == 64