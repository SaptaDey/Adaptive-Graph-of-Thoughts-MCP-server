# Tests written with pytest
import pytest

from adaptive_graph_of_thoughts.config import validate_config, ConfigValidationError

@pytest.mark.parametrize("cfg", [
    {"model": "gpt-4", "temperature": 0.7},
    {"model": "gpt-3.5", "temperature": 0.0},
])
def test_validate_config_happy(cfg):
    """Happy path: valid configurations should pass without error."""
    assert validate_config(cfg) is None

def test_validate_config_missing_model():
    """Missing 'model' key should raise ConfigValidationError."""
    cfg = {"temperature": 0.5}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)

def test_validate_config_missing_temperature():
    """Missing 'temperature' key should raise ConfigValidationError."""
    cfg = {"model": "gpt-3.5"}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)

def test_validate_config_non_float_temperature():
    """Non‐float 'temperature' type should raise ConfigValidationError."""
    cfg = {"model": "gpt-3.5", "temperature": "hot"}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)

@pytest.mark.parametrize("temperature", [-0.1, 1.1])
def test_validate_config_temperature_out_of_range(temperature):
    """Temperature value out of range (<0 or >1) should raise ConfigValidationError."""
    cfg = {"model": "gpt-3.5", "temperature": temperature}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)

def test_validate_config_empty_dict():
    """Empty configuration should raise ConfigValidationError."""
    cfg = {}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)

def test_validate_config_extra_keys():
    """Configuration with unexpected additional keys should raise ConfigValidationError."""
    cfg = {"model": "gpt-4", "temperature": 0.5, "unexpected": True}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)

def test_validate_config_extremely_large_temperature():
    """An extremely large temperature value should raise ConfigValidationError."""
    cfg = {"model": "gpt-3.5", "temperature": 1e6}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)