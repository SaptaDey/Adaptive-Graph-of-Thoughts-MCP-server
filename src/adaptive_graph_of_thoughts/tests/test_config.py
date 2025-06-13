import pytest
import yaml

from adaptive_graph_of_thoughts.config import Config

def test_config_load_from_yaml(tmp_path):
    config_data = {"learning_rate": 0.01, "batch_size": 32, "max_steps": 1000}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    config = Config.load(str(config_file))
    assert config.learning_rate == pytest.approx(0.01)
    assert config.batch_size == 32
    assert config.max_steps == 1000

def test_config_load_with_defaults(tmp_path):
    """
    Tests that loading a configuration YAML file missing optional keys assigns default values.
    
    Verifies that when the 'max_steps' key is absent, the loaded Config object includes 'max_steps' with an integer default value.
    """
    config_data = {"learning_rate": 0.01, "batch_size": 32}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    config = Config.load(str(config_file))
    # verify default for max_steps is set
    assert hasattr(config, "max_steps")
    assert isinstance(config.max_steps, int)

@pytest.mark.parametrize("missing_key", ["learning_rate", "batch_size"])
def test_config_missing_required_key_raises(tmp_path, missing_key):
    """
    Tests that loading a configuration file missing a required key raises a ValueError.
    
    Args:
        tmp_path: Temporary directory path fixture for file operations.
        missing_key: The required configuration key to omit from the YAML file.
    """
    full_data = {"learning_rate": 0.01, "batch_size": 32}
    data = {k: v for k, v in full_data.items() if k != missing_key}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(data))
    with pytest.raises(ValueError):
        Config.load(str(config_file))

@pytest.mark.parametrize("bad_data", [
    {"learning_rate": "fast", "batch_size": 32},
    {"learning_rate": 0.01, "batch_size": "large"},
    {"learning_rate": 0.01, "batch_size": 32, "max_steps": "many"},
])
def test_config_invalid_type_raises(tmp_path, bad_data):
    """
    Tests that loading a configuration file with invalid data types raises a ValueError.
    
    Writes a YAML file with incorrect types for configuration fields, then asserts that
    Config.load() raises a ValueError when attempting to load the file.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(bad_data))
    with pytest.raises(ValueError):
        Config.load(str(config_file))