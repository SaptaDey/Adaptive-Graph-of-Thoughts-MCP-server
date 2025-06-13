import pytest
import yaml

from adaptive_graph_of_thoughts.config import Config

def test_config_load_from_yaml(tmp_path):
    """
    Tests that the Config class correctly loads all configuration values from a YAML file.
    
    Creates a temporary YAML file with required configuration keys, loads it using Config.load(),
    and asserts that the loaded values match the expected data.
    """
    config_data = {"learning_rate": 0.01, "batch_size": 32, "max_steps": 1000}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    config = Config.load(str(config_file))
    assert config.learning_rate == pytest.approx(0.01)
    assert config.batch_size == 32
    assert config.max_steps == 1000

def test_config_load_with_defaults(tmp_path):
    """
    Tests that loading a config file missing optional keys assigns default values.
    
    Verifies that when the 'max_steps' key is absent from the YAML config, the loaded
    Config object includes 'max_steps' with a default integer value.
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
    Tests that loading a config file missing a required key raises a ValueError.
    
    Removes one required key from the configuration data, writes it to a YAML file, and asserts that loading the file with Config.load() raises a ValueError.
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
    Tests that loading a config file with invalid data types raises a ValueError.
    
    This test writes a YAML configuration with incorrect types for required keys and asserts
    that attempting to load it using Config.load() results in a ValueError.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(bad_data))
    with pytest.raises(ValueError):
        Config.load(str(config_file))