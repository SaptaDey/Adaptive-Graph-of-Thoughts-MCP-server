import copy
import stat
from pathlib import Path

import pytest
import yaml

from adaptive_graph_of_thoughts.config import validate_config_schema

BASE_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


@pytest.fixture
def base_config_dict() -> dict:
    with open(BASE_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def valid_config(tmp_path, base_config_dict):
    file_path = tmp_path / "settings.yaml"
    file_path.write_text(yaml.safe_dump(base_config_dict))
    return file_path, copy.deepcopy(base_config_dict)


def test_valid_config(valid_config):
    path, expected_config_data = valid_config
    # Load YAML data first
    with open(path) as f:
        config_data_to_validate = yaml.safe_load(f)

    assert validate_config_schema(config_data_to_validate) is True
    # The function validate_config_schema doesn't return the dict,
    # so we compare the originally loaded dict if needed for other assertions.
    assert config_data_to_validate == expected_config_data


@pytest.fixture
def minimal_config(tmp_path, base_config_dict):
    cfg = copy.deepcopy(base_config_dict)
    cfg.pop("google_scholar", None)
    cfg.pop("pubmed", None)
    cfg.pop("exa_search", None)
    cfg.pop("knowledge_domains", None)
    file_path = tmp_path / "config_min.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    return file_path, cfg


def test_minimal_config(minimal_config):
    path, expected_config_data = minimal_config
    # Load YAML data first
    with open(path) as f:
        config_data_to_validate = yaml.safe_load(f)

    assert validate_config_schema(config_data_to_validate) is True
    assert "google_scholar" not in config_data_to_validate


@pytest.mark.parametrize(
    "missing_key",
    ["app", "asr_got", "mcp_settings"],
)
def test_missing_keys(tmp_path, base_config_dict, missing_key):
    cfg = copy.deepcopy(base_config_dict)
    cfg.pop(missing_key)
    file_path = tmp_path / "config_missing.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    with open(file_path) as f:
        config_data_to_validate = yaml.safe_load(f)
    with pytest.raises(ValueError) as exc:
        validate_config_schema(config_data_to_validate)
    assert missing_key in str(exc.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("app.port", "not_an_int"),
        ("mcp_settings.protocol_version", 123),
    ],
)
def test_wrong_types(tmp_path, base_config_dict, field, value):
    cfg = copy.deepcopy(base_config_dict)
    section, key = field.split(".")
    cfg[section][key] = value
    file_path = tmp_path / "config_bad_type.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    with open(file_path) as f:
        config_data_to_validate = yaml.safe_load(f)
    with pytest.raises(ValueError) as exc:
        validate_config_schema(config_data_to_validate)
    assert key in str(exc.value)


def test_empty_file(tmp_path):
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    # yaml.safe_load on an empty file returns None
    config_data_to_validate = None
    with pytest.raises(ValueError):  # Changed from ConfigValidationError
        # jsonschema.validate(None, schema) will raise an error.
        validate_config_schema(config_data_to_validate)


def test_unreadable_file(tmp_path):
    file_path = tmp_path / "config_unreadable.yaml"
    file_path.write_text("host: localhost\nport: 8080")
    # remove read permissions
    file_path.chmod(0)
    try:
        # Reading the file will cause PermissionError before validation
        with pytest.raises(PermissionError), open(file_path) as f:
            yaml.safe_load(f)
        # If we could read it, then validate_config_schema would be called.
        # This part of the test might be redundant if open() already fails.
        # For the sake of argument, if it didn't fail:
        # config_data_to_validate = {"host": "localhost", "port": 8080} # dummy data
        # with pytest.raises(ValueError): # Changed from ConfigValidationError
        #     validate_config_schema(config_data_to_validate) # This line might not be reached
    finally:
        # restore so pytest can clean up
        file_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.parametrize(
    "port",
    [1, 65535],
)
def test_boundary_values(tmp_path, base_config_dict, port):
    cfg = copy.deepcopy(base_config_dict)
    cfg["app"]["port"] = port
    file_path = tmp_path / f"config_port_{port}.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    with open(file_path) as f:
        config_data_to_validate = yaml.safe_load(f)
    assert validate_config_schema(config_data_to_validate) is True
    assert config_data_to_validate["app"]["port"] == port
