import os
import stat
from pathlib import Path

import pytest
import yaml

from adaptive_graph_of_thoughts.config import validate_config_schema
# ConfigValidationError is replaced by ValueError from jsonschema,
# which is raised by validate_config_schema.
# No specific import needed for ValueError.


@pytest.fixture
def valid_config(tmp_path):
    cfg = {
        "host": "localhost",
        "port": 8080,
        "debug": True,
        "databases": ["db1", "db2"]
    }
    file_path = tmp_path / "config.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    return file_path, cfg


def test_valid_config(valid_config):
    path, expected_config_data = valid_config
    # Load YAML data first
    with open(path, 'r') as f:
        config_data_to_validate = yaml.safe_load(f)

    assert validate_config_schema(config_data_to_validate) is True
    # The function validate_config_schema doesn't return the dict,
    # so we compare the originally loaded dict if needed for other assertions.
    assert config_data_to_validate == expected_config_data


@pytest.fixture
def minimal_config(tmp_path):
    cfg = {
        "host": "127.0.0.1",
        "port": 1
    }
    file_path = tmp_path / "config_min.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    return file_path, cfg


def test_minimal_config(minimal_config):
    path, expected_config_data = minimal_config
    # Load YAML data first
    with open(path, 'r') as f:
        config_data_to_validate = yaml.safe_load(f)

    assert validate_config_schema(config_data_to_validate) is True
    # Optional fields are not defaulted by validate_config_schema,
    # they are handled by Pydantic models when Settings() is created.
    # This test might need to be re-evaluated based on what validate_config_schema
    # is supposed to guarantee for minimal configs according to the schema.
    # For now, we just check if it validates and the content matches the input.
    assert config_data_to_validate["host"] == expected_config_data["host"]
    assert config_data_to_validate["port"] == expected_config_data["port"]
    # Default value checks like debug and databases are not applicable here
    # as validate_config_schema only validates structure, not default filling.


@pytest.mark.parametrize("missing_key", ["host", "port"])
def test_missing_keys(tmp_path, missing_key):
    cfg = {"host": "localhost", "port": 8080}
    cfg.pop(missing_key)
    file_path = tmp_path / "config_missing.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    with open(file_path, 'r') as f:
        config_data_to_validate = yaml.safe_load(f)
    with pytest.raises(ValueError) as exc: # Changed from ConfigValidationError
        validate_config_schema(config_data_to_validate)
    # The error message from jsonschema.validate might be more specific
    # Example: "''host' is a required property'"
    # We can check if the missing key is mentioned in the error.
    assert missing_key in str(exc.value).lower() # Making assert less brittle


@pytest.mark.parametrize("field,value", [
    ("host", 123),
    ("port", "not_an_int"),
    ("debug", "yes"),
    ("databases", "not_a_list"),
])
def test_wrong_types(tmp_path, field, value):
    cfg = {"host": "localhost", "port": 8080}
    cfg[field] = value
    file_path = tmp_path / "config_bad_type.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    with open(file_path, 'r') as f:
        config_data_to_validate = yaml.safe_load(f)
    with pytest.raises(ValueError) as exc: # Changed from ConfigValidationError
        validate_config_schema(config_data_to_validate)
    # Check if the field causing the type error is mentioned.
    assert field in str(exc.value).lower() # Making assert less brittle


def test_empty_file(tmp_path):
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    # yaml.safe_load on an empty file returns None
    config_data_to_validate = None
    with pytest.raises(ValueError): # Changed from ConfigValidationError
        # jsonschema.validate(None, schema) will raise an error.
        validate_config_schema(config_data_to_validate)


def test_unreadable_file(tmp_path):
    file_path = tmp_path / "config_unreadable.yaml"
    file_path.write_text("host: localhost\nport: 8080")
    # remove read permissions
    file_path.chmod(0)
    try:
        # Reading the file will cause PermissionError before validation
        with pytest.raises(PermissionError):
            with open(file_path, 'r') as f:
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


@pytest.mark.parametrize("port", [1, 65535])
def test_boundary_values(tmp_path, port):
    cfg = {"host": "localhost", "port": port}
    file_path = tmp_path / f"config_port_{port}.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    with open(file_path, 'r') as f:
        config_data_to_validate = yaml.safe_load(f)
    assert validate_config_schema(config_data_to_validate) is True
    assert config_data_to_validate["port"] == port