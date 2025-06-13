import os
import stat
from pathlib import Path

import pytest
import yaml

from src.config.validation import validate_config, ConfigValidationError


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
    path, expected = valid_config
    result = validate_config(path)
    assert isinstance(result, dict)
    assert result == expected


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
    path, expected = minimal_config
    result = validate_config(path)
    # optional fields should default
    assert result["host"] == expected["host"]
    assert result["port"] == expected["port"]
    assert result.get("debug", False) is False
    assert result.get("databases") == []


@pytest.mark.parametrize("missing_key", ["host", "port"])
def test_missing_keys(tmp_path, missing_key):
    cfg = {"host": "localhost", "port": 8080}
    cfg.pop(missing_key)
    file_path = tmp_path / "config_missing.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigValidationError) as exc:
        validate_config(file_path)
    assert missing_key in str(exc.value)


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
    with pytest.raises(ConfigValidationError) as exc:
        validate_config(file_path)
    assert field in str(exc.value)


def test_empty_file(tmp_path):
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    with pytest.raises(ConfigValidationError):
        validate_config(file_path)


def test_unreadable_file(tmp_path):
    file_path = tmp_path / "config_unreadable.yaml"
    file_path.write_text("host: localhost\nport: 8080")
    # remove read permissions
    file_path.chmod(0)
    try:
        with pytest.raises((ConfigValidationError, PermissionError)):
            validate_config(file_path)
    finally:
        # restore so pytest can clean up
        file_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.parametrize("port", [1, 65535])
def test_boundary_values(tmp_path, port):
    cfg = {"host": "localhost", "port": port}
    file_path = tmp_path / f"config_port_{port}.yaml"
    file_path.write_text(yaml.safe_dump(cfg))
    result = validate_config(file_path)
    assert result["port"] == port