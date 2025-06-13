# Tests for adaptive_graph_of_thoughts.config
# Using pytest as the chosen test framework per repository conventions

import importlib
import json
import textwrap
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest


@pytest.fixture
def temp_config_files(tmp_path, monkeypatch):
    # Arrange: create settings YAML
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text(
        textwrap.dedent(
            """
            app:
              name: Test App
              port: 9000
            """
        )
    )
    # Arrange: create JSON schema
    schema_json = tmp_path / "config.schema.json"
    schema_json.write_text(
        json.dumps({
            "type": "object",
            "properties": {
                "app": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "port": {"type": "integer"}
                    },
                    "required": ["name", "port"]
                }
            },
            "required": ["app"]
        })
    )
    # Patch config paths before import
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.config_file_path",
        settings_yaml,
        raising=False
    )
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.schema_file_path",
        schema_json,
        raising=False
    )
    yield settings_yaml, schema_json


def test_settings_load_success(temp_config_files):
    # Act
    import adaptive_graph_of_thoughts.config as cfg
    settings = cfg.Settings()
    # Assert
    assert settings.app.name == "Test App"
    assert settings.app.port == 9000


def test_invalid_yaml_schema(monkeypatch, tmp_path):
    # Arrange: write invalid YAML and empty schema
    bad_yaml = tmp_path / "settings.yaml"
    bad_yaml.write_text("not: conforming")
    bad_schema = tmp_path / "config.schema.json"
    bad_schema.write_text("{}")
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.config_file_path",
        bad_yaml,
        raising=False
    )
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.schema_file_path",
        bad_schema,
        raising=False
    )
    # Act & Assert: reload should raise on schema validation
    with pytest.raises(ValueError):
        importlib.reload(importlib.import_module("adaptive_graph_of_thoughts.config"))


def test_missing_schema_file(monkeypatch, tmp_path, caplog):
    # Arrange: valid YAML but missing schema file
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text("app:\n  name: MissingSchema\n")
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.config_file_path",
        settings_yaml,
        raising=False
    )
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.schema_file_path",
        tmp_path / "nope.json",
        raising=False
    )
    caplog.set_level("WARNING")
    # Act
    module = importlib.reload(importlib.import_module("adaptive_graph_of_thoughts.config"))
    # Assert
    assert "Schema file not found" in caplog.text
    assert module.Settings().app.name == "MissingSchema"


def test_env_override(monkeypatch, temp_config_files):
    # Arrange: override via environment variable
    monkeypatch.setenv("APP__PORT", "7007")
    import adaptive_graph_of_thoughts.config as cfg
    # Act
    importlib.reload(cfg)
    settings = cfg.Settings()
    # Assert
    assert settings.app.port == 7007


def test_extra_yaml_field_ignored(monkeypatch, tmp_path):
    # Arrange: unexpected key with permissive schema
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text("foo: bar")
    schema_file = tmp_path / "config.schema.json"
    schema_file.write_text("{}")
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.config_file_path",
        yaml_file,
        raising=False
    )
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.config.schema_file_path",
        schema_file,
        raising=False
    )
    # Act
    module = importlib.reload(importlib.import_module("adaptive_graph_of_thoughts.config"))
    settings = module.Settings()
    # Assert
    assert not hasattr(settings, "foo")


def test_public_api_stable():
    # Act
    import adaptive_graph_of_thoughts.config as cfg
    # Assert: public API remains unchanged
    expected_attrs = {"Settings", "validate_config_schema"}
    assert expected_attrs.issubset(set(dir(cfg)))