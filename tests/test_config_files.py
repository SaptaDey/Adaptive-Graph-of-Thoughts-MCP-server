import json
from pathlib import Path

import pytest
import yaml

# Directories to scan for configuration files
config_dirs = [Path("config"), Path("settings")]

# Gather all YAML files (*.yml, *.yaml)
yaml_paths = []
for d in config_dirs:
    if d.exists():
        yaml_paths.extend(d.rglob("*.yml"))
        yaml_paths.extend(d.rglob("*.yaml"))

# Gather all JSON files
json_paths = []
for d in config_dirs:
    if d.exists():
        json_paths.extend(d.rglob("*.json"))


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_is_valid(yaml_path):
    """
    Validate that each YAML file parses without error and
    has a top-level mapping or sequence.
    """
    content = yaml_path.read_text()
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML syntax error in {yaml_path}: {e}")
    assert isinstance(data, (dict, list)), (
        f"Top-level YAML should be a mapping or list in {yaml_path}"
    )

    # Illustrative: require "name" and "version" in app.yml
    if yaml_path.name == "app.yml" and isinstance(data, dict):
        required_keys = {"name", "version"}
        missing = required_keys - set(data.keys())
        assert not missing, f"{yaml_path} missing required keys {missing}"


@pytest.mark.parametrize("json_path", json_paths)
def test_json_is_valid(json_path):
    """
    Validate that each JSON file parses without error and
    has a top-level mapping or array.
    """
    content = json_path.read_text()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        pytest.fail(f"JSON syntax error in {json_path}: {e}")
    assert isinstance(data, (dict, list)), (
        f"Top-level JSON should be a mapping or list in {json_path}"
    )


def test_no_empty_config_files():
    """
    Ensure no configuration file is empty.
    """
    for d in config_dirs:
        if not d.exists():
            continue
        for pattern in ("*.yml", "*.yaml", "*.json"):
            for path in d.rglob(pattern):
                size = path.stat().st_size
                assert size > 0, f"{path} is unexpectedly empty (size={size} bytes)"
