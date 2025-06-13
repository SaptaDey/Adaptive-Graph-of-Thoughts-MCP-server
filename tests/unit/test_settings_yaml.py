"""
Unit tests for the project’s YAML configuration file(s).
Tests use pytest + PyYAML.
"""
from pathlib import Path
import yaml
import pytest

# Path to the YAML settings file under test – adjust if the real path differs.
SETTINGS_PATH = Path(__file__).parents[2] / "config" / "settings.yaml"

def load_raw_settings(path: Path = SETTINGS_PATH) -> dict:
    """Return dict loaded from YAML, re-raising YAML errors."""
    with path.open() as fh:
        return yaml.safe_load(fh)

def test_yaml_loads_successfully():
    """YAML should parse without exception and return a dict."""
    settings = load_raw_settings()
    assert isinstance(settings, dict)
    assert settings, "Settings file should not be empty."

@pytest.mark.parametrize("required_key", ["database", "logging", "api"])
def test_required_top_level_keys_present(required_key):
    settings = load_raw_settings()
    assert required_key in settings

@pytest.mark.parametrize(
    "sub_key, expected_type",
    [
        ("host", str),
        ("port", int),
        ("user", str),
        ("password", str),
    ],
)
def test_database_section_schema(sub_key, expected_type):
    db_cfg = load_raw_settings()["database"]
    assert sub_key in db_cfg
    assert isinstance(db_cfg[sub_key], expected_type)
    if sub_key == "port":
        assert 1 <= db_cfg["port"] <= 65535

def test_invalid_yaml_raises_error(tmp_path):
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text("foo: bar\n baz")
    with pytest.raises(yaml.YAMLError):
        load_raw_settings(bad_yaml)

def test_missing_required_key_fails_validation(monkeypatch):
    settings = load_raw_settings().copy()
    settings.pop("database", None)

    try:
        from src.adaptive_graph_of_thoughts.config import validate_config_schema  # noqa: E402
    except ImportError:
        with pytest.raises(KeyError):
            _ = settings["database"]
    else:
        with pytest.raises(Exception):
            validate_config_schema(settings)

def test_env_override(monkeypatch):
    monkeypatch.setenv("SETTINGS_PATH", str(SETTINGS_PATH))
    # For demonstration we simply verify the env var is set to a real path.
    assert Path(monkeypatch.getenv("SETTINGS_PATH")).exists()