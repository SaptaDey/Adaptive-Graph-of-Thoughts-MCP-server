# Framework: pytest
import copy
import json
import yaml

import pytest

from config_schema import validate_config, ConfigSchema


@pytest.fixture
def base_valid_config():
    """Return the minimal valid configuration dict."""
    return {
        "database": {
            "url": "postgres://user:pass@localhost:5432/db",
            "pool_size": 5,
        },
        "logging": {
            "level": "INFO",
            "handlers": ["console"],
        },
        "features": {
            "enable_x": True,
            "threshold": 10,
        },
    }


def mutate_config(cfg, **changes):
    """Deep-copy the base config and apply keyword overrides."""
    new_cfg = copy.deepcopy(cfg)
    for key, value in changes.items():
        new_cfg[key] = value
    return new_cfg


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(base_valid_config(), id="minimal_valid"),
        pytest.param(
            mutate_config(
                base_valid_config(),
                features={"enable_x": False, "threshold": 20},
            ),
            id="full_config",
        ),
        pytest.param(
            {k: v for k, v in base_valid_config().items() if k != "logging"},
            id="missing_optional_logging",
        ),
        pytest.param(
            mutate_config(
                base_valid_config(),
                features={"threshold": 0},
            ),
            id="boundary_numeric_limits",
        ),
    ],
)
def test_validate_config_happy_path(config):
    """Test that validate_config accepts valid configurations."""
    result = validate_config(config)
    assert isinstance(result, dict)
    # Basic sanity check on returned structure
    assert "database" in result
    assert "url" in result["database"]


@pytest.mark.parametrize(
    "config, expected_exception, expected_msg_part",
    [
        pytest.param(
            mutate_config(base_valid_config(), database=None),
            KeyError,
            "database",
            id="missing_database",
        ),
        pytest.param(
            mutate_config(
                base_valid_config(), database={"url": 123}
            ),
            TypeError,
            "database.url",
            id="incorrect_database_url_type",
        ),
        pytest.param(
            mutate_config(
                base_valid_config(), logging={"level": "VERBOSE"}
            ),
            ValueError,
            "logging.level",
            id="invalid_logging_level",
        ),
        pytest.param(
            mutate_config(
                base_valid_config(), features={"threshold": -1}
            ),
            ValueError,
            "features.threshold",
            id="out_of_range_threshold",
        ),
        pytest.param(
            {**base_valid_config(), "extra": "value"},
            KeyError,
            "extra",
            id="unexpected_extra_key",
        ),
    ],
)
def test_validate_config_invalid_configs(
    config, expected_exception, expected_msg_part
):
    """Test that validate_config rejects invalid configurations."""
    with pytest.raises(expected_exception) as excinfo:
        validate_config(config)
    assert expected_msg_part in str(excinfo.value)


def test_validate_config_from_file(tmp_path):
    """Test that validate_config loads configuration from a file path."""
    config = base_valid_config()
    file_path = tmp_path / "config.yaml"
    file_path.write_text(yaml.safe_dump(config))
    result = validate_config(str(file_path))
    assert isinstance(result, dict)
    assert result["database"]["url"] == config["database"]["url"]


def test_validate_config_empty_config_raises():
    """Test that empty configuration dict is rejected."""
    with pytest.raises(TypeError):
        validate_config({})


def test_validate_config_none_raises_type_error():
    """Test that passing None raises TypeError."""
    with pytest.raises(TypeError):
        validate_config(None)