import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import threading
from contextlib import contextmanager
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class AGoTSettings(BaseSettings):
    """Application settings loaded from environment variables or `.env`."""

    llm_provider: str = Field(
        default="openai",
        description="LLM provider identifier: 'openai' or 'claude'",
    )
    openai_api_key: Optional[str] = Field(
        default=None, description="API key for OpenAI completions"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description="API key for Anthropic Claude"
    )

    model_config = SettingsConfigDict(env_file=".env")



class EnvSettings(AGoTSettings):
    """Backward-compatible alias used in tests."""


env_settings = EnvSettings()

# Thread safety lock
_config_lock = threading.RLock()


def validate_learning_rate(lr: float) -> None:
    """Validate learning rate is in valid range."""
    if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
        raise ValueError(f"Learning rate must be between 0 and 1.0, got {lr}")


def validate_batch_size(batch_size: int) -> None:
    """Validate batch size is positive integer."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer, got {batch_size}")


def validate_max_steps(max_steps: int) -> None:
    """Validate max steps is positive integer."""
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"Max steps must be a positive integer, got {max_steps}")


def validate_config_schema(_config_data: dict) -> bool:
    """Validate settings data against the Pydantic schema."""

    try:
        SettingsFileModel.model_validate(_config_data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return True


# Simple data classes for configuration
class AppConfig:
    def __init__(
        self,
        name: str = "Adaptive Graph of Thoughts",
        version: str = "0.1.0",
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = True,
        log_level: str = "INFO",
        cors_allowed_origins_str: str = "*",
        auth_token: str | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.host = host
        self.port = port
        self.reload = reload
        self.log_level = log_level
        self.cors_allowed_origins_str = cors_allowed_origins_str
        self.auth_token = auth_token


class ASRGoTDefaultParams:
    def __init__(
        self,
        initial_confidence=0.8,
        confidence_threshold=0.75,
        max_iterations=10,
        convergence_threshold=0.05,
    ):
        self.initial_confidence = initial_confidence
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold


class PubMedConfig:
    def __init__(self, api_key=None, max_results=20, rate_limit_delay=0.5):
        self.api_key = api_key
        self.max_results = max_results
        self.rate_limit_delay = rate_limit_delay


class GoogleScholarConfig:
    def __init__(self, max_results=10, rate_limit_delay=1.0):
        self.max_results = max_results
        self.rate_limit_delay = rate_limit_delay


class ExaSearchConfig:
    def __init__(self, api_key=None, max_results=10):
        self.api_key = api_key
        self.max_results = max_results


class KnowledgeDomain:
    def __init__(self, name, description="", keywords=None):
        self.name = name
        self.description = description
        self.keywords = keywords or []


class LegacyConfig:
    def __init__(
        self, learning_rate=0.01, batch_size=32, max_steps=1000, frozen=False, **kwargs
    ):
        with _config_lock:
            # Main configuration attributes expected by tests
            validate_learning_rate(learning_rate)
            validate_batch_size(batch_size)
            validate_max_steps(max_steps)

            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.max_steps = max_steps
            self._frozen = frozen

            # Legacy configuration structure
            self.app = AppConfig()
            self.asr_got = ASRGoTDefaultParams()
            self.google_scholar = None
            self.pubmed = None
            self.exa_search = None
            self.knowledge_domains = []
            # Load YAML configuration if exists
            config_file_path = (
                Path(__file__).parent.parent.parent / "config" / "settings.yaml"
            )
            if config_file_path.exists():
                try:
                    with open(config_file_path) as f:
                        yaml.safe_load(f)  # Just validate, don't store
                except Exception:
                    pass

            # Apply additional kwargs
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def __setattr__(self, name, value):
        if hasattr(self, "_frozen") and self._frozen and hasattr(self, name):
            raise AttributeError("Cannot modify frozen config")
        super().__setattr__(name, value)

    def __eq__(self, other):
        if not isinstance(other, LegacyConfig):
            return False
        return (
            self.learning_rate == other.learning_rate
            and self.batch_size == other.batch_size
            and self.max_steps == other.max_steps
        )

    def __repr__(self):
        return f"Config(learning_rate={self.learning_rate}, batch_size={self.batch_size}, max_steps={self.max_steps})"

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for pydantic v2 compatibility."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
        }

    def copy(self) -> "LegacyConfig":
        """Create a deep copy of the config."""
        return LegacyConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            frozen=False,
        )

    def update(self, updates: dict[str, Any]) -> None:
        """Update config with new values."""
        with _config_lock:
            for key, value in updates.items():
                if hasattr(self, key):
                    if key == "learning_rate":
                        validate_learning_rate(value)
                    elif key == "batch_size":
                        validate_batch_size(value)
                    elif key == "max_steps":
                        validate_max_steps(value)
                    setattr(self, key, value)

    def merge(self, other: "LegacyConfig") -> "LegacyConfig":
        """Merge with another config, other takes precedence."""
        return LegacyConfig(
            learning_rate=(
                other.learning_rate
                if hasattr(other, "learning_rate")
                else self.learning_rate
            ),
            batch_size=(
                other.batch_size if hasattr(other, "batch_size") else self.batch_size
            ),
            max_steps=(
                other.max_steps if hasattr(other, "max_steps") else self.max_steps
            ),
        )

    @classmethod
    def load(cls, file_path: str) -> "LegacyConfig":
        """Load config from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        content = path.read_text().strip()
        if not content:
            raise ValueError("Empty configuration file")
        try:
            if path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(content)
            elif path.suffix.lower() == ".json":
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        except yaml.YAMLError as err:
            raise yaml.YAMLError(f"Invalid YAML: {err}") from err
        except json.JSONDecodeError as err:
            raise json.JSONDecodeError(err.msg, err.doc, err.pos) from err

        if not data:
            raise ValueError("Empty configuration file")

        # Check for required keys
        required_keys = ["learning_rate", "batch_size"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        # Validate data types
        if not isinstance(data.get("learning_rate"), (int, float)):
            raise ValueError("learning_rate must be a number")
        if not isinstance(data.get("batch_size"), int):
            raise ValueError("batch_size must be an integer")
        if "max_steps" in data and not isinstance(data["max_steps"], int):
            raise ValueError("max_steps must be an integer")

        return cls(**data)

    def save(self, file_path: str) -> None:
        """Save config to file."""
        path = Path(file_path)
        data = self.model_dump()

        if path.suffix.lower() in [".yaml", ".yml"]:
            content = yaml.dump(data, default_flow_style=False)
        elif path.suffix.lower() == ".json":
            content = json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        try:
            with _config_lock:
                with open(file_path, "w") as f:
                    f.write(content)
        except PermissionError as err:
            raise PermissionError(f"Permission denied writing to: {file_path}") from err

    @classmethod
    def from_env(cls) -> "LegacyConfig":
        """Load config from environment variables."""
        data = {}

        if "LEARNING_RATE" in os.environ:
            data["learning_rate"] = float(os.environ["LEARNING_RATE"])
        else:
            data["learning_rate"] = 0.01

        if "BATCH_SIZE" in os.environ:
            data["batch_size"] = int(os.environ["BATCH_SIZE"])
        else:
            data["batch_size"] = 32

        if "MAX_STEPS" in os.environ:
            data["max_steps"] = int(os.environ["MAX_STEPS"])
        else:
            data["max_steps"] = 1000

        return cls(**data)

    @classmethod
    def load_with_overrides(cls, base_file: str, override_file: str) -> "LegacyConfig":
        """Load config with hierarchical overrides."""
        # Load base config normally
        base_config = cls.load(base_file)

        # Load override data without validation
        path = Path(override_file)
        if not path.exists():
            raise FileNotFoundError(f"Override file not found: {override_file}")

        content = path.read_text().strip()
        if not content:
            return base_config  # No overrides, return base

        try:
            if path.suffix.lower() in [".yaml", ".yml"]:
                override_data = yaml.safe_load(content)
            elif path.suffix.lower() == ".json":
                override_data = json.loads(content)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        except yaml.YAMLError as err:
            raise yaml.YAMLError(f"Invalid YAML: {err}") from err
        except json.JSONDecodeError as err:
            raise json.JSONDecodeError(err.msg, err.doc, err.pos) from err

        if not override_data:
            return base_config  # No overrides, return base

        # Apply overrides manually
        new_config = base_config.copy()
        new_config.update(override_data)
        return new_config


# Create some aliases for backward compatibility
class Settings(LegacyConfig):
    pass


# Global config instance
config = LegacyConfig()
legacy_settings = config  # For backward compatibility

# ---------------------------------------------------------------------------
# New Pydantic-based settings
# ---------------------------------------------------------------------------


class Neo4jSettingsModel(BaseSettings):
    """Connection details for Neo4j database."""

    uri: str = "neo4j://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=".env")


class AppSettingsModel(BaseModel):
    """Application runtime settings loaded from YAML and env vars."""

    name: str = "Adaptive Graph of Thoughts"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    uvicorn_reload: bool = True
    uvicorn_workers: int = 1
    cors_allowed_origins_str: str = "*"
    auth_token: str | None = None
    mcp_transport_type: str = "http"
    mcp_stdio_enabled: bool = True
    mcp_http_enabled: bool = True
    log_level: str = "INFO"


class MCPSettingsModel(BaseModel):
    """Model Context Protocol server settings."""

    protocol_version: str
    server_name: str
    server_version: str
    vendor_name: str

    model_config = SettingsConfigDict(extra="forbid")


class GoogleScholarSettingsModel(BaseModel):
    api_key: str | None = None
    base_url: str = "https://serpapi.com/search"

    model_config = SettingsConfigDict(extra="forbid")


class PubMedSettingsModel(BaseModel):
    api_key: str | None = None
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email: str | None = None

    model_config = SettingsConfigDict(extra="forbid")


class ExaSearchSettingsModel(BaseModel):
    api_key: str | None = None
    base_url: str = "https://api.exa.ai"

    model_config = SettingsConfigDict(extra="forbid")


class KnowledgeDomainModel(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)
    description: str | None = None

    model_config = SettingsConfigDict(extra="forbid")


class SettingsFileModel(BaseModel):
    """Schema for validating `settings.yaml`."""

    app: AppSettingsModel
    asr_got: dict[str, Any]
    mcp_settings: MCPSettingsModel
    google_scholar: GoogleScholarSettingsModel | None = None
    pubmed: PubMedSettingsModel | None = None
    exa_search: ExaSearchSettingsModel | None = None
    knowledge_domains: list[KnowledgeDomainModel] = Field(default_factory=list)

    model_config = SettingsConfigDict(extra="forbid")


class RuntimeSettings(BaseSettings):
    """Central runtime settings loaded from YAML and environment."""

    app: AppSettingsModel = Field(
        default_factory=AppSettingsModel,
        description="Web application configuration",
    )
    neo4j: Neo4jSettingsModel = Field(
        default_factory=Neo4jSettingsModel,
        description="Neo4j connection options",
    )
    asr_got: dict[str, Any] = Field(
        default_factory=dict,
        description="Advanced reasoning parameters",
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore"
    )


def load_runtime_settings() -> RuntimeSettings:
    """
    Load runtime application settings from a YAML file and environment variables.

    If `config/settings.yaml` exists, its contents are loaded and used as defaults, with environment variables taking precedence. Returns a `RuntimeSettings` instance containing the merged configuration.

    Returns:
        RuntimeSettings: The combined runtime settings loaded from file and environment.
    """

    yaml_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    data: dict[str, Any] = {}
    if yaml_path.exists():
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
        validate_config_schema(data)
    return RuntimeSettings(**data)


runtime_settings = load_runtime_settings()

# Re-export runtime settings for application modules
settings = runtime_settings

# ---------------------------------------------------------------------------
# Simplified configuration used in tests
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model related configuration."""

    name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class GraphConfig:
    """Graph processing configuration."""

    max_depth: int = 5
    max_breadth: int = 3
    pruning_threshold: float = 0.1
    enable_caching: bool = True
    cache_size: int = 1000


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    file_path: Optional[str] = None
    enable_console: bool = True


@dataclass
class Config:
    """Main application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            graph=GraphConfig(**data.get("graph", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "graph": asdict(self.graph),
            "logging": asdict(self.logging),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError("Configuration file not found")
        text = p.read_text().strip()
        if not text:
            raise ValueError("Failed to parse configuration file")
        try:
            if p.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(text)
            elif p.suffix.lower() == ".json":
                data = json.loads(text)
            else:
                raise ValueError("Unsupported file format")
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ValueError("Failed to parse configuration file") from exc
        except ValueError:
            raise
        return cls.from_dict(data or {})

    def save_to_file(self, path: Union[str, Path]) -> None:
        p = Path(path)
        data = self.to_dict()
        if p.suffix.lower() in {".yaml", ".yml"}:
            content = yaml.dump(data)
        elif p.suffix.lower() == ".json":
            content = json.dumps(data)
        else:
            raise ValueError("Unsupported file format")
        try:
            with open(p, "w") as fh:
                fh.write(content)
        except Exception as exc:
            raise RuntimeError("Failed to save configuration") from exc

    @classmethod
    def from_env(cls, prefix: str = "AGOT_") -> "Config":
        prefix = prefix.upper()

        def read(key: str, cast, default=None):
            val = os.getenv(prefix + key)
            if val is None:
                return default
            if cast is bool:
                return val.lower() in {"1", "true", "yes", "on"}
            try:
                return cast(val)
            except ValueError as exc:
                raise ValueError(f"Invalid value for {prefix}{key}") from exc

        model = ModelConfig(
            name=read("MODEL_NAME", str, ModelConfig.name),
            temperature=read("MODEL_TEMPERATURE", float, ModelConfig.temperature),
            max_tokens=read("MODEL_MAX_TOKENS", int, ModelConfig.max_tokens),
            timeout=read("MODEL_TIMEOUT", int, ModelConfig.timeout),
            api_key=read("MODEL_API_KEY", str, None),
            base_url=read("MODEL_BASE_URL", str, None),
        )

        graph = GraphConfig(
            max_depth=read("GRAPH_MAX_DEPTH", int, GraphConfig.max_depth),
            max_breadth=read("GRAPH_MAX_BREADTH", int, GraphConfig.max_breadth),
            pruning_threshold=read(
                "GRAPH_PRUNING_THRESHOLD", float, GraphConfig.pruning_threshold
            ),
            enable_caching=read(
                "GRAPH_ENABLE_CACHING", bool, GraphConfig.enable_caching
            ),
            cache_size=read("GRAPH_CACHE_SIZE", int, GraphConfig.cache_size),
        )

        logging_cfg = LoggingConfig(
            level=read("LOGGING_LEVEL", str, LoggingConfig.level),
            format=read("LOGGING_FORMAT", str, LoggingConfig.format),
            file_path=read("LOGGING_FILE_PATH", str, None),
            enable_console=read(
                "LOGGING_ENABLE_CONSOLE", bool, LoggingConfig.enable_console
            ),
        )

        return cls(model=model, graph=graph, logging=logging_cfg)

    # ------------------------------------------------------------------
    # Validation and update helpers
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if not 0.0 <= self.model.temperature <= 2.0:
            raise ValueError("Model temperature must be between 0.0 and 2.0")
        if self.model.max_tokens <= 0:
            raise ValueError("Model max_tokens must be positive")
        if self.model.timeout <= 0:
            raise ValueError("Model timeout must be positive")
        if self.graph.max_depth <= 0:
            raise ValueError("Graph max_depth must be positive")
        if self.graph.max_breadth <= 0:
            raise ValueError("Graph max_breadth must be positive")
        if not 0.0 <= self.graph.pruning_threshold <= 1.0:
            raise ValueError("Graph pruning_threshold must be between 0.0 and 1.0")
        if self.graph.cache_size <= 0:
            raise ValueError("Graph cache_size must be positive")
        if self.logging.level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                "Logging level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )

    def update(self, **sections: dict[str, Any]) -> None:
        for section, values in sections.items():
            target = getattr(self, section, None)
            if target is None:
                logging.warning("Unknown config key: %s", section)
                continue
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
                else:
                    logging.warning("Unknown nested config key: %s.%s", section, key)


class ThreadSafeConfig:
    """Thread-safe wrapper around a ``Config`` instance."""

    def __init__(self, initial: Optional[Config] = None) -> None:
        self._lock = threading.RLock()
        self._config = initial or Config()

    @contextmanager
    def _acquire_lock(self):
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def get_config(self) -> Config:
        with self._acquire_lock():
            return self._config

    def set_config(self, cfg: Config) -> None:
        with self._acquire_lock():
            cfg.validate()
            self._config = cfg

    def update_config(self, updates: dict[str, Any]) -> None:
        with self._acquire_lock():
            self._config.update(**updates)


_global_config = ThreadSafeConfig()


def get_config() -> Config:
    """Return the process-wide configuration instance."""
    return _global_config.get_config()


def set_config(cfg: Config) -> None:
    """Replace the global configuration instance after validation."""
    _global_config.set_config(cfg)


def load_config(
    file_path: Optional[Union[str, Path]] = None, env_prefix: str = "AGOT_"
) -> Config:
    cfg = Config()
    if file_path:
        try:
            cfg = Config.from_file(file_path)
        except Exception as exc:
            logging.warning("Failed to load config from file: %s", exc)
    try:
        env_cfg = Config.from_env(prefix=env_prefix)
        updates = {}
        if (
            os.getenv(f"{env_prefix}MODEL_NAME")
            or os.getenv(f"{env_prefix}MODEL_TEMPERATURE")
            or os.getenv(f"{env_prefix}MODEL_MAX_TOKENS")
            or os.getenv(f"{env_prefix}MODEL_TIMEOUT")
            or os.getenv(f"{env_prefix}MODEL_API_KEY")
            or os.getenv(f"{env_prefix}MODEL_BASE_URL")
        ):
            updates["model"] = {
                k: v for k, v in asdict(env_cfg.model).items() if v is not None
            }
        if (
            os.getenv(f"{env_prefix}GRAPH_MAX_DEPTH")
            or os.getenv(f"{env_prefix}GRAPH_MAX_BREADTH")
            or os.getenv(f"{env_prefix}GRAPH_PRUNING_THRESHOLD")
            or os.getenv(f"{env_prefix}GRAPH_ENABLE_CACHING")
            or os.getenv(f"{env_prefix}GRAPH_CACHE_SIZE")
        ):
            updates["graph"] = {
                k: v for k, v in asdict(env_cfg.graph).items() if v is not None
            }
        if (
            os.getenv(f"{env_prefix}LOGGING_LEVEL")
            or os.getenv(f"{env_prefix}LOGGING_FORMAT")
            or os.getenv(f"{env_prefix}LOGGING_FILE_PATH")
            or os.getenv(f"{env_prefix}LOGGING_ENABLE_CONSOLE")
        ):
            updates["logging"] = {
                k: v for k, v in asdict(env_cfg.logging).items() if v is not None
            }
        if updates:
            cfg.update(**updates)
    except Exception as exc:
        logging.warning("Failed to load config from environment: %s", exc)

    set_config(cfg)
    return cfg
