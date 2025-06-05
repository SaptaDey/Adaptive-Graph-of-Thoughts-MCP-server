from pathlib import Path
from typing import Any, Optional, Type  # Added Type for settings_cls hint
import sys  # For type checking PydanticBaseSettingsSource

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

import json
import jsonschema
from loguru import logger

# Load the YAML configuration file
config_file_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
yaml_config = {}
if config_file_path.exists():
    with open(config_file_path) as f:
        yaml_config = yaml.safe_load(f)

schema_file_path = Path(__file__).parent.parent.parent / "config" / "config.schema.json"

def validate_config_schema(config_data: dict) -> bool:
    """Validate configuration against JSON Schema."""
    try:
        if schema_file_path.exists():
            with open(schema_file_path) as f:
                schema = json.load(f)
            jsonschema.validate(config_data, schema)
            logger.info("Configuration validation successful")
            return True
        else:
            logger.warning(f"Schema file not found: {schema_file_path}")
            return False
    except jsonschema.ValidationError as e:
        logger.error(f"Configuration validation failed: {e.message}")
        raise ValueError(f"Invalid configuration: {e.message}")
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False

# --- Models for NexusMind Default Parameters ---
class HypothesisParams(BaseModel):
    min_hypotheses: int = Field(default=2, alias="min")
    max_hypotheses: int = Field(default=4, alias="max")

class DecompositionDimension(BaseModel):
    label: str
    description: str

class ASRGoTDefaultParams(BaseModel):
    initial_confidence: list[float] = Field(default=[0.9, 0.9, 0.9, 0.9])
    initial_layer: str = Field(default="root_layer")
    default_decomposition_dimensions: list[DecompositionDimension] = Field(
        default_factory=list
    )
    dimension_confidence: list[float] = Field(default=[0.8, 0.8, 0.8, 0.8])
    hypotheses_per_dimension: HypothesisParams = Field(
        default_factory=HypothesisParams, alias="hypotheses_per_dimension"
    )
    hypothesis_confidence: list[float] = Field(default=[0.5, 0.5, 0.5, 0.5])
    default_disciplinary_tags: list[str] = Field(default_factory=list)
    default_plan_types: list[str] = Field(default_factory=list)
    evidence_max_iterations: int = Field(default=5)
    pruning_confidence_threshold: float = Field(default=0.2)
    pruning_impact_threshold: float = Field(default=0.3)
    merging_semantic_overlap_threshold: float = Field(default=0.8)
    subgraph_min_confidence_threshold: float = Field(default=0.6)
    subgraph_min_impact_threshold: float = Field(default=0.5)
    # temporal_recency_days: Optional[int] = None # Example if used

class LayerDefinition(BaseModel):
    description: str

class StageItemConfig(BaseModel):
    name: str = Field(description="A friendly name for the stage (e.g., 'Initialization').")
    module_path: str = Field(description="The full Python path to the stage class (e.g., 'src.asr_got_reimagined.domain.stages.InitializationStage').")
    enabled: bool = Field(default=True, description="Whether this stage is enabled and should be included in the pipeline.")

class ASRGoTConfig(BaseModel):
    default_parameters: ASRGoTDefaultParams = Field(default_factory=ASRGoTDefaultParams)
    layers: dict[str, LayerDefinition] = Field(default_factory=dict)
    pipeline_stages: list[StageItemConfig] = Field(default_factory=list, description="Defines the sequence of stages in the ASR-GoT processing pipeline.")

# --- Models for MCP Settings ---
class MCPSettings(BaseModel):
    protocol_version: str = Field(default="2024-11-05")
    server_name: str = Field(default="NexusMind MCP Server")
    server_version: str = Field(default="0.1.0")
    vendor_name: str = Field(default="AI Research Group")
    # display_name: Optional[str] = None
    # description: Optional[str] = None

# --- Models for optional Claude API direct integration ---
class ClaudeAPIConfig(BaseModel):
    api_key: Optional[str] = (
        None  # Example: "env_var:CLAUDE_API_KEY" or actual key for dev
    )
    default_model: str = Field(default="claude-3-opus-20240229")
    timeout_seconds: int = Field(default=120)
    max_retries: int = Field(default=2)

class KnowledgeDomain(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)
    description: Optional[str] = None

# --- Main Application Settings Model ---
class AppSettings(BaseModel):
    name: str = Field(default="NexusMind")
    version: str = Field(default="0.1.0")
    host: str = Field(default="0.0.0.0")
    port: int = Field(
        default=8000
    )  # Can be overridden by APP__PORT environment variable
    log_level: str = Field(default="INFO")
    cors_allowed_origins_str: str = Field(default="*", validation_alias="APP_CORS_ALLOWED_ORIGINS_STR", description="Comma-separated list of allowed CORS origins, or '*' for all.")
    uvicorn_reload: bool = Field(default=True, validation_alias="APP_UVICORN_RELOAD", description="Enable Uvicorn auto-reload (True for dev, False for prod).")
    uvicorn_workers: int = Field(default=1, validation_alias="APP_UVICORN_WORKERS", description="Number of Uvicorn workers (e.g., (2 * CPU_CORES) + 1). Default is 1.")
    auth_token: Optional[str] = Field(default=None, validation_alias="APP_AUTH_TOKEN", description="Optional API authentication token for MCP endpoint.")
    # debug: bool = False

    # MCP Transport Configuration
    mcp_transport_type: str = Field(
        default="http",
        validation_alias="MCP_TRANSPORT_TYPE",
        description="MCP transport type: http, stdio, or both"
    )
    mcp_stdio_enabled: bool = Field(
        default=True,
        validation_alias="MCP_STDIO_ENABLED",
        description="Enable STDIO transport"
    )
    mcp_http_enabled: bool = Field(
        default=True,
        validation_alias="MCP_HTTP_ENABLED",
        description="Enable HTTP transport"
    )

class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    asr_got: ASRGoTConfig = Field(default_factory=ASRGoTConfig)
    mcp_settings: MCPSettings = Field(default_factory=MCPSettings)
    claude_api: Optional[ClaudeAPIConfig] = None  # Optional section
    knowledge_domains: list[KnowledgeDomain] = Field(default_factory=list)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",  # e.g., APP__HOST to override app.host
        # Add other sources if needed, e.g., .env file
        # env_file = '.env',
        # env_file_encoding = 'utf-8',
        extra="ignore",  # Ignore extra fields from YAML if any
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate entire configuration against JSON Schema
        config_dict = self.model_dump(by_alias=True)
        validate_config_schema(config_dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings], # Re-added settings_cls
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        
        """
        Customizes the order and sources from which Pydantic loads settings.

        Inserts a YAML-based settings source, using a pre-loaded configuration dictionary,
        into the settings source priority after dotenv settings. This allows settings to
        be loaded from initialization, environment variables, dotenv files, the YAML file,
        and file secrets, in that order.
        """
        class YamlConfigSettingsSource(PydanticBaseSettingsSource):
            def __init__(self, settings_cls: Type[BaseSettings]):
                super().__init__(settings_cls)
                self._yaml_config = yaml_config  # Use pre-loaded config

            def get_field_value(self, field: Field, field_name: str) -> tuple[Any, str] | None:
                # Not using field-level logic from YAML here
                return None

            def __call__(self) -> dict[str, Any]:
                return self._yaml_config
            
            def prepare_field_value(self, field_name: str, field: Field, value: Any, value_is_complex: bool) -> Any:
                return value

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),  # Pass settings_cls here
            file_secret_settings,
        )

# Global settings instance, to be imported by other modules
settings = Settings()

# Example of how to access settings:
# from src.asr_got_reimagined.config import settings
# print(settings.app.name)
# print(settings.asr_got.default_parameters.initial_confidence)