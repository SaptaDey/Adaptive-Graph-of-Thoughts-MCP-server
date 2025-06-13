# Configuration

Adaptive Graph of Thoughts's behavior can be configured through a combination of YAML files and environment variables. Pydantic is used for settings management, allowing for type validation and clear defaults.

## Main Application Settings (`config/settings.yaml`)

The primary configuration file is `config/settings.yaml`. This file defines settings for the application, the ASR-GoT pipeline, MCP server behavior, and optional integrations.

**Structure Overview:**

The settings are defined by Pydantic models in `src/adaptive_graph_of_thoughts/config.py`. Environment variables can override values from `settings.yaml`. For nested structures, use double underscores for environment variables (e.g., `APP__PORT=8001` overrides `app.port`).

```yaml
# config/settings.yaml (Illustrative Snippet)

# Core application settings (corresponds to AppSettings in config.py)
app:
  name: "Adaptive Graph of Thoughts"
  version: "0.1.0"
  log_level: "INFO" # Env: APP__LOG_LEVEL or LOG_LEVEL
  host: "0.0.0.0"   # Env: APP__HOST
  port: 8000        # Env: APP__PORT
  
  uvicorn_reload: true # Env: APP__UVICORN_RELOAD (False for production)
  uvicorn_workers: 1   # Env: APP__UVICORN_WORKERS (e.g., 4 for production)
  cors_allowed_origins_str: "*" # Env: APP__CORS_ALLOWED_ORIGINS_STR
  auth_token: null # Optional API auth token. Env: APP__AUTH_TOKEN

  mcp_transport_type: "http" # http, stdio, or both. Env: MCP_TRANSPORT_TYPE
  mcp_stdio_enabled: true    # Env: MCP_STDIO_ENABLED
  mcp_http_enabled: true     # Env: MCP_HTTP_ENABLED

# ASR-GoT Framework settings (corresponds to ASRGoTConfig in config.py)
asr_got:
  default_parameters: # Corresponds to ASRGoTDefaultParams
    initial_confidence: [0.9, 0.9, 0.9, 0.9]
    pruning_confidence_threshold: 0.2
    # ... other ASRGoTDefaultParams fields ...
  
  layers:
    root_layer:
      description: "The initial layer where the query is processed."
    # ... other layer definitions ...

  pipeline_stages:
    - name: "Initialization"
      module_path: "adaptive_graph_of_thoughts.domain.stages.InitializationStage"
    # ... other stages ...
# MCP Server Settings (corresponds to MCPSettings in config.py)
mcp_settings:
  protocol_version: "2024-11-05"
  server_name: "Adaptive Graph of Thoughts MCP Server"
  # ... other mcp_settings fields ...

# Optional Claude API integration (corresponds to ClaudeAPIConfig in config.py)
claude_api: # Ensure this is uncommented if you intend to use it
  api_key: "env_var:CLAUDE_API_KEY" # Recommended. Set CLAUDE_API_KEY in your environment.
  default_model: "claude-3-opus-20240229"
  # ... other claude_api fields ...

# Optional PubMed integration (corresponds to PubMedConfig in config.py)
pubmed:
  api_key: "env_var:PUBMED_API_KEY" # Set PUBMED_API_KEY in your environment.

# Optional Exa Search integration (corresponds to ExaSearchConfig in config.py)
exa_search:
  api_key: "env_var:EXA_SEARCH_API" # Set EXA_SEARCH_API in your environment.

# Optional OpenAI API integration (corresponds to OpenAIAPIConfig in config.py)
openai_api:
  api_key: "env_var:OPENAI_API_KEY" # Set OPENAI_API_KEY in your environment.

# Neo4j Database Configuration (corresponds to Neo4jConfig in config.py)
neo4j:
  uri: "env_var:NEO4J_URI"             # Set NEO4J_URI (e.g., "neo4j://localhost:7687")
  username: "env_var:NEO4J_USERNAME"   # Set NEO4J_USERNAME (e.g., "neo4j")
  password: "env_var:NEO4J_PASSWORD"   # Set NEO4J_PASSWORD
  database: "neo4j"                    # Default, or set NEO4J_DATABASE

# Knowledge Domains (list of KnowledgeDomain models)
# knowledge_domains:
#   - name: "Immunology"
#     keywords: ["immune system", "antibodies"]
#     description: "Study of the immune system."
```

Refer to `config/config.schema.json` for the full schema and `src/adaptive_graph_of_thoughts/config.py` for the Pydantic models defining these settings.

## Neo4j Database Configuration (Critical)

Connection to your Neo4j instance is configured within the main `config/settings.yaml` file under the `neo4j` key and loaded via the global `settings.neo4j` object from `src/adaptive_graph_of_thoughts/config.py`. For production and sensitive data, these values should be set using environment variables.

The application expects the following environment variables to configure Neo4j:

*   **`NEO4J_URI`**: The URI for your Neo4j instance.
    *   Example: `neo4j://localhost:7687`
    *   Example for AuraDB: `neo4j+s://your-neo4j-aura-instance.databases.neo4j.io`
*   **`NEO4J_USERNAME`**: The Neo4j username.
    *   Example: `neo4j`
*   **`NEO4J_PASSWORD`**: (Required) The password for your Neo4j database.
!!! critical
    *   **This variable is mandatory if not set directly in `settings.yaml` (which is not recommended for passwords).**
    *   **Security:** For production, always set this as a secure environment variable provided by your deployment platform. Do not hardcode it in configuration files or commit it to version control.
*   **`NEO4J_DATABASE`**: The Neo4j database name to use.
    *   Default: `neo4j` (as set in `Neo4jConfig` and `settings.yaml`)
    *   Example: `my_custom_graph_db`

These variables map to the `neo4j` section in your `settings.yaml` like so:
```yaml
neo4j:
  uri: "env_var:NEO4J_URI"
  username: "env_var:NEO4J_USERNAME"
  password: "env_var:NEO4J_PASSWORD"
  database: "env_var:NEO4J_DATABASE" # or a fixed default like "neo4j"
```

**Local Development using `.env` file:**

For local development, you can place these (and other) environment variables in a `.env` file in the project root. This file is automatically loaded by Pydantic if `python-dotenv` is installed (it's a dependency).

```env
# .env example
NEO4J_URI="neo4j://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_local_neo4j_password" # Replace with your actual password
# NEO4J_DATABASE="neo4j" # Optional if using default, or set your specific DB
# CLAUDE_API_KEY="your_claude_key_here" # If using Claude
# PUBMED_API_KEY="your_pubmed_key_here" # If using PubMed
# EXA_SEARCH_API="your_exa_key_here"   # If using Exa Search
# OPENAI_API_KEY="your_openai_key_here" # If using OpenAI

# You can also set other application environment variables here
# APP__LOG_LEVEL="DEBUG"
# APP__PORT="8001"
# APP__AUTH_TOKEN="your-secret-dev-token"

!!! warning "Ensure .env is in .gitignore"
    Ensure `.env` is listed in your `.gitignore` file to prevent accidental commits of credentials.

## Production Environment Variables

When deploying Adaptive Graph of Thoughts to a production environment (e.g., Smithery.ai, Heroku, AWS, Azure, GCP), it's crucial to manage configuration securely using the platform's environment variable or secrets management system.

**Essential Production Variables:**

*   **`NEO4J_PASSWORD`**: (Required) The password for your production Neo4j database.
*   **`NEO4J_URI`**: The URI of your production Neo4j instance.
*   **`NEO4J_USERNAME`**: The username for your production Neo4j database.
*   `NEO4J_DATABASE`: (Optional, defaults to `neo4j`) The specific database name.
*   **`CLAUDE_API_KEY`**: (Optional) API key if using direct Claude integration.
*   **`PUBMED_API_KEY`**: (Optional) API key for PubMed integration (if implemented).
*   **`EXA_SEARCH_API`**: (Optional) API key for Exa Search integration (if implemented).
*   **`OPENAI_API_KEY`**: (Optional) API key for OpenAI integration (if implemented).
*   `APP__UVICORN_RELOAD="False"`: Disable Uvicorn's auto-reload feature.
*   `APP__UVICORN_WORKERS="<number_of_workers>"`: Set to an appropriate number based on your server resources (e.g., `4`).
*   `LOG_LEVEL="INFO"` (or `APP__LOG_LEVEL="INFO"`): Set a less verbose log level for production.
*   `APP__CORS_ALLOWED_ORIGINS_STR="<your_frontend_domain_here>"`: Configure allowed CORS origins if your API is accessed from a specific frontend.
*   `APP__AUTH_TOKEN="<your_secure_random_token>"`: If MCP endpoint authentication is desired, set this to a strong, randomly generated token.

!!! danger "Security Notes on Passwords & Secrets"
    *   **Never hardcode `NEO4J_PASSWORD` or other API keys/secrets** (like `APP_AUTH_TOKEN`, `CLAUDE_API_KEY`, etc.) directly in `config/settings.yaml` or any committed files.
    *   Always use environment variables for sensitive data, configured through your deployment platform's secure mechanisms.

## MCP Client Configuration (`config/claude_mcp_config.json`)

This file is used when registering Adaptive Graph of Thoughts as an external tool with an MCP client like Claude Desktop. It describes the capabilities and endpoint of your Adaptive Graph of Thoughts instance to the client.

```json
{
  "endpoints": {
    "mcp": "http://localhost:8000/mcp"
  },
  "capabilities": [
    "scientific_reasoning",
    "graph_analysis",
    "confidence_assessment",
    "bias_detection"
  ]
}
```
When deploying, ensure the `endpoints.mcp` URL in this file (or a version of it used for registration) points to the publicly accessible URL of your deployed Adaptive Graph of Thoughts MCP endpoint.

## Docker Configuration Override

When running Adaptive Graph of Thoughts using Docker (not Docker Compose), the image includes a default set of configurations from the `config/` directory. To use a custom `settings.yaml` or other configuration files:

1.  Prepare your custom configuration files in a local directory (e.g., `./my_custom_config`).
2.  Mount this directory to `/app/config` in the container using the `-v` flag:
    ```bash
    docker run -d \
      -p 8000:8000 \
      -v /path/to/your/my_custom_config:/app/config \
      --env-file .env \
      adaptive-graph-of-thoughts:latest 
    ```
    Replace `/path/to/your/my_custom_config` with the actual path to your configuration directory.
    Ensure your custom directory contains all necessary files (e.g., `settings.yaml`).

The development `docker-compose.yml` already mounts the local `./config` directory. For production `docker-compose.prod.yml`, environment variables are the primary way to manage configuration, as code/config is baked into the image.
