# Configuration

NexusMind's behavior can be configured through a combination of YAML files and environment variables. Pydantic is used for settings management, allowing for type validation and clear defaults.

## Main Application Settings (`config/settings.yaml`)

The primary configuration file is `config/settings.yaml`. This file defines settings for the application, the ASR-GoT pipeline, MCP server behavior, and optional integrations.

**Structure Overview:**

The settings are defined by Pydantic models in `src/asr_got_reimagined/config.py`. Environment variables can override values from `settings.yaml`. For nested structures, use double underscores for environment variables (e.g., `APP__PORT=8001` overrides `app.port`).

```yaml
# config/settings.yaml (Illustrative Snippet)

# Core application settings (corresponds to AppSettings in config.py)
app:
  name: "NexusMind"
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
      module_path: "asr_got_reimagined.domain.stages.InitializationStage"
    # ... other stages ...
# MCP Server Settings (corresponds to MCPSettings in config.py)
mcp_settings:
  protocol_version: "2024-11-05"
  server_name: "NexusMind MCP Server"
  # ... other mcp_settings fields ...

# Optional Claude API integration (corresponds to ClaudeAPIConfig in config.py)
# claude_api:
#   api_key: "env_var:CLAUDE_API_KEY" # Recommended
#   default_model: "claude-3-opus-20240229"
#   # ... other claude_api fields ...

# Knowledge Domains (list of KnowledgeDomain models)
# knowledge_domains:
#   - name: "Immunology"
#     keywords: ["immune system", "antibodies"]
#     description: "Study of the immune system."
```

Refer to `config/config.schema.json` for the full schema and `src/asr_got_reimagined/config.py` for the Pydantic models defining these settings.

## Neo4j Database Configuration (Critical)

Connection to your Neo4j instance is managed via environment variables. These settings are defined in the `Neo4jSettings` model within `src/asr_got_reimagined/domain/services/neo4j_utils.py`.

*   **`NEO4J_URI`**: The URI for your Neo4j instance.
    *   Default: `neo4j://localhost:7687`
    *   Example for AuraDB: `neo4j+s://your-neo4j-aura-instance.databases.neo4j.io`
*   **`NEO4J_USER`**: The Neo4j username.
    *   Default: `neo4j`
*   **`NEO4J_PASSWORD`**: (Required) The password for your Neo4j database.
    *   **This variable is mandatory and has no default.** The application will not start if this is not set.
    *   **Security:** For production, always set this as a secure environment variable provided by your deployment platform. Do not hardcode it in configuration files or commit it to version control.
*   **`NEO4J_DATABASE`**: The Neo4j database name to use.
    *   Default: `neo4j`

**Local Development using `.env` file:**

For local development, you can place these variables in a `.env` file in the project root. This file is automatically loaded by Pydantic if `python-dotenv` is installed (it's a dependency).

```env
# .env example
NEO4J_URI="neo4j://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_local_neo4j_password" # Replace with your actual password
# NEO4J_DATABASE="neo4j" # Optional if using default

# You can also set other application environment variables here
# APP__LOG_LEVEL="DEBUG"
# APP__PORT="8001"
# APP__AUTH_TOKEN="your-secret-dev-token"
**Important**: Ensure `.env` is listed in your `.gitignore` file to prevent accidental commits of credentials.

## Production Environment Variables

When deploying NexusMind to a production environment (e.g., Smithery.ai, Heroku, AWS, Azure, GCP), it's crucial to manage configuration securely using the platform's environment variable or secrets management system.

**Essential Production Variables:**

*   **`NEO4J_PASSWORD`**: (Required) The password for your production Neo4j database.
*   **`NEO4J_URI`**: The URI of your production Neo4j instance.
*   **`NEO4J_USER`**: The username for your production Neo4j database.
*   `NEO4J_DATABASE`: (Optional, defaults to `neo4j`) The specific database name.
*   `APP__UVICORN_RELOAD="False"`: Disable Uvicorn's auto-reload feature.
*   `APP__UVICORN_WORKERS="<number_of_workers>"`: Set to an appropriate number based on your server resources (e.g., `4`).
*   `LOG_LEVEL="INFO"` (or `APP__LOG_LEVEL="INFO"`): Set a less verbose log level for production.
*   `APP__CORS_ALLOWED_ORIGINS_STR="<your_frontend_domain_here>"`: Configure allowed CORS origins if your API is accessed from a specific frontend.
*   `APP__AUTH_TOKEN="<your_secure_random_token>"`: If MCP endpoint authentication is desired, set this to a strong, randomly generated token.

**Security Notes on Passwords & Secrets:**

*   **Never hardcode `NEO4J_PASSWORD` or other secrets** (like `APP_AUTH_TOKEN`) directly in `config/settings.yaml` or any committed files.
*   Always use environment variables for sensitive data, configured through your deployment platform's secure mechanisms.

## MCP Client Configuration (`config/claude_mcp_config.json`)

This file is used when registering NexusMind as an external tool with an MCP client like Claude Desktop. It describes the capabilities and endpoint of your NexusMind instance to the client.

 {
   "endpoints": {
     "mcp": "http://localhost:8000/mcp"
   }
 }
  "capabilities": [
    "scientific_reasoning",
    "graph_analysis",
    "confidence_assessment",
    "bias_detection"
  ]
}
```
When deploying, ensure the `endpoints.mcp` URL in this file (or a version of it used for registration) points to the publicly accessible URL of your deployed NexusMind MCP endpoint.

## Docker Configuration Override

When running NexusMind using Docker (not Docker Compose), the image includes a default set of configurations from the `config/` directory. To use a custom `settings.yaml` or other configuration files:

1.  Prepare your custom configuration files in a local directory (e.g., `./my_custom_config`).
2.  Mount this directory to `/app/config` in the container using the `-v` flag:
    ```bash
    docker run -d \
      -p 8000:8000 \
      -v /path/to/your/my_custom_config:/app/config \
      --env-file .env \
      nexusmind:latest 
    ```
    Replace `/path/to/your/my_custom_config` with the actual path to your configuration directory.
    Ensure your custom directory contains all necessary files (e.g., `settings.yaml`).

The development `docker-compose.yml` already mounts the local `./config` directory. For production `docker-compose.prod.yml`, environment variables are the primary way to manage configuration, as code/config is baked into the image.
