# Getting Started with NexusMind

This guide will help you get NexusMind up and running, whether for local development or Docker-based deployment.

## Deployment Prerequisites

Before running NexusMind (either locally or via Docker if not using the provided `docker-compose.prod.yml` which includes Neo4j), ensure you have:

-   **A running Neo4j Instance**: NexusMind requires a connection to a Neo4j graph database.
    -   **APOC Library**: The Neo4j instance **must** have the APOC (Awesome Procedures On Cypher) library installed. Many Cypher queries rely on APOC procedures. See the [official APOC website](https://neo4j.com/labs/apoc/installation/) for installation.
    -   **Indexing**: For optimal performance, ensure appropriate Neo4j indexes are created. See [Neo4j Indexing Strategy](neo4j_indexing.md) for details. <!-- TODO: Ensure this link works after file moves -->

    *Note: The provided `docker-compose.yml` (for development) and `docker-compose.prod.yml` (for production) already include a Neo4j service with the APOC library pre-configured, satisfying this requirement when using Docker Compose.*

## Core Prerequisites

- **Python 3.11+** (as specified in `pyproject.toml`)
- **[Poetry](https://python-poetry.org/docs/#installation)**: For dependency management.
- **[Docker](https://www.docker.com/get-started)** and **[Docker Compose](https://docs.docker.com/compose/install/)**: For containerized deployment.

## Installation and Setup (Local Development)

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SaptaDey/NexusMind.git # Adjust if your fork/repo is different
    cd NexusMind
    ```

2.  **Install dependencies using Poetry**:
    ```bash
    poetry install --with dev # Installs main and development dependencies
    ```
    This creates a virtual environment (if one isn't already activated) and installs all necessary packages.

3.  **Activate the virtual environment**:
    ```bash
    poetry shell
    ```

4.  **Configure Neo4j Connection (Critical)**:
    NexusMind connects to Neo4j using environment variables. See the [Configuration Guide](configuration.md#neo4j-database-configuration) for detailed instructions on setting `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE`. For local development, using a `.env` file is recommended.

5.  **Application Configuration**:
    Other application settings are in `config/settings.yaml`. You can review and customize this file if needed. See the [Configuration Guide](configuration.md#application-settings) for more details.

6.  **Run the development server**:
    Ensure your Neo4j instance is running and accessible with the configured credentials.
    
    If you haven't set `NEO4J_PASSWORD` in a `.env` file, you might need to provide it directly (though `.env` is preferred):
    ```bash
    NEO4J_PASSWORD="your_neo4j_password" poetry run uvicorn src.asr_got_reimagined.main:app --reload --host 0.0.0.0 --port 8000
    ```
    If using a `.env` file (recommended for all Neo4j credentials):
    ```bash
    poetry run uvicorn src.asr_got_reimagined.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000` (or the port you configured, e.g., via `APP_PORT` in your `.env` file).

## Docker Deployment

NexusMind is designed to be easily deployed using Docker.

<!-- Mermaid diagram for Docker deployment can be included here if desired -->
<!-- ```mermaid
graph TB
    subgraph "Development Environment"
        A[👨‍💻 Developer] --> B[🐳 Docker Compose]
    end
    
    subgraph "Container Orchestration"
        B --> C[📦 NexusMind Container]
        B --> D[📊 Monitoring Container] # Placeholder if you add monitoring
        B --> E[🗄️ Database Container]
    end
    
    subgraph "NexusMind Application"
        C --> F[⚡ FastAPI Server]
        F --> G[🧠 ASR-GoT Engine]
        F --> H[🔌 MCP Protocol]
    end
    
    subgraph "External Integrations"
        H --> I[🤖 Claude Desktop]
        H --> J[🔗 Other AI Clients]
    end
``` -->

### 1. Quick Start with Docker Compose (Recommended for Development)

The `docker-compose.yml` file is pre-configured for local development and includes the NexusMind API service and a Neo4j service with APOC.

   ```bash
   # Build and run all services
   docker-compose up --build
   
   # For detached mode (background)
   docker-compose up --build -d
   
   # View logs for the API service
   docker-compose logs -f nexusmind-api
   ```
   Ensure you have a `.env` file with your `NEO4J_PASSWORD` (and other Neo4j settings if not using defaults) as `docker-compose.yml` is set up to use it.

### 2. Individual Docker Container (Manual Run)

   ```bash
   # Build the image
   docker build -t nexusmind:latest .
   
   # Run the container (ensure NEO4J_* env vars are set, e.g., via --env-file)
   docker run -d \
     -p 8000:8000 \
     --env-file .env \
     -v /path/to/your/local/config:/app/config \
     nexusmind:latest
   ```
   Replace `/path/to/your/local/config` with the actual path to your *custom* configuration directory if you need to override the defaults baked into the image. See the [Configuration Guide](configuration.md#docker-configuration-override) for more details.

### 3. Production Deployment

For production, use the `docker-compose.prod.yml` file:
   ```bash
   # Ensure all required environment variables (especially NEO4J_PASSWORD) are set
   # in your production environment or a secure .env file used by the compose file.
   docker-compose -f docker-compose.prod.yml up --build -d
   ```
   Refer to the [Production Configuration section in the Configuration Guide](configuration.md#production-environment-variables) for details on required environment variables.

### Notes on Specific Deployment Platforms

-   **Smithery.ai**: Deployment to the Smithery.ai platform typically involves using the provided Docker image directly.
    *   Consult Smithery.ai's specific documentation for instructions on deploying custom Docker images.
    *   **Port Configuration**: Ensure that the platform is configured to expose port 8000 (or the port configured via `APP_PORT` if overridden) for the NexusMind container.
    *   **Health Checks**: The NexusMind Docker image includes a `HEALTHCHECK` instruction verifying `/health`. Ensure Smithery.ai is configured to use this endpoint.
    *   **Environment Variables**: Configure all necessary environment variables (especially `NEO4J_PASSWORD` and other Neo4j connection details) through the Smithery.ai platform.

### Accessing the Services (after deployment)

- **API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **MCP Endpoint**: `http://localhost:8000/mcp` (or the relevant service address if deployed)

Navigate to these URLs in your browser or API client after starting the application.
