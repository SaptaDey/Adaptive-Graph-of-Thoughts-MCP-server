# Installation Guide for MCP Clients

## Prerequisites

- Python 3.11+
- Neo4j database with APOC library
- Poetry for dependency management

## Quick Setup

### 1. Clone and Install
```bash
git clone https://github.com/SaptaDey/Adaptive-Graph-of-Thoughts-MCP-server.git
cd Adaptive-Graph-of-Thoughts-MCP-server
poetry install
```

### 2. Configure Neo4j

# Install Neo4j with APOC library
# Update config/settings.yaml with your Neo4j credentials

3. Environment Variables

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export NEO4J_DATABASE="neo4j"
```

To load secrets from AWS, GCP or Vault set `SECRETS_PROVIDER` and provide
`<VAR>_SECRET_NAME` environment variables as needed.

##Client-Specific Setup
###Claude Desktop
      1. Copy configuration from config/client_configurations/claude_desktop.json
      2. Update paths and credentials
      3. Add to your Claude Desktop MCP settings
###VS Code
      1. Copy configuration from config/client_configurations/vscode.json
      2. Add to your VS Code settings.json
      3. Install MCP extension if required

##Docker Setup (Recommended)

```bash
# Full setup including Neo4j
docker-compose up --build

# Production deployment
docker-compose -f docker-compose.prod.yml up --build -d
```

### Kubernetes (Helm)

For production clusters, a simple Helm chart is provided under `helm/agot-server`.

```bash
helm install agot helm/agot-server
```

Adjust `values.yaml` for your image repository and resource requirements.
##Verification
###Test your setup:
```bash
poetry run python src/adaptive_graph_of_thoughts/main.py
```

Visit http://localhost:8000/health to verify the server is running.
