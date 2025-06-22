# MCP Integration Guide

## Overview

The Adaptive Graph of Thoughts MCP server provides advanced scientific reasoning capabilities through the Model Context Protocol (MCP). This guide covers integration with various MCP clients.

## Available Tools

### 1. scientific_reasoning_query
Perform advanced scientific reasoning using graph-based analysis.

**Parameters:**
- `query` (required): Scientific question to analyze
- `include_reasoning_trace` (optional): Include detailed reasoning steps
- `include_graph_state` (optional): Include graph visualization data
- `max_nodes_in_response_graph` (optional): Limit graph size in response
- `output_detail_level` (optional): "summary" or "detailed"
- `session_id` (optional): Session tracking identifier

**Example:**
```json
{
  "query": "What is the relationship between microbiome diversity and cancer progression?",
  "include_reasoning_trace": true,
  "include_graph_state": true
}
```

2. analyze_research_hypothesis
Analyze and evaluate research hypotheses with confidence scoring.

Parameters:

hypothesis (required): Research hypothesis to analyze
research_domain (optional): Scientific field context
evidence_sources (optional): External sources to query
Example:
```
{
  "hypothesis": "Increased gut microbiome diversity correlates with better cancer treatment outcomes",
  "research_domain": "oncology",
  "evidence_sources": ["pubmed", "google_scholar"]
}
```
##3. explore_scientific_relationships
Explore relationships between scientific concepts using graph analysis.

###Parameters:
  primary_concept (required): Main concept to explore
  related_concepts (optional): Related concepts to analyze
  relationship_types (optional): Types of relationships to explore
  depth_level (optional): Analysis depth (1-5)

##4. validate_scientific_claims
Validate scientific claims against existing evidence.

###Parameters:
  claim (required): Scientific claim to validate
  evidence_threshold (optional): Minimum confidence threshold
  validation_criteria (optional): Validation criteria to apply
  include_counterevidence (optional): Include contradictory evidence

# Client Integration
## Claude Desktop
### 1. Install the server:

```
git clone https://github.com/SaptaDey/Adaptive-Graph-of-Thoughts-MCP-server.git
cd Adaptive-Graph-of-Thoughts-MCP-server
poetry install
```
### 2. Configure Claude Desktop:
```
./scripts/setup_mcp_client.sh claude-desktop
```
### 3. Add to Claude Desktop MCP settings: Use the generated configuration file.

##VS Code
### 1. Install MCP extension in VS Code

### 2. Configure the server:
```
./scripts/setup_mcp_client.sh vscode
```
### 3. Add to VS Code settings.json: Use the generated configuration.

## Docker Deployment
### For containerized deployment:

```
# Development
docker-compose up --build
```
```
# Production
docker-compose -f docker-compose.prod.yml up -d
```


## Configuration Options
### Environment Variables
        NEO4J_URI: Neo4j connection string
        NEO4J_USER: Neo4j username
        NEO4J_PASSWORD: Neo4j password
        NEO4J_DATABASE: Neo4j database name
        MCP_TRANSPORT_TYPE: Transport mode ("stdio" or "http")
        LOG_LEVEL: Logging level
        ENABLE_EVIDENCE_SOURCES: Enable external evidence gathering
  
