# Adaptive Graph of Thoughts - Desktop Extension (DXT)

This is a Desktop Extension (DXT) for the Adaptive Graph of Thoughts scientific reasoning framework. It provides a standardized MCP (Model Context Protocol) interface that can be easily integrated with AI desktop applications like Claude Desktop.

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+**: Required for the DXT MCP server
- **Python 3.11+**: Required for the backend Adaptive Graph of Thoughts server
- **Neo4j Database**: Required for graph-based reasoning (with APOC library)

### Installation

1. **Ensure the Python backend is running**:
   ```bash
   # In the root directory of this project
   poetry install
   poetry run uvicorn src.adaptive_graph_of_thoughts.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Install DXT server dependencies**:
   ```bash
   cd server
   npm install
   ```

3. **Configure environment variables** (optional):
   ```bash
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USERNAME="neo4j"
   export NEO4J_PASSWORD="your_password"
   export OPENAI_API_KEY="your_openai_key"  # Optional
   export ANTHROPIC_API_KEY="your_anthropic_key"  # Optional
   export LOG_LEVEL="INFO"  # DEBUG, INFO, WARN, ERROR
   ```

### Testing the Extension

You can test the DXT server directly:

```bash
cd server
node index.js
```

Then send MCP messages via stdin:

```json
{"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "0.1.0", "capabilities": {}}, "id": 1}
{"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
```

## 🔧 Configuration

The extension supports the following configuration options via environment variables:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEO4J_URI` | Neo4j database connection URI | `bolt://localhost:7687` | Yes |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` | Yes |
| `NEO4J_PASSWORD` | Neo4j password | - | Yes |
| `OPENAI_API_KEY` | OpenAI API key for enhanced reasoning | - | No |
| `ANTHROPIC_API_KEY` | Anthropic API key for enhanced reasoning | - | No |
| `PUBMED_API_KEY` | PubMed API key for evidence gathering | - | No |
| `ENABLE_EXTERNAL_SEARCH` | Enable external database search | `true` | No |
| `MAX_REASONING_DEPTH` | Maximum graph traversal depth | `5` | No |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for hypothesis acceptance | `0.7` | No |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARN, ERROR) | `INFO` | No |

## 🛠️ Available Tools

### 1. scientific_reasoning_query
Advanced scientific reasoning with graph analysis using the ASR-GoT framework.

**Parameters:**
- `query` (string, required): The scientific question or research query to analyze
- `parameters` (object, optional):
  - `include_reasoning_trace` (boolean): Include detailed reasoning steps (default: true)
  - `include_graph_state` (boolean): Include graph state information (default: false)
  - `max_depth` (number): Override maximum reasoning depth (1-10)
  - `confidence_threshold` (number): Override confidence threshold (0.0-1.0)

**Example:**
```json
{
  "name": "scientific_reasoning_query",
  "arguments": {
    "query": "What is the relationship between microbiome diversity and cancer progression?",
    "parameters": {
      "include_reasoning_trace": true,
      "max_depth": 5
    }
  }
}
```

### 2. analyze_research_hypothesis
Hypothesis evaluation with confidence scoring and evidence integration.

**Parameters:**
- `hypothesis` (string, required): The research hypothesis to analyze
- `context` (string, optional): Additional context or background information
- `evidence_sources` (array of strings, optional): Specific evidence sources to consider

### 3. explore_scientific_relationships
Concept relationship mapping through graph-based analysis.

**Parameters:**
- `concepts` (array of strings, required): Scientific concepts to explore relationships between
- `relationship_types` (array of strings, optional): Specific types of relationships to focus on
- `depth` (number): Depth of relationship exploration (1-5, default: 3)

### 4. validate_scientific_claims
Evidence-based claim validation with external database integration.

**Parameters:**
- `claims` (array of strings, required): Scientific claims to validate
- `evidence_requirement` (string): Level of evidence required ('low', 'medium', 'high')
- `sources` (array of strings, optional): Preferred evidence sources

## 📋 Available Prompts

### 1. analyze_research_question
Generate comprehensive analysis of a scientific research question.

**Arguments:**
- `research_question` (required): The research question to analyze
- `domain` (optional): Scientific domain or field

### 2. hypothesis_generator
Generate and evaluate multiple hypotheses for a given scientific problem.

**Arguments:**
- `problem_statement` (required): The scientific problem to generate hypotheses for
- `constraints` (optional): Any constraints or limitations to consider

## 🔍 Integration Examples

### Claude Desktop Configuration

Add this to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "adaptive-graph-of-thoughts": {
      "command": "node",
      "args": ["server/index.js"],
      "cwd": "/path/to/Adaptive-Graph-of-Thoughts-MCP-server",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your_password",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Generic MCP Client Configuration

```yaml
servers:
  adaptive-graph-of-thoughts:
    command: node
    args: ["server/index.js"]
    cwd: "/path/to/Adaptive-Graph-of-Thoughts-MCP-server"
    env:
      NEO4J_URI: "bolt://localhost:7687"
      NEO4J_USERNAME: "neo4j"
      NEO4J_PASSWORD: "your_password"
```

## 📊 Logging and Debugging

The extension provides comprehensive logging capabilities:

- **File Logging**: Logs are written to `server/logs/dxt-server.log`
- **Log Rotation**: Automatic log rotation when files exceed 10MB
- **Structured Logging**: JSON-formatted logs with timestamps and context
- **Request Tracing**: Each request gets a unique ID for tracing

### Log Levels

- **ERROR**: Critical errors and failures
- **WARN**: Warnings and recoverable issues
- **INFO**: General operational information
- **DEBUG**: Detailed debugging information

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure the Python server is running on port 8000
   - Check that the `/health` endpoint returns 200 OK
   - Verify firewall settings allow local connections

2. **Neo4j Connection Issues**
   - Verify Neo4j is running and accessible
   - Check Neo4j credentials in environment variables
   - Ensure APOC library is installed in Neo4j

3. **Tool Execution Timeouts**
   - Complex queries may take longer to process
   - Monitor the Python server logs for bottlenecks
   - Consider adjusting the `timeout` parameter

4. **Permission Errors**
   - Ensure the Node.js process has write access to the logs directory
   - Check file permissions on the server directory

### Getting Help

- Check the log files in `server/logs/` for detailed error information
- Each error includes a unique ID for easier tracking
- Monitor both the DXT server logs and the Python backend logs

## 🔒 Security Considerations

- **API Keys**: Store sensitive API keys in environment variables, not in configuration files
- **Network Access**: The extension only connects to localhost by default
- **Input Validation**: All tool inputs are validated using Zod schemas
- **Error Handling**: Errors are logged but sensitive information is redacted

## 📦 Building and Packaging

To create a DXT package:

```bash
# Ensure all dependencies are installed
cd server && npm install

# Create a zip archive of the extension
cd ..
zip -r adaptive-graph-of-thoughts-dxt.zip manifest.json server/ assets/ -x "server/node_modules/.cache/*" "server/logs/*"
```

## 🚀 Development

### Running in Development Mode

```bash
cd server
NODE_ENV=development LOG_LEVEL=DEBUG node --inspect index.js
```

This enables:
- Debug logging
- Node.js inspector for debugging
- Detailed request/response logging

### Testing

```bash
cd server
npm test  # Run unit tests (if available)
```

## 📄 License

This extension inherits the Apache License 2.0 from the main Adaptive Graph of Thoughts project.