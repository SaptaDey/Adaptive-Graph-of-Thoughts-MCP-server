# Integrating with Claude Desktop

NexusMind is designed to work with Claude Desktop through the Model Context Protocol (MCP). This allows Claude 3.7 Sonnet to use the Graph of Thoughts reasoning process when answering complex scientific questions.

## Getting Started

1. **Start the NexusMind Server**:
   ```
   docker-compose up -d
   ```

2. **Run the Setup Helper**:
   ```
   ./setup_claude_connection.py
   ```
   This script will:
   - Check if the server is running
   - Test the MCP endpoint
   - Verify the configuration file
   - Display instructions for connecting

3. **Connect Claude Desktop**:
   - Open Claude Desktop
   - Go to Settings → Integrations
   - Add a new integration by importing the `config/claude_mcp_config.json` file

4. **Test the Integration**:
   - Start a new conversation in Claude Desktop
   - Enable the NexusMind integration
   - Ask a scientific question
   - Claude will use the Graph of Thoughts process to analyze and answer your question

## What Happens Behind the Scenes

When you ask a question with the NexusMind integration enabled:

1. Claude Desktop sends your question to the NexusMind server
2. The server processes your query through several stages:
   - Initialization and understanding
   - Question decomposition
   - Hypothesis generation
   - Evidence gathering
   - Pruning and merging hypotheses
   - Creating coherent thinking subgraphs
   - Composing the final answer 
   - Reflecting on confidence
3. The server returns:
   - A comprehensive answer
   - A reasoning trace showing the Graph of Thoughts process
   - The graph state (for visualization or further analysis)
4. Claude Desktop presents this information to you

## Configuration Options

The `config/claude_mcp_config.json` file defines how NexusMind registers itself with MCP clients like Claude Desktop. It typically includes:
-   `name`: A short name for the integration (e.g., "nexusmind").
-   `description`: A brief description of what NexusMind does.
-   `version`: The version of NexusMind.
-   `endpoints`: The URL for the MCP endpoint (e.g., `http://localhost:8000/mcp`).
-   `capabilities`: A list of capabilities the server offers (e.g., "scientific_reasoning").

Runtime behaviors, such as whether to include the reasoning trace or full graph state in responses, are typically controlled by parameters sent by the client in the `asr_got.query` request (see the `MCPASRGoTQueryParams` schema in `src/asr_got_reimagined/api/schemas.py` for details like `include_reasoning_trace` and `include_graph_state`). These are not configured in `claude_mcp_config.json`.

## Status Page

A status page is available to check the connection and test queries:
```
open status.html
```

For more detailed instructions, see the [Claude Desktop Integration Guide](claude_desktop_integration.md).
