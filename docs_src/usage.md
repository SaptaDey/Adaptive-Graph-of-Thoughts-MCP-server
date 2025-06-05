# Using NexusMind

This section describes how to interact with the NexusMind application, primarily through its API, and details on session handling and testing.

## API Endpoints

NexusMind exposes its functionalities via a FastAPI backend. The primary interaction points are:

### MCP Protocol Endpoint

*   **Endpoint:** `POST /mcp`
*   **Description:** This is the main endpoint for communication with Model Context Protocol (MCP) clients like Claude Desktop. It handles JSON-RPC requests for various methods defined by the ASR-GoT framework.
*   **Example Request (`asr_got.query` method):**
    ```json
    {
      "jsonrpc": "2.0",
      "method": "asr_got.query",
      "params": {
        "query": "Analyze the relationship between microbiome diversity and cancer progression.",
        "parameters": {
          "include_reasoning_trace": true,
          "include_graph_state": false
        }
      },
      "id": "123"
    }
    ```
*   **Supported MCP Methods:**
    *   `initialize`: To initialize the connection with the MCP server.
    *   `asr_got.query`: To submit a query for processing through the ASR-GoT pipeline.
    *   `shutdown`: To signal the server to prepare for shutdown.

### Health Check Endpoint

*   **Endpoint:** `GET /health`
*   **Description:** Provides a simple health status of the application, indicating if it's running and accessible.
*   **Example Response:**
    ```json
    {
      "status": "healthy",
      "version": "0.1.0" 
    }
    ```

### API Documentation (Swagger UI)

*   **Endpoint:** `GET /docs`
*   **Description:** Access the interactive Swagger UI documentation for a detailed view of all available API endpoints, request/response schemas, and to try out the API directly from your browser.

## Session Handling (`session_id`)

The `session_id` parameter is available in API requests (e.g., for `asr_got.query`) and is included in responses. Its primary functions are:

*   **Tracking:** To identify and track a single, complete query-response cycle.
*   **Correlation:** Used for correlating progress notifications (e.g., `got/queryProgress` if implemented via Server-Sent Events or WebSockets) with the originating query.

**Current Limitations:**
NexusMind does not currently support true multi-turn conversational continuity where the detailed graph state or reasoning context from a previous query is automatically loaded and reused for a follow-up query using the same `session_id`. Each query is processed independently at this time.

### Future Enhancement: Persistent Sessions

A potential future enhancement for NexusMind is the implementation of persistent sessions. This would enable more interactive and evolving reasoning processes by allowing users to:

1.  **Persist State:** Store the generated graph state and relevant reasoning context from a query, associated with its `session_id`, likely within the Neo4j database.
2.  **Reload State:** When a new query is submitted with an existing `session_id`, the system could reload this saved state as the starting point for further processing.
3.  **Refine and Extend:** Allow the new query to interact with the loaded graph—for example, by refining previous hypotheses, adding new evidence to existing structures, or exploring alternative reasoning paths based on the established context.

This is a significant feature that could greatly enhance the interactive capabilities of NexusMind.

### Future Enhancement: Asynchronous and Parallel Stage Execution

Currently, the 8 stages of the NexusMind reasoning pipeline are executed sequentially. For complex queries or to further optimize performance, exploring asynchronous or parallel execution for certain parts of the pipeline is a potential future enhancement.

**Potential Areas for Parallelism:**

*   **Hypothesis Generation:** Hypothesis generation for different, independent dimensions could potentially be parallelized.
*   **Evidence Integration (Partial):** The "plan execution" phase for different hypotheses might be performed concurrently.

**Challenges and Considerations:**
Implementing parallelism requires careful management of data consistency, transaction management, dependency sequencing, resource utilization, and overall complexity.

## Testing & Quality Assurance

NexusMind uses Pytest for testing, Ruff for linting and formatting, and MyPy/Pyright for type checking.

<div align="center">
  <table>
    <tr>
      <td align="center">🧪<br><b>Testing</b></td>
      <td align="center">🔍<br><b>Type Checking</b></td>
      <td align="center">✨<br><b>Linting</b></td>
      <td align="center">📊<br><b>Coverage</b></td>
    </tr>
    <tr>
      <td align="center">
        <pre>poetry run pytest</pre>
        <pre>make test</pre>
      </td>
      <td align="center">
        <pre>poetry run mypy src/</pre>
        <pre>poetry run pyright src/</pre>
      </td>
      <td align="center">
        <pre>poetry run ruff check .</pre>
        <pre>poetry run ruff format .</pre>
      </td>
      <td align="center">
        <pre>poetry run pytest --cov=src</pre>
        <pre>coverage html</pre>
      </td>
    </tr>
  </table>
</div>

### Development Commands

```bash
# Run full test suite with coverage using Poetry
poetry run pytest --cov=src --cov-report=html --cov-report=term

# Or using Makefile for the default test run
make test

# Run specific test categories (using poetry)
poetry run pytest tests/unit/stages/          # Stage-specific tests
poetry run pytest tests/integration/         # Integration tests
poetry run pytest -k "test_confidence"       # Tests matching pattern

# Type checking and linting (can also be run via Makefile targets: make lint, make check-types)
poetry run mypy src/ --strict                # Strict type checking
poetry run ruff check . --fix                # Auto-fix linting issues
poetry run ruff format .                     # Format code

# Pre-commit hooks (recommended for contributors)
poetry run pre-commit install                # Install hooks
poetry run pre-commit run --all-files       # Run all hooks

# See Makefile for other useful targets like 'make all-checks'.
```

### Quality Metrics

- **Type Safety**: 
  - Fully typed codebase with strict mypy configuration.
  - Configured with `mypy.ini` and `pyrightconfig.json`.
- **Code Quality**:
  - Aim for 95%+ test coverage.
  - Automated formatting with Ruff.
  - Pre-commit hooks for consistent code quality.
  - Comprehensive integration tests for the 8-stage pipeline.
```
