# Status Page Information

The `status.html` page, located in the root of the repository, provides a simple, standalone HTML interface for interacting with a running NexusMind server.

## Features

*   **Server Status Check:** Quickly verify if the NexusMind MCP server is running and accessible.
*   **Basic Query Interface:** Allows sending a query string and parameters (as JSON) to the `asr_got.query` MCP method.
*   **View Responses:** Displays the JSON response received from the server.

## Usage

1.  Ensure your NexusMind server is running locally (e.g., via `poetry run uvicorn src.asr_got_reimagined.main:app --reload`).
2.  Open the `status.html` file directly in your web browser (e.g., `file:///path/to/your/NexusMind-2.0/status.html`).
3.  The page will typically attempt to connect to the server at `http://localhost:8000/mcp`.
4.  Use the form to send test queries.

**Note:** This page makes direct HTTP requests from your browser to the local server. It's intended for development and testing purposes. It does not require MkDocs to run and is not part of the generated documentation site in the same way as these Markdown files.
