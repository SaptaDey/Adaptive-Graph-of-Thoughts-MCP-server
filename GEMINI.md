# Project: 
Adaptive-Graph-of-Thoughts-MCP-server


## Gemini Added Memories

- I have access to the following MCP servers and their capabilities: sequentialthinking, memory, desktop-commander, filesystem, github-official, wolfram-alpha, MCP_DOCKER, toolbox, code-mcp, vscode-mcp-server, exa, pubmed-mcp-server, mem0-memory-mcp, mermaid-mcp-server, r-playground-mcp, basic-memory, smithery-cli.

## General Instructions:

- When generating new TypeScript code, please follow the existing coding style.
- Ensure all new functions and classes have JSDoc comments.
- Prefer functional programming paradigms where appropriate.
- All code should be compatible with TypeScript 5.0 and Node.js 18+.

## Coding Style:

- Use 2 spaces for indentation.
- Interface names should be prefixed with `I` (e.g., `IUserService`).
- Private class members should be prefixed with an underscore (`_`).
- Always use strict equality (`===` and `!==`).

## Specific Component: `src/api/client.ts`

- This file handles all outbound API requests.
- When adding new API call functions, ensure they include robust error handling and logging.
- Use the existing `fetchWithRetry` utility for all GET requests.

## Regarding Dependencies:

- Avoid introducing new external dependencies unless absolutely necessary.
- If a new dependency is required, please state the reason.
