# MCP Inspector Test Execution Issues

This document outlines the issues encountered while attempting to run the MCP inspector tests using `scripts/run_mcp_inspector.sh` for both STDIO and HTTP transports. Both test scenarios resulted in persistent timeouts, preventing successful completion.

## 1. STDIO Transport Issues

**Problem:**
The command `scripts/run_mcp_inspector.sh stdio` (which executes `mcp-inspector -v validate stdio --program python3 src/adaptive_graph_of_thoughts/main_stdio.py`) consistently timed out after 400 seconds.

**Troubleshooting Steps Taken:**
1.  **Initial `mcp-inspector` Not Found:**
    *   Identified that `mcp-inspector` was not found because it's installed locally in `node_modules/.bin`.
    *   **Fix:** Modified `scripts/run_mcp_inspector.sh` to prepend `$PROJECT_ROOT/node_modules/.bin` to the `PATH` environment variable. This resolved the "command not found" error.

2.  **Python Interpreter Version:**
    *   Ensured that `mcp-inspector` explicitly uses `python3` for the `--program` argument, consistent with the project's environment.
    *   **Action:** Changed `"--program", "python", ...` to `"--program", "python3", ...` in `scripts/mcp_inspector_executor.py`. This did not resolve the timeout.

3.  **Initial Output/Handshake:**
    *   Hypothesized that `mcp-inspector` might be waiting for an initial signal on `stdout` from the Python STDIO server.
    *   **Action:** Added `print("", flush=True)` at the beginning of the server's STDIO processing loop in `src/adaptive_graph_of_thoughts/server_factory.py`. This did not resolve the timeout.

4.  **Verbose Output from `mcp-inspector`:**
    *   Added the `-v` (verbose) flag to the `mcp-inspector` command.
    *   **Action:** No additional useful diagnostic output was observed from `mcp-inspector` before the timeout occurred.

5.  **STDERR Interference:**
    *   Suspected that log messages from the Python server on `stderr` might be interfering with `mcp-inspector`.
    *   **Action:** Modified `src/adaptive_graph_of_thoughts/main_stdio.py` to redirect all `loguru` output to a file (`stdio_server.log`), ensuring a clean `stderr` stream for `mcp-inspector`. This did not resolve the timeout.

6.  **Server-Side Initialization Speed:**
    *   Thoroughly analyzed the initialization process of the Python STDIO server. This included:
        *   Loading of settings (`src/adaptive_graph_of_thoughts/config.py`), including YAML parsing and schema validation.
        *   Initialization of `GoTProcessor` (`src/adaptive_graph_of_thoughts/domain/services/got_processor.py`).
        *   Initialization of all 8 pipeline stages (`src/adaptive_graph_of_thoughts/domain/stages/stage_*.py`) by checking their `__init__` methods.
    *   **Conclusion:** All server-side Python components were found to initialize very quickly, ruling out a slow server startup as the cause of the pre-communication delay.

**Assessment for STDIO:**
The persistent 400-second timeout, despite these troubleshooting steps, suggests a fundamental issue such as a deadlock in the STDIO communication between `mcp-inspector` and the Python script, or an internal issue/hang within the `mcp-inspector` tool itself when operating in `stdio` mode. Without more detailed diagnostics from `mcp-inspector`, further debugging was not feasible.

## 2. HTTP Transport Issues

**Problem:**
Attempts to run the HTTP transport tests via `scripts/run_mcp_inspector.sh http` also failed. The primary issue was the inability to reliably start the Uvicorn server in the background within the `run_in_bash_session` tool environment. The session block would time out after 400 seconds during the server startup phase.

**Troubleshooting Steps Taken:**
1.  **Background Process Management:**
    *   The Uvicorn server was launched using `... &` to run it in the background.
    *   **Issue:** The `run_in_bash_session` consistently timed out, even when the script was simplified to only start Uvicorn and check its status. This indicated the Uvicorn process was not detaching or backgrounding correctly within the tool's environment, or was hanging during startup.

2.  **Uvicorn Output Redirection:**
    *   To prevent Uvicorn's console output from interfering with the bash session or backgrounding, its `stdout` and `stderr` were redirected to log files (`uvicorn.stdout.log`, `uvicorn.stderr.log`).
    *   **Action:** This did not resolve the timeout.

3.  **Uvicorn Reload Mode:**
    *   Uvicorn's auto-reload feature can sometimes cause issues in constrained environments.
    *   **Action:** Disabled auto-reload by setting the environment variable `APP_UVICORN_RELOAD=False`. This did not resolve the timeout.

4.  **FastAPI Application Initialization Speed:**
    *   Analyzed the FastAPI application setup in `src/adaptive_graph_of_thoughts/main.py` and `src/adaptive_graph_of_thoughts/app_setup.py`.
    *   **Conclusion:** Similar to the STDIO server, the FastAPI app (including `GoTProcessor` initialization via `app.state.got_processor`) was found to have a very fast startup sequence.

**Assessment for HTTP:**
The inability to start the Uvicorn server correctly in the background within the `run_in_bash_session` environment prevented the execution of the `mcp-inspector` HTTP tests. The timeout occurs before the tests themselves can even begin. This points to a possible incompatibility or limitation in how background server processes are handled by the available bash execution tool.

## Overall Conclusion

Due to persistent and intractable timeout issues with both STDIO and HTTP transports, primarily related to the behavior of the `mcp-inspector` tool and the constraints of the Uvicorn server execution within the provided environment, it was not possible to run the MCP inspector tests to completion for all transports as required by the subtask.
The root causes appear external to the core application logic's initialization speed, pointing towards tool interaction or environment limitations.
