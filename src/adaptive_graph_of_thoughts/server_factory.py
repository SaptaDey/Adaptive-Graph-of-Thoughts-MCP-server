import json
import sys
import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Only for type hints, not actual imports
if TYPE_CHECKING:
    from adaptive_graph_of_thoughts.domain.services.got_processor import (
        GoTProcessor,
    )

from loguru import logger

from adaptive_graph_of_thoughts.api.schemas import (
    JSONRPCResponse,
    MCPASRGoTQueryParams,
    MCPASRGoTQueryResult,
    MCPInitializeParams,
    MCPInitializeResult,
    create_jsonrpc_error,
)
from adaptive_graph_of_thoughts.config import settings
from adaptive_graph_of_thoughts.services.resource_monitor import ResourceMonitor

# Using lazy imports to avoid circular dependencies


class ProcessManager:
    """Manages background processes for HTTP server testing."""
    
    def __init__(self):
        self.processes = []
    
    def start_http_server(self, timeout=30):
        """Start HTTP server in background and wait for it to be ready."""
        logger.info("Starting HTTP server...")
        
        # Change to project root directory
        project_root = Path(__file__).parent.parent.parent
        
        # Start server process
        cmd = [
            "python3", "-m", "uvicorn", 
            "src.adaptive_graph_of_thoughts.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group
            cwd=project_root
        )
        self.processes.append(process)
        
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if server is responding
                health_check = subprocess.run(
                    ["curl", "-s", "http://localhost:8000/health"],
                    capture_output=True,
                    timeout=5
                )
                if health_check.returncode == 0:
                    logger.info("HTTP server is ready!")
                    return process
            except subprocess.TimeoutExpired:
                pass
            
            # Check if process crashed
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error("Server process crashed: {}", stderr.decode())
                raise RuntimeError("HTTP server failed to start")
            
            time.sleep(2)
        
        raise TimeoutError("HTTP server failed to start within timeout")
    
    def cleanup(self):
        """Clean up all managed processes."""
        for process in self.processes:
            try:
                # Kill entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


class MCPServerFactory:
    """Factory class for creating and managing MCP servers in different transport modes."""

    @staticmethod
    def detect_transport_mode() -> str:
        """
        Auto-detect the appropriate transport mode based on the environment.

        Returns:
            "stdio" if running in a STDIO environment, "http" otherwise
        """
        if hasattr(sys.stdin, "isatty") and not sys.stdin.isatty():
            return "stdio"
        return "http"

    @staticmethod
    def should_run_http() -> bool:
        """
        Determine if HTTP transport should be enabled.

        Returns:
            True if HTTP transport should be enabled
        """
        transport_type = settings.app.mcp_transport_type.lower()
        return settings.app.mcp_http_enabled and transport_type in ["http", "both"]

    @staticmethod
    def should_run_stdio() -> bool:
        """
        Determine if STDIO transport should be enabled.

        Returns:
            True if STDIO transport should be enabled
        """
        transport_type = settings.app.mcp_transport_type.lower()
        return settings.app.mcp_stdio_enabled and transport_type in ["stdio", "both"]

    @staticmethod
    def run_with_retry(command, max_retries=2, timeout=300):
        """Run command with retry logic and proper timeout handling."""
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries + 1}: Running command: {' '.join(command)}")
                
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    logger.info("Command executed successfully")
                    return True, result.stdout, result.stderr
                else:
                    logger.warning(f"Command failed with exit code {result.returncode}")
                    
                    if attempt < max_retries:
                        wait_time = 5 * (attempt + 1)  # Progressive backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
            
            except subprocess.TimeoutExpired:
                logger.warning(f"Command timed out after {timeout} seconds")
                if attempt < max_retries:
                    logger.info("Retrying...")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < max_retries:
                    logger.info("Retrying...")
                    time.sleep(5)
        
        return False, "", ""

    @staticmethod
    async def run_stdio_server():
        """
        Run the MCP server using STDIO transport with proper MCP protocol implementation.

        This method handles JSON-RPC communication over stdin/stdout following MCP protocol.
        """
        logger.info("Starting MCP STDIO server...")

        # Import GoTProcessor only when needed to avoid circular dependencies
        from adaptive_graph_of_thoughts.domain.services.got_processor import (
            GoTProcessor,
        )

        # Initialize GoT processor
        resource_monitor = ResourceMonitor()
        got_processor = GoTProcessor(settings=settings, resource_monitor=resource_monitor)
        
        # MCP protocol state
        initialized = False
        
        try:
            # Send initial MCP handshake signal to stdout
            # This helps mcp-inspector know the server is ready
            print(json.dumps({
                "jsonrpc": "2.0",
                "method": "server/ready",
                "params": {}
            }), flush=True)
            
            # Main STDIO loop
            while True:
                try:
                    # Read a line from stdin
                    line = sys.stdin.readline()
                    
                    if not line:
                        logger.info("STDIO input closed, shutting down server.")
                        break

                    line = line.strip()
                    if not line:
                        continue

                    # Parse JSON-RPC request
                    try:
                        request_data = json.loads(line)
                        logger.debug("Received STDIO request: {}", request_data)
                    except json.JSONDecodeError as e:
                        logger.error("Invalid JSON received: {}", e)
                        error_response = create_jsonrpc_error(
                            request_id=None, code=-32700, message="Parse error"
                        )
                        print(json.dumps(error_response.model_dump()), flush=True)
                        continue

                    # Handle MCP protocol initialization
                    method = request_data.get("method")
                    if method == "initialize" and not initialized:
                        initialized = True
                        logger.info("MCP protocol initialization received")
                    elif not initialized and method != "initialize":
                        # Reject requests before initialization
                        error_response = create_jsonrpc_error(
                            request_id=request_data.get("id"),
                            code=-32002,
                            message="Server not initialized"
                        )
                        print(json.dumps(error_response.model_dump()), flush=True)
                        continue

                    # Process the request
                    response = await MCPServerFactory._handle_stdio_request(
                        request_data, got_processor
                    )

                    # Send response
                    if response:
                        response_json = json.dumps(response.model_dump())
                        print(response_json, flush=True)
                        logger.debug("Sent STDIO response: {}", response_json)
                    
                    # Handle shutdown
                    if method == "shutdown":
                        logger.info("Shutdown request received, exiting...")
                        break

                except KeyboardInterrupt:
                    logger.info(
                        "Received interrupt signal, shutting down STDIO server."
                    )
                    break
                except Exception as e:
                    logger.exception("Error in STDIO server loop: {}", e)
                    error_response = create_jsonrpc_error(
                        request_id=None, code=-32603, message="Internal error"
                    )
                    print(json.dumps(error_response.model_dump()), flush=True)

        finally:
            # Cleanup
            try:
                await got_processor.shutdown_resources()
            except Exception as e:
                logger.error("Error shutting down GoT processor: {}", e)

            logger.info("MCP STDIO server shutdown complete.")

    @staticmethod
    async def _handle_stdio_request(
        request_data: dict[str, Any], got_processor: "GoTProcessor"
    ) -> Optional[JSONRPCResponse]:
        """
        Handle a single STDIO JSON-RPC request.

        Args:
            request_data: The parsed JSON-RPC request
            got_processor: The GoT processor instance

        Returns:
            JSON-RPC response or None
        """
        try:
            if "jsonrpc" not in request_data or request_data["jsonrpc"] != "2.0":
                return create_jsonrpc_error(
                    request_id=request_data.get("id"),
                    code=-32600,
                    message="Invalid Request",
                )
            method = request_data.get("method")
            params = request_data.get("params", {})
            request_id = request_data.get("id")

            if method == "initialize":
                if request_id is None:
                    await MCPServerFactory._handle_initialize(params, None)
                    return None
                return await MCPServerFactory._handle_initialize(params, request_id)
            elif method == "asr_got.query":
                return await MCPServerFactory._handle_asr_got_query(
                    params, request_id, got_processor
                )
            elif method == "listTools":
                return await MCPServerFactory._handle_list_tools(
                    request_id, got_processor
                )
            elif method == "callTool":
                return await MCPServerFactory._handle_call_tool(
                    params, request_id, got_processor
                )
            elif method == "shutdown":
                await MCPServerFactory._handle_shutdown(params, request_id)
                return JSONRPCResponse(id=request_id, result=None)
            else:
                return create_jsonrpc_error(
                    request_id=request_id,
                    code=-32601,
                    message=f"Method '{method}' not found",
                )
        except Exception as e:
            logger.exception("Error handling STDIO request: {}", e)
            return create_jsonrpc_error(
                request_id=request_data.get("id"), code=-32603, message="Internal error"
            )

    @staticmethod
    async def _handle_initialize(
        params: dict[str, Any], request_id: Optional[str]
    ) -> JSONRPCResponse:
        """Handle MCP initialize request."""
        try:
            parsed_params = MCPInitializeParams(**params)
            logger.info(
                "MCP Initialize request received via STDIO. Client: {}, Process ID: {}",
                parsed_params.client_info.client_name
                if parsed_params.client_info
                else "Unknown",
                parsed_params.process_id,
            )

            # Provide comprehensive server capabilities
            result = MCPInitializeResult(
                server_name="Adaptive Graph of Thoughts MCP Server",
                server_version=settings.app.version,
                mcp_version=settings.mcp_settings.protocol_version,
                capabilities={
                    "tools": {"listChanged": False},
                    "prompts": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "logging": {}
                }
            )
            return JSONRPCResponse(id=request_id, result=result)
        except Exception as e:
            logger.error("Error in initialize handler: {}", e)
            return create_jsonrpc_error(
                request_id=request_id, code=-32602, message="Invalid parameters"
            )

    @staticmethod
    async def _handle_asr_got_query(
        params: dict[str, Any], request_id: Optional[str], got_processor: "GoTProcessor"
    ) -> JSONRPCResponse:
        """Handle ASR-GoT query request."""
        try:
            parsed_params = MCPASRGoTQueryParams(**params)
            logger.info("Processing ASR-GoT query via STDIO: {}", parsed_params.query)
            result = await got_processor.process_query(
                query=parsed_params.query,
                session_id=parsed_params.session_id,
                operational_params=parsed_params.operational_params,
            )
            result_dict = result.model_dump()
            mcp_result = MCPASRGoTQueryResult(
                answer=result_dict.get("final_answer", ""),
                reasoning_trace_summary=result_dict.get("reasoning_trace_summary"),
                graph_state_full=result_dict.get("graph_state_full"),
                confidence_vector=result_dict.get("final_confidence_vector"),
                execution_time_ms=result_dict.get("execution_time_ms"),
                session_id=result_dict.get("session_id"),
            )
            return JSONRPCResponse(id=request_id, result=mcp_result)
        except Exception as e:
            logger.exception("Error in ASR-GoT query handler: {}", e)
            return create_jsonrpc_error(
                request_id=request_id,
                code=-32603,
                message="Internal error processing query",
            )

    @staticmethod
    async def _handle_shutdown(
        _params: dict[str, Any], _request_id: Optional[str]
    ) -> None:
        """Handle shutdown request."""
        logger.info("MCP Shutdown request received via STDIO.")

    @staticmethod
    async def _handle_list_tools(
        request_id: Optional[str], got_processor: "GoTProcessor"
    ) -> JSONRPCResponse:
        """Return available tools if the processor is ready."""
        if not getattr(got_processor, "models_loaded", True):
            return JSONRPCResponse(id=request_id, result=[])
        tools = [
            {
                "name": "graph_reasoning",
                "description": "Perform graph-based reasoning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "confidence_threshold": {"type": "number", "default": 0.7},
                    },
                },
            }
        ]
        return JSONRPCResponse(id=request_id, result=tools)

    @staticmethod
    async def _handle_call_tool(
        params: dict[str, Any], request_id: Optional[str], got_processor: "GoTProcessor"
    ) -> JSONRPCResponse:
        """Execute a tool call with basic error handling."""
        if not getattr(got_processor, "models_loaded", True):
            return JSONRPCResponse(
                id=request_id,
                result=[{"type": "text", "text": "Server is still initializing. Please wait..."}],
            )
        name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            if name == "graph_reasoning":
                query = arguments.get("query")
                if not query:
                    return create_jsonrpc_error(
                        request_id=request_id,
                        code=-32602,
                        message="Missing 'query' argument",
                    )
                conf = arguments.get("confidence_threshold", 0.7)
                result = await got_processor.process_query(
                    query=query,
                    operational_params={"confidence_threshold": conf},
                )
                answer = result.final_answer or ""
                return JSONRPCResponse(
                    id=request_id,
                    result=[{"type": "text", "text": answer}],
                )
            return create_jsonrpc_error(
                request_id=request_id,
                code=-32601,
                message=f"Tool '{name}' not found",
            )
        except Exception as e:  # pragma: no cover - best effort
            logger.error(f"Tool execution failed: {e}")
            return JSONRPCResponse(
                id=request_id,
                result=[{"type": "text", "text": f"Error: {e}"}],
            )