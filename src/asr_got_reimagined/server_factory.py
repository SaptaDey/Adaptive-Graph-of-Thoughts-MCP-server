import json
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING

# Only for type hints, not actual imports
if TYPE_CHECKING:
    from src.asr_got_reimagined.domain.services.got_processor import GoTProcessor

from loguru import logger

from src.asr_got_reimagined.config import settings
from src.asr_got_reimagined.api.schemas import (
    JSONRPCResponse,
    MCPInitializeParams,
    MCPInitializeResult,
    MCPASRGoTQueryParams,
    MCPASRGoTQueryResult,
    create_jsonrpc_error,
)
# Using lazy imports to avoid circular dependencies


class MCPServerFactory:
    """Factory class for creating and managing MCP servers in different transport modes."""

    @staticmethod
    def detect_transport_mode() -> str:
        """
        Auto-detect the appropriate transport mode based on the environment.

        Returns:
            "stdio" if running in a STDIO environment, "http" otherwise
        """
        # Check if we're running in a STDIO environment
        # This typically means stdin/stdout are connected to pipes
        if hasattr(sys.stdin, 'isatty') and not sys.stdin.isatty():
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
        return (
            settings.app.mcp_http_enabled and 
            transport_type in ["http", "both"]
        )

    @staticmethod
    def should_run_stdio() -> bool:
        """
        Determine if STDIO transport should be enabled.

        Returns:
            True if STDIO transport should be enabled
        """
        transport_type = settings.app.mcp_transport_type.lower()
        return (
            settings.app.mcp_stdio_enabled and 
            transport_type in ["stdio", "both"]
        )

    @staticmethod
    async def run_stdio_server():
        """
        Run the MCP server using STDIO transport.

        This method handles JSON-RPC communication over stdin/stdout.
        """
        logger.info("Starting MCP STDIO server...")

        # Import GoTProcessor only when needed to avoid circular dependencies
        from src.asr_got_reimagined.domain.services.got_processor import GoTProcessor

        # Initialize GoT processor
        got_processor = GoTProcessor(settings=settings)

        try:
            # Main STDIO loop
            while True:
                try:
                    # Read a line from stdin without blocking the event loop
                    import asyncio
                    line = await asyncio.to_thread(sys.stdin.readline)
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
                            request_id=None,
                            code=-32700,
                            message="Parse error"
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

                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, shutting down STDIO server.")
                    break
                except Exception as e:
                    logger.exception("Error in STDIO server loop: {}", e)
                    error_response = create_jsonrpc_error(
                        request_id=None,
                        code=-32603,
                        message="Internal error"
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
        request_data: Dict[str, Any], 
        got_processor: "GoTProcessor"
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
            # Validate basic JSON-RPC structure
            if "jsonrpc" not in request_data or request_data["jsonrpc"] != "2.0":
                return create_jsonrpc_error(
                    request_id=request_data.get("id"),
                    code=-32600,
                    message="Invalid Request"
                )

            method = request_data.get("method")
            params = request_data.get("params", {})
            request_id = request_data.get("id")

            if method == "initialize":
                # JSON-RPC notifications (no `id`) do not expect a result (§5)
                if request_id is None:
                    await MCPServerFactory._handle_initialize(params, None)
                    return None

                return await MCPServerFactory._handle_initialize(params, request_id)
            elif method == "asr_got.query":
                return await MCPServerFactory._handle_asr_got_query(
                    params, request_id, got_processor
                )

            elif method == "shutdown":
                await MCPServerFactory._handle_shutdown(params, request_id)
                return JSONRPCResponse(id=request_id, result=None)

            else:
                return create_jsonrpc_error(
                    request_id=request_id,
                    code=-32601,
                    message=f"Method '{method}' not found"
                )

        except Exception as e:
            logger.exception("Error handling STDIO request: {}", e)
            return create_jsonrpc_error(
                request_id=request_data.get("id"),
                code=-32603,
                message="Internal error"
            )

    @staticmethod
    async def _handle_initialize(
        params: Dict[str, Any], 
        request_id: Optional[str]
    ) -> JSONRPCResponse:
        """Handle MCP initialize request."""
        try:
            parsed_params = MCPInitializeParams(**params)
            logger.info(
                "MCP Initialize request received via STDIO. Client: {}, Process ID: {}",
                parsed_params.client_info.client_name if parsed_params.client_info else "Unknown",
                parsed_params.process_id,
            )

            result = MCPInitializeResult(
                server_name="NexusMind MCP Server",
                server_version="0.1.0",
                mcp_version="2024-11-05",
            )

            return JSONRPCResponse(id=request_id, result=result)

        except Exception as e:
            logger.error("Error in initialize handler: {}", e)
            return create_jsonrpc_error(
                request_id=request_id,
                code=-32602,
                message="Invalid parameters"
            )

    @staticmethod
    async def _handle_asr_got_query(
        params: Dict[str, Any], 
        request_id: Optional[str],
        got_processor: "GoTProcessor"
    ) -> JSONRPCResponse:
        """Handle ASR-GoT query request."""
        try:
            parsed_params = MCPASRGoTQueryParams(**params)
            logger.info("Processing ASR-GoT query via STDIO: {}", parsed_params.query)

            # Process the query using GoT processor
            result = await got_processor.process_query(
                query=parsed_params.query,
                session_id=parsed_params.session_id,
                operational_params=parsed_params.operational_params
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
                message="Internal error processing query"
            )

    @staticmethod
    async def _handle_shutdown(
        params: Dict[str, Any], 
        request_id: Optional[str]
    ) -> None:
        """Handle shutdown request."""
        logger.info("MCP Shutdown request received via STDIO.")
        # Note: The actual shutdown will be handled by the main loop
