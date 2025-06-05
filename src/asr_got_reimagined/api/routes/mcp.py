import time
import secrets
from typing import Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import ValidationError

from src.asr_got_reimagined.api.schemas import (
    GraphStateSchema,
    JSONRPCErrorObject,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPASRGoTQueryParams,
    MCPASRGoTQueryResult,
    MCPInitializeParams,
    MCPInitializeResult,
    ShutdownParams,
)
from src.asr_got_reimagined.config import settings # Import settings
from src.asr_got_reimagined.domain.services.got_processor import (
    GoTProcessor,
    GoTProcessorSessionData,
)

mcp_router = APIRouter()


# Dependency for authentication
async def verify_token(http_request: Request):
    if settings.app.auth_token:
        auth_header = http_request.headers.get("Authorization")
        if not auth_header:
            logger.warning("MCP request missing Authorization header when auth_token is configured.")
            raise HTTPException(status_code=401, detail="Not authenticated")

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning(f"MCP request with malformed Authorization header: {auth_header}")
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        token = parts[1]
        if token != settings.app.auth_token:
        if not secrets.compare_digest(token, settings.app.auth_token):
            logger.warning("MCP request with invalid token.")
            raise HTTPException(status_code=403, detail="Invalid token")
        logger.debug("MCP request authenticated successfully via token.")
    else:
        logger.debug("MCP auth_token not configured, skipping authentication.")
    return True


def create_jsonrpc_error(
    request_id: Optional[Union[str, int]],
    code: int,
    message: str,
    data: Optional[Any] = None,
) -> JSONRPCResponse[Any, Any]:
    logger.error(f"MCP Error (ID: {request_id}): Code {code} - {message}. Data: {data}")
    return JSONRPCResponse(
        id=request_id, error=JSONRPCErrorObject(code=code, message=message, data=data)
    )


async def handle_initialize(
    params: MCPInitializeParams, request_id: Optional[Union[str, int]]
) -> JSONRPCResponse[MCPInitializeResult, Any]:
    logger.info(
        "MCP Initialize request received. Client: {}, Process ID: {}",
        params.client_info.client_name
        if params.client_info and params.client_info.client_name
        else "Unknown",
        params.process_id,
    )
    # Use protocol_version from settings
    initialize_result = MCPInitializeResult(
        server_name="NexusMind MCP Server",
        server_version="0.1.0",
        mcp_version="2024-11-05",
    )

    logger.info("MCP Initialize successful. Server info sent: {}", initialize_result)
    return JSONRPCResponse(id=request_id, result=initialize_result)


async def handle_asr_got_query(
    request_obj: Request,
    params: MCPASRGoTQueryParams,
    request_id: Optional[Union[str, int]],
) -> JSONRPCResponse[MCPASRGoTQueryResult, Any]:
    """
    Processes an "asr_got.query" JSON-RPC request using the GoTProcessor and returns the result or an error response.
    
    Forwards the query and parameters to the GoTProcessor, optionally including the graph state and a reasoning trace summary in the response. Converts and validates returned data, measures execution time, and provides robust error handling with fallback responses if processing fails.
    
    Returns:
        A JSON-RPC response containing the ASR-GoT query result, which may include the answer, reasoning trace summary, graph state, confidence vector, execution time, and session ID. If an error occurs, returns a JSON-RPC error response with diagnostic details.
    """
    logger.info(
        "MCP asr_got.query request received for query: '{}'",
        params.query[:100] + "..." if params.query else "N/A",
    )
    logger.debug("Full asr_got.query params: {}", params)

    start_time_ns = time.time_ns()
    processor: GoTProcessor = request_obj.app.state.got_processor

    if not processor:
        logger.error("GoTProcessor not found in app state! Cannot process query.")
        return create_jsonrpc_error(
            request_id=request_id,
            code=-32002,
            message="NexusMind Core Processor is not available.",
        )

    try:
        op_params_dict = (
            params.parameters.model_dump(exclude_none=True) if params.parameters else {}
        )
        context_dict = (
            params.context.model_dump(exclude_none=True) if params.context else {}
        )

        session_data_result: GoTProcessorSessionData = await processor.process_query(
            query=params.query,
            session_id=params.session_id,
            operational_params=op_params_dict,
            initial_context=context_dict,
        )

        response_graph_state: Optional[GraphStateSchema] = None
        if params.parameters and params.parameters.include_graph_state:
            if (
                hasattr(session_data_result, "graph_state")
                and session_data_result.graph_state
            ):
                if isinstance(session_data_result.graph_state, GraphStateSchema):
                    response_graph_state = session_data_result.graph_state
                elif isinstance(session_data_result.graph_state, dict):
                    try:
                        response_graph_state = GraphStateSchema(
                            **session_data_result.graph_state
                        )
                    except Exception as e_conv:
                        logger.error(
                            f"Failed to parse graph_state dict into GraphStateSchema for session {session_data_result.session_id}: {e_conv}"
                        )
                else:
                    logger.warning(
                        f"Graph state in session data for session {session_data_result.session_id} is not of a recognized type for conversion."
                    )

            if not response_graph_state and params.parameters.include_graph_state:
                logger.warning(
                    f"Graph state was requested but could not be retrieved or converted for session {session_data_result.session_id}."
                )

        execution_time_ms = (time.time_ns() - start_time_ns) // 1_000_000

        reasoning_trace_text = "Reasoning trace not requested or not available."
        if params.parameters and params.parameters.include_reasoning_trace:
            if session_data_result.stage_outputs_trace:  # Check if list is not empty
                try:
                    trace_lines = [
                        f"Stage {s.get('stage_number', 'N/A')}. {s.get('stage_name', 'Unknown Stage')}: {s.get('summary', 'N/A')} ({s.get('duration_ms', 0)}ms)"
                        for s in session_data_result.stage_outputs_trace
                    ]
                    reasoning_trace_text = "\n".join(trace_lines)
                except AttributeError:
                    # If we get an AttributeError, the stage_outputs_trace might have a different structure
                    # Convert each item to a dict and try again
                    try:
                        trace_lines = []
                        for item in session_data_result.stage_outputs_trace:
                            if hasattr(item, "__dict__"):
                                s = item.__dict__
                                trace_lines.append(
                                    f"Stage {s.get('stage_number', 'N/A')}. {s.get('stage_name', 'Unknown Stage')}: {s.get('summary', 'N/A')} ({s.get('duration_ms', 0)}ms)"
                                )
                            else:
                                # As a last resort, just convert to string
                                trace_lines.append(str(item))
                        reasoning_trace_text = "\n".join(trace_lines)
                    except Exception as e:
                        logger.error(f"Failed to process stage_outputs_trace: {e}")
                        reasoning_trace_text = "Error processing reasoning trace."
            else:
                reasoning_trace_text = (
                    "Reasoning trace requested, but no trace data was generated."
                )

        query_result = MCPASRGoTQueryResult(
            answer=session_data_result.final_answer
            or "Processing complete, but no explicit answer generated.",
            reasoning_trace_summary=reasoning_trace_text,
            graph_state_full=response_graph_state,
            confidence_vector=session_data_result.final_confidence_vector,
            execution_time_ms=execution_time_ms,
            session_id=session_data_result.session_id,
        )
        logger.info(
            "MCP asr_got.query processed successfully. Answer generated for ID: {}",
            request_id,
        )
        return JSONRPCResponse(id=request_id, result=query_result)

    except AttributeError as ae:
        logger.exception(
            f"AttributeError during asr_got.query processing for ID {request_id}: {ae}. This might indicate a mismatch in method names (e.g. process_query parameters) or data structures."
        )
        # Try to still return a response, even if we can't process the trace output
        try:
            # Create a fallback response
            query_result = MCPASRGoTQueryResult(
                answer="Sorry, there was an error processing this query, but I'll try to provide a response.",
                reasoning_trace_summary=f"Error in processing: {ae!s}",
                graph_state_full=None,
                confidence_vector=[0.1, 0.1, 0.1, 0.1],  # Low confidence due to error
                execution_time_ms=0,
                session_id=getattr(
                    session_data_result, "session_id", f"error-session-{request_id}"
                ),
            )
            logger.warning(
                f"Returning fallback response for ID: {request_id} after error"
            )
            return JSONRPCResponse(id=request_id, result=query_result)
        except Exception as e2:
            # If even our fallback fails, return the original error
            logger.error(f"Fallback response also failed: {e2}")
            return create_jsonrpc_error(
                request_id=request_id,
                code=-32003,
                message="Internal server error: Incompatible data structures or method signature mismatch.",
                data={"details": str(ae), "method": "asr_got.query"},
            )
    except Exception as e:
        logger.exception("Error during asr_got.query processing for ID: {}", request_id)
        return create_jsonrpc_error(
            request_id=request_id,
            code=-32000,
            message="Error processing ASR-GoT query.",
            data={"details": str(e), "method": "asr_got.query"},
        )


async def handle_shutdown(
    params: Optional[ShutdownParams], request_id: Optional[Union[str, int]]
) -> JSONRPCResponse[None, Any]:
    logger.info("MCP Shutdown request received. Params: {}", params)
    logger.info("Application will prepare to shut down.")
    return JSONRPCResponse(id=request_id, result=None)


@mcp_router.post("", dependencies=[Depends(verify_token)])
async def mcp_endpoint_handler(
    request_payload: JSONRPCRequest[dict[str, Any]], http_request: Request # http_request is still available here if needed by handlers
):
    """
    Handles incoming MCP JSON-RPC requests and dispatches them to the appropriate method handler.
    
    Parses the method and parameters from the request payload, invokes the corresponding handler for "initialize", "asr_got.query", or "shutdown" methods, and returns a JSON-RPC response. Returns a JSON-RPC error for unsupported methods or invalid parameters. Exceptions are logged and mapped to appropriate JSON-RPC error responses.
    """
    logger.debug(
        "MCP Endpoint received raw request: method={}, id={}",
        request_payload.method,
        request_payload.id,
    )

    method = request_payload.method
    req_id = request_payload.id
    params_data = request_payload.params if request_payload.params is not None else {}

    try:
        if method == "initialize":
            parsed_params = MCPInitializeParams(**params_data)
            return await handle_initialize(params=parsed_params, request_id=req_id)

        elif method == "asr_got.query":
            parsed_params = MCPASRGoTQueryParams(**params_data)
            return await handle_asr_got_query(http_request, parsed_params, req_id)

        elif method == "shutdown":
            parsed_params = ShutdownParams(**params_data)
            return await handle_shutdown(params=parsed_params, request_id=req_id)

        else:
            logger.warning("Unsupported MCP method received: {}", method)
            return create_jsonrpc_error(
                request_id=req_id, code=-32601, message=f"Method '{method}' not found."
            )

    except HTTPException:
        raise
    except ValidationError as ve:
        logger.warning(f"MCP Validation Error for method {method}: {ve}")
        return create_jsonrpc_error(
            request_id=req_id,
            code=-32602,
            message="Invalid parameters.",
            data={"details": ve.errors(), "method": method},
        )
    except Exception as e:
        logger.exception("Error in MCP endpoint_handler for method {}: {}", method, e)
        return create_jsonrpc_error(
            request_id=req_id,
            code=-32603,
            message=f"Internal error processing request for method '{method}'.",
            data={"details": str(e), "method": method},
        )
