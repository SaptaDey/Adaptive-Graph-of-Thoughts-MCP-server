import secrets
import hmac

import time
from typing import Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import ValidationError

from adaptive_graph_of_thoughts.api.schemas import (
    GraphStateSchema,
    JSONRPCErrorObject,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPASRGoTQueryParams,
    MCPQueryOperationalParams,
    MCPASRGoTQueryResult,
    MCPInitializeParams,
    MCPInitializeResult,
    ShutdownParams,
)
from adaptive_graph_of_thoughts.config import settings  # Import settings
from adaptive_graph_of_thoughts.application import GoTProcessor
from adaptive_graph_of_thoughts.domain.models.common_types import (
    GoTProcessorSessionData,
)

mcp_router = APIRouter()


def _parse_dot_notation(params: dict[str, str]) -> dict[str, Any]:
    """Parse dot-notation query parameters into a nested dict."""
    config: dict[str, Any] = {}
    for key, value in params.items():
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config


# Dependency for authentication
async def verify_token(http_request: Request):
    if not settings.app.auth_token:
        raise HTTPException(status_code=401, detail="Authentication required")

    auth_header = http_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = auth_header[7:]

    if not hmac.compare_digest(settings.app.auth_token.encode(), token.encode()):
        raise HTTPException(status_code=403, detail="Invalid token")

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
        server_name="Adaptive Graph of Thoughts MCP Server",
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
            message="Adaptive Graph of Thoughts Core Processor is not available.",
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
    request_payload: JSONRPCRequest[dict[str, Any]],
    http_request: Request,  # http_request is still available here if needed by handlers
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


@mcp_router.get("", dependencies=[Depends(verify_token)])
async def mcp_get(http_request: Request) -> dict[str, Any]:
    """Handle GET requests for MCP configuration."""
    config = _parse_dot_notation(dict(http_request.query_params))
    return {"status": "ok", "config": config}


@mcp_router.delete("", dependencies=[Depends(verify_token)])
async def mcp_delete(http_request: Request) -> dict[str, Any]:
    """Handle DELETE requests for MCP shutdown acknowledgment."""
    config = _parse_dot_notation(dict(http_request.query_params))
    return {"status": "deleted", "config": config}

@mcp_router.get("/tools", dependencies=[Depends(verify_token)])
async def get_available_tools() -> dict[str, Any]:
    """Return available MCP tools for client discovery."""
    import json
    import os
    
    # Load tools definition
    tools_file = os.path.join(os.path.dirname(__file__), "../../../config/mcp_tools_definition.json")
    try:
        with open(tools_file, 'r') as f:
            tools_data = json.load(f)
        return {
            "status": "success",
            "tools": tools_data.get("tools", []),
            "resources": tools_data.get("resources", []),
            "prompts": tools_data.get("prompts", [])
        }
    except FileNotFoundError:
        logger.warning(f"Tools definition file not found: {tools_file}")
        return {
            "status": "error",
            "message": "Tools definition not available",
            "tools": []
        }

@mcp_router.get("/capabilities", dependencies=[Depends(verify_token)])
async def get_server_capabilities() -> dict[str, Any]:
    """Return server capabilities for MCP client negotiation."""
    return {
        "server_name": "Adaptive Graph of Thoughts MCP Server",
        "server_version": "0.1.0",
        "mcp_version": "2024-11-05",
        "capabilities": {
            "tools": True,
            "resources": True,
            "prompts": True,
            "logging": True,
            "experimental": {
                "graph_reasoning": True,
                "evidence_integration": True,
                "confidence_scoring": True
            }
        },
        "supported_transports": ["stdio", "http"],
        "evidence_sources": ["pubmed", "google_scholar", "exa_search"]
    }

@mcp_router.post("/tools/{tool_name}", dependencies=[Depends(verify_token)])
async def execute_tool(
    tool_name: str,
    tool_params: dict[str, Any],
    http_request: Request
) -> dict[str, Any]:
    """Execute a specific MCP tool by name."""
    logger.info(f"Executing tool: {tool_name} with params: {tool_params}")
    
    # Map tool names to existing handlers
    tool_mapping = {
        "scientific_reasoning_query": "asr_got.query",
        "analyze_research_hypothesis": "asr_got.query",
        "explore_scientific_relationships": "asr_got.query",
        "validate_scientific_claims": "asr_got.query"
    }
    
    if tool_name not in tool_mapping:
        return {
            "status": "error",
            "message": f"Tool '{tool_name}' not found",
            "available_tools": list(tool_mapping.keys())
        }
    
    # Transform tool parameters to MCP query format
    mcp_method = tool_mapping[tool_name]
    
    # Create appropriate query based on tool type
    if tool_name == "scientific_reasoning_query":
        query = tool_params.get("query", "")
    elif tool_name == "analyze_research_hypothesis":
        hypothesis = tool_params.get("hypothesis", "")
        domain = tool_params.get("research_domain", "")
        query = f"Analyze the following research hypothesis in the context of {domain}: {hypothesis}"
    elif tool_name == "explore_scientific_relationships":
        primary = tool_params.get("primary_concept", "")
        related = tool_params.get("related_concepts", [])
        query = f"Explore the scientific relationships between {primary} and the following concepts: {', '.join(related)}"
    elif tool_name == "validate_scientific_claims":
        claim = tool_params.get("claim", "")
        threshold = tool_params.get("evidence_threshold", 0.7)
        query = f"Validate the following scientific claim with evidence threshold {threshold}: {claim}"
    else:
        query = str(tool_params)
    
    # Create MCP request
    mcp_params = MCPASRGoTQueryParams(
        query=query,
        parameters=MCPQueryOperationalParams(
            include_reasoning_trace=tool_params.get("include_reasoning_trace", True),
            include_graph_state=tool_params.get("include_graph_state", True),
            max_nodes_in_response_graph=tool_params.get("max_nodes_in_response_graph", 50),
            output_detail_level=tool_params.get("output_detail_level", "summary")
        ),
        session_id=tool_params.get("session_id")
    )
    
    try:
        # Execute the query
        response = await handle_asr_got_query(http_request, mcp_params, None)
        
        if hasattr(response, 'result') and response.result:
            return {
                "status": "success",
                "tool": tool_name,
                "result": response.result.model_dump()
            }
        else:
            return {
                "status": "error",
                "tool": tool_name,
                "message": "Tool execution failed",
                "error": response.error.model_dump() if hasattr(response, 'error') and response.error else None
            }
    except Exception as e:
        logger.exception(f"Error executing tool {tool_name}: {e}")
        return {
            "status": "error",
            "tool": tool_name,
            "message": str(e)
        }
