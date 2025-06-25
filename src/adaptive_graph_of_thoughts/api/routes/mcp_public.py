from fastapi import APIRouter, Request
from typing import Any
import json
import os
from loguru import logger

# Import the handlers but not the authentication dependency
from adaptive_graph_of_thoughts.api.routes.mcp import (
    handle_initialize,
    handle_asr_got_query,
    handle_shutdown,
    create_jsonrpc_error
)
from adaptive_graph_of_thoughts.api.schemas import (
    JSONRPCRequest,
    MCPInitializeParams,
    MCPASRGoTQueryParams,
    ShutdownParams
)
from pydantic import ValidationError

# Public MCP router without authentication for Smithery
mcp_public_router = APIRouter()

@mcp_public_router.post("/")
async def public_mcp_endpoint(
    request_payload: JSONRPCRequest[dict[str, Any]],
    http_request: Request
):
    """Public MCP endpoint for Smithery and other clients without authentication."""
    logger.debug(
        "Public MCP Endpoint received request: method={}, id={}",
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

    except ValidationError as ve:
        logger.warning(f"MCP Validation Error for method {method}: {ve}")
        return create_jsonrpc_error(
            request_id=req_id,
            code=-32602,
            message="Invalid parameters.",
            data={"details": ve.errors(), "method": method},
        )
    except Exception as e:
        logger.exception("Error in public MCP endpoint for method {}: {}", method, e)
        return create_jsonrpc_error(
            request_id=req_id,
            code=-32603,
            message=f"Internal error processing request for method '{method}'.",
            data={"details": str(e), "method": method},
        )

@mcp_public_router.get("/tools")
async def public_get_available_tools() -> dict[str, Any]:
    """Public endpoint to get available MCP tools for client discovery."""
    # Load tools definition from smithery.yaml or tools definition file
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
        # Fallback to basic tool definitions
        return {
            "status": "success",
            "tools": [
                {
                    "name": "scientific_reasoning_query",
                    "description": "Perform advanced scientific reasoning using the Adaptive Graph of Thoughts framework",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The scientific question or research query to analyze"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "analyze_research_hypothesis",
                    "description": "Analyze and evaluate research hypotheses with confidence scoring",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "hypothesis": {
                                "type": "string",
                                "description": "The research hypothesis to analyze"
                            }
                        },
                        "required": ["hypothesis"]
                    }
                }
            ],
            "resources": [],
            "prompts": []
        }

@mcp_public_router.get("/capabilities")
async def public_get_server_capabilities() -> dict[str, Any]:
    """Public endpoint to get server capabilities for MCP client negotiation."""
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

@mcp_public_router.get("/info")
async def public_mcp_server_info() -> dict[str, Any]:
    """MCP server discovery endpoint for Smithery."""
    return {
        "name": "Adaptive Graph of Thoughts MCP Server",
        "version": "0.1.0",
        "mcp_version": "2024-11-05",
        "description": "Advanced scientific reasoning through graph-based analysis with evidence integration",
        "capabilities": {
            "tools": True,
            "resources": True,
            "prompts": True,
            "graph_reasoning": True,
            "evidence_integration": True,
            "confidence_scoring": True
        },
        "endpoints": {
            "mcp": "/mcp",
            "tools": "/mcp/tools",
            "capabilities": "/mcp/capabilities"
        }
    }