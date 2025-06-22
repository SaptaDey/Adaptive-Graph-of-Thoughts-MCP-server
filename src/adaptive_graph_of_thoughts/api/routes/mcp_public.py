from fastapi import APIRouter, Request
from adaptive_graph_of_thoughts.api.routes.mcp import mcp_endpoint_handler

# Public MCP router without authentication
mcp_public_router = APIRouter()


@mcp_public_router.post("")
async def public_mcp_endpoint(request_payload, http_request: Request):
    """Public MCP endpoint for Smithery integration"""
    return await mcp_endpoint_handler(request_payload, http_request)


@mcp_public_router.get("")
async def mcp_server_info():
    """MCP server discovery endpoint"""
    return {
        "name": "Adaptive Graph of Thoughts MCP Server",
        "version": "0.1.0",
        "mcp_version": "2024-11-05",
        "capabilities": {
            "tools": [
                {"name": "initialize", "description": "Initialize connection"},
                {"name": "asr_got.query", "description": "Execute reasoning query"},
                {"name": "shutdown", "description": "Shutdown server"},
            ]
        },
    }
