from fastapi import APIRouter

TOOLS_LIST = [
    {
        "name": "initialize",
        "description": "Initialize connection with the MCP server",
    },
    {
        "name": "asr_got.query",
        "description": "Run an Adaptive Graph of Thoughts reasoning query",
    },
    {
        "name": "shutdown",
        "description": "Shut down the MCP server",
    },
]

tools_router = APIRouter()


@tools_router.get("/tools", tags=["Tools"])
async def list_tools() -> dict[str, list[dict[str, str]]]:
    """Return a list of available MCP tools."""
    return {"tools": TOOLS_LIST}
