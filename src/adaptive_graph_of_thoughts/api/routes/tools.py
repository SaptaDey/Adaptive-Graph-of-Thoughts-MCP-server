from fastapi import APIRouter, Request

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
    {
        "name": "nlq",
        "description": "Translate natural language to Cypher and execute the query",
    },
    {
        "name": "graph_explorer",
        "description": "Retrieve a subgraph for visualization",
    },
]

tools_router = APIRouter()


@tools_router.get("/tools", tags=["Tools"])
async def list_tools(request: Request) -> dict[str, list[dict[str, str]]]:
    """Return a list of available MCP tools only when ready."""
    got_processor = getattr(request.app.state, "got_processor", None)
    if got_processor and not getattr(got_processor, "models_loaded", True):
        return {"tools": []}

    return {"tools": TOOLS_LIST}
