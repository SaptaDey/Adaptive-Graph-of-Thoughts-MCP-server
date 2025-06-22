import asyncio
import logging

from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities, JSONRPCMessage
from mcp.shared.message import SessionMessage

from adaptive_graph_of_thoughts.config import settings
from adaptive_graph_of_thoughts.server_factory import MCPServerFactory
from adaptive_graph_of_thoughts.domain.services.got_processor import GoTProcessor


class AdaptiveGraphServer:
    """Minimal MCP stdio server for Adaptive Graph of Thoughts."""

    def __init__(self) -> None:
        self._processor: GoTProcessor | None = None

    async def initialize_resources(self) -> None:
        self._processor = GoTProcessor(settings=settings)

    async def run(self, read_stream, write_stream) -> None:
        if self._processor is None:
            raise RuntimeError("Server resources not initialized")
        async for session_message in read_stream:
            try:
                if isinstance(session_message, Exception):
                    logging.error("Session error: %s", session_message)
                    continue
                request_dict = session_message.message.model_dump()
                response = await MCPServerFactory._handle_stdio_request(
                    request_dict, self._processor
                )
                if response:
                    message = JSONRPCMessage.model_validate(response.model_dump())
                    await write_stream.send(SessionMessage(message))
                if request_dict.get("method") == "shutdown":
                    break
            except Exception as e:
                logging.exception("Error processing request: %s", e)
        await self._processor.shutdown_resources()


async def main() -> None:
    server = AdaptiveGraphServer()
    await server.initialize_resources()
    options = InitializationOptions(
        server_name="adaptive-graph-of-thoughts",
        server_version="1.0.0",
        capabilities=ServerCapabilities(),
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
