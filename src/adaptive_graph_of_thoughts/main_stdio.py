import asyncio
import logging

from adaptive_graph_of_thoughts.config import settings
from adaptive_graph_of_thoughts.server_factory import MCPServerFactory


async def main() -> None:
    """Main entry point for STDIO MCP server."""
    # Configure logging to file to avoid interfering with STDIO communication
    logging.basicConfig(
        filename="stdio_server.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    
    logging.info("Starting MCP STDIO server...")
    
    # Run the STDIO server using the factory method
    try:
        await MCPServerFactory.run_stdio_server()
    except Exception as e:
        logging.exception("STDIO server failed: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(main())