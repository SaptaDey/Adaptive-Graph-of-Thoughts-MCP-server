import asyncio
import os
import sys

# Add src directory to Python path if not already there
# This allows running this script as a module directly for imports to work
# Corrected path logic for being inside src/adaptive_graph_of_thoughts/
# when __file__ is src/adaptive_graph_of_thoughts/main_stdio.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from loguru import logger  # noqa: E402

# Ensure settings are loaded before other project imports that might depend on them
try:
    # Attempt to import settings to ensure it's configured early
    from src.adaptive_graph_of_thoughts.config import settings

    logger.info(
        f"Settings loaded. Initial MCP transport type: {settings.app.mcp_transport_type}"
    )
except ImportError as e:
    logger.error(
        f"Failed to import settings: {e}. Ensure PYTHONPATH is correct or run as module."
    )
    # Depending on strictness, might exit here
except Exception as e:
    logger.error(f"Error loading settings: {e}")


# Now import the server factory
from src.adaptive_graph_of_thoughts.server_factory import MCPServerFactory  # noqa: E402

# It's possible MCPServerFactory or its dependencies (like app_setup) also configure logging.
# Re-adding a basic stderr sink if no handlers exist or if log output isn't appearing.
# Redirect all logs to a file for stdio mode to keep stderr clean for mcp-inspector
logger.remove()  # Remove existing handlers
logger.add(
    "stdio_server.log", level=os.getenv("LOG_LEVEL", "INFO").upper(), rotation="10 MB"
)


async def main():
    logger.info(
        "Attempting to start STDIO server via main_stdio.py... (logging to stdio_server.log)"
    )
    # This script is intended to *force* STDIO mode or be the dedicated entry for it.
    # We could override settings here if MCPServerFactory.run_stdio_server()
    # doesn't already imply/force STDIO mode.
    # For now, assume run_stdio_server() is sufficient.
    # settings.app.mcp_transport_type = "stdio" # Example if override needed
    try:
        await MCPServerFactory.run_stdio_server()
        logger.info("STDIO server finished.")
    except Exception as e:
        logger.opt(exception=True).error(f"Error running STDIO server: {e}")
        sys.exit(1)  # Exit with error if server fails to run


if __name__ == "__main__":
    # Configure logger minimally if it's not already configured by imports
    # This ensures that if this script is run directly, logs are visible.
    # Check if any handlers are configured for the root logger.
    # The logger in app_setup is comprehensive; here, just ensure something is present.
    # Ensure logger is configured if running as main, now redirecting to file
    # logger.remove() # Already removed above
    # logger.add("stdio_main.log", level=os.getenv("LOG_LEVEL", "INFO").upper())

    logger.info(
        "Executing main_stdio.py with __name__ == '__main__' (logging to stdio_server.log)"
    )
    asyncio.run(main())
