import asyncio
import os
import sys

# Add src directory to Python path if not already there
# This allows running this script as a module directly for imports to work
# Corrected path logic for being inside src/asr_got_reimagined/
# when __file__ is src/asr_got_reimagined/main_stdio.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from loguru import logger

# Ensure settings are loaded before other project imports that might depend on them
try:
    # Attempt to import settings to ensure it's configured early
    from src.asr_got_reimagined.config import settings
    logger.info(f"Settings loaded. Initial MCP transport type: {settings.app.mcp_transport_type}")
except ImportError as e:
    logger.error(f"Failed to import settings: {e}. Ensure PYTHONPATH is correct or run as module.")
    # Depending on strictness, might exit here
except Exception as e:
    logger.error(f"Error loading settings: {e}")


# Now import the server factory
from src.asr_got_reimagined.server_factory import MCPServerFactory
# It's possible MCPServerFactory or its dependencies (like app_setup) also configure logging.
# Re-adding a basic stderr sink if no handlers exist or if log output isn't appearing.
if not logger._core.handlers:
    logger.add(sys.stderr, level="INFO")


async def main():
    logger.info("Attempting to start STDIO server via main_stdio.py...")
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
        sys.exit(1) # Exit with error if server fails to run

if __name__ == "__main__":
    # Configure logger minimally if it's not already configured by imports
    # This ensures that if this script is run directly, logs are visible.
    # Check if any handlers are configured for the root logger.
    # The logger in app_setup is comprehensive; here, just ensure something is present.
    if not logger._core.handlers: # Check if Loguru has any handlers
         logger.remove() # Remove default handler if any (usually none unless configured)
         logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO").upper())

    logger.info(f"Executing main_stdio.py with __name__ == '__main__'")
    asyncio.run(main())
