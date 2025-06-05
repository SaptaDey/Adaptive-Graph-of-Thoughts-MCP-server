import os
import sys

import uvicorn
from loguru import logger  # Import logger if you want to log here too

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.asr_got_reimagined.app_setup import create_app
from src.asr_got_reimagined.config import settings
from src.asr_got_reimagined.server_factory import MCPServerFactory

app = create_app()

async def run_server():
    """Run the appropriate server based on configuration."""
    # Auto-detect transport if configured
    if settings.app.mcp_transport_type == "auto":
        detected_mode = MCPServerFactory.detect_transport_mode()
        logger.info(f"Auto-detected transport mode: {detected_mode}")
        if detected_mode == "stdio":
            logger.info("Starting STDIO server...")
            await MCPServerFactory.run_stdio_server()
            return
    # STDIO-only if configured
    if settings.app.mcp_transport_type == "stdio" and not MCPServerFactory.should_run_http():
        logger.info("Starting STDIO-only server...")
        await MCPServerFactory.run_stdio_server()
        return
    # Default to HTTP server
    logger.info("Starting HTTP server using Uvicorn...")
    logger.info(
        "Host: {}, Port: {}, Log Level: {}",
        settings.app.host,
        settings.app.port,
        settings.app.log_level,
    )
    logger.info(f"Uvicorn reload mode: {settings.app.uvicorn_reload}")
    logger.info(f"Uvicorn workers: {settings.app.uvicorn_workers}")
    if MCPServerFactory.should_run_stdio():
        logger.info("STDIO transport also available - use main_stdio.py for STDIO mode")
    uvicorn.run(
        "src.asr_got_reimagined.main:app",
        host=settings.app.host,
        port=settings.app.port,
        log_level=settings.app.log_level.lower(),
        reload=settings.app.uvicorn_reload,
        workers=settings.app.uvicorn_workers,
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server())