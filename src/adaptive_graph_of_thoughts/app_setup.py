import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import set_key
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from loguru import logger  # type: ignore
from neo4j import GraphDatabase

from src.adaptive_graph_of_thoughts.api.routes.mcp import mcp_router
from src.adaptive_graph_of_thoughts.config import runtime_settings, settings
from src.adaptive_graph_of_thoughts.domain.services.got_processor import (
    GoTProcessor,
)

# Add src directory to Python path if not already there
# This must be done before other project imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app startup and shutdown events.
    This replaces the deprecated @app.on_event decorators.
    """
    # Startup
    logger.info("Application startup sequence initiated.")
    # Any other async initializations can go here.
    logger.info("Application startup completed successfully.")

    yield  # This is where the app runs

    # Shutdown
    logger.info("Application shutdown sequence initiated.")
    # Clean up resources
    if hasattr(app.state, "got_processor") and hasattr(
        app.state.got_processor, "shutdown_resources"
    ):
        try:
            await app.state.got_processor.shutdown_resources()
        except Exception as e:
            logger.error(f"Error shutting down GoTProcessor: {e}")
    logger.info("Application shutdown completed.")


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application instance.

    Initializes logging, sets up CORS middleware based on allowed origins from settings, attaches a GoTProcessor to the application state, defines a health check endpoint, and mounts the MCP router.

    Returns:
        The configured FastAPI application instance.
    """
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.app.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    logger.info("Logger configured with level: {}", settings.app.log_level.upper())
    # Create FastAPI app with lifespan
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Adaptive Graph of Thoughts: Intelligent Scientific Reasoning through Graph-of-Thoughts MCP Server",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    # Store GoTProcessor instance on app.state
    app.state.got_processor = GoTProcessor(settings=settings)
    logger.info("GoTProcessor instance created and attached to app state.")

    # Process allowed origins from settings
    allowed_origins_str = settings.app.cors_allowed_origins_str
    if allowed_origins_str == "*":
        allowed_origins = ["*"]
    else:
        allowed_origins = [
            origin.strip()
            for origin in allowed_origins_str.split(",")
            if origin.strip()
        ]
        if not allowed_origins:  # Default if empty or only whitespace after split
            logger.warning(
                "APP_CORS_ALLOWED_ORIGINS_STR was not '*' and parsed to empty list. Defaulting to ['*']."
            )
            allowed_origins = [
                "*"
            ]  # Default to all if configuration is invalid or empty

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # Use the parsed list
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    logger.info(f"CORS middleware configured with origins: {allowed_origins}")

    # ----------------------- Setup Wizard -----------------------
    @app.get("/setup", response_class=HTMLResponse)
    async def setup_get(request: Request):
        values = {
            "uri": runtime_settings.neo4j.uri,
            "user": runtime_settings.neo4j.user,
            "password": runtime_settings.neo4j.password,
            "database": runtime_settings.neo4j.database,
        }
        return templates.TemplateResponse(
            "setup_neo4j.html",
            {"request": request, "values": values, "message": None},
        )

    @app.post("/setup", response_class=HTMLResponse)
    async def setup_post(
        request: Request,
        uri: str = Form(...),
        user: str = Form(...),
        password: str = Form(...),
        database: str = Form(...),
    ):
        if not _test_conn(uri, user, password, database):
            msg = "Failed to connect to Neo4j"
            values = {
                "uri": uri,
                "user": user,
                "password": password,
                "database": database,
            }
            return templates.TemplateResponse(
                "setup_neo4j.html",
                {
                    "request": request,
                    "values": values,
                    "message": msg,
                    "success": False,
                },
            )
        env_path = Path(".env")
        env_path.touch(mode=0o600, exist_ok=True)
        set_key(str(env_path), "NEO4J_URI", uri)
        set_key(str(env_path), "NEO4J_USER", user)
        set_key(str(env_path), "NEO4J_PASSWORD", password)
        set_key(str(env_path), "NEO4J_DATABASE", database)
        env_path.chmod(0o600)
        return RedirectResponse("/setup/settings", status_code=303)

    yaml_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    original_settings = yaml.safe_load(yaml_path.read_text()) or {}

    def _read_settings() -> dict[str, str]:
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data.get("app", {}))

    def _write_settings(data: dict[str, str]) -> None:
        with open(yaml_path) as fh:
            existing = yaml.safe_load(fh) or {}
        existing.setdefault("app", {}).update(data)
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(existing, fh)

    def _test_conn(uri: str, user: str, password: str, database: str) -> bool:
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session(database=database) as session:
                session.run("MATCH (n) RETURN count(n) LIMIT 1")
            driver.close()
            return True
        except Exception:
            return False

    @app.get("/setup/settings", response_class=HTMLResponse)
    async def edit_settings(request: Request):
        return templates.TemplateResponse(
            "setup_settings.html",
            {"request": request, "settings": _read_settings(), "message": None},
        )

    @app.post("/setup/settings", response_class=HTMLResponse)
    async def save_settings(request: Request):
        form = await request.form()
        data = {k: form[k] for k in form}
        _write_settings(data)
        return templates.TemplateResponse(
            "setup_settings.html",
            {"request": request, "settings": _read_settings(), "message": "Saved"},
        )

    @app.post("/setup/settings/reset", name="reset_settings")
    async def reset_settings() -> RedirectResponse:
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(original_settings, fh)
        return RedirectResponse("/setup/settings", status_code=303)

    # Add health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Return application and Neo4j status."""

        logger.debug("Health check endpoint was called.")  # type: ignore
        payload = {"status": "ok"}
        try:
            driver = GraphDatabase.driver(
                runtime_settings.neo4j.uri,
                auth=(runtime_settings.neo4j.user, runtime_settings.neo4j.password),
            )
            with driver.session(database=runtime_settings.neo4j.database) as session:
                session.run("RETURN 1")
            driver.close()
            payload["neo4j"] = "up"
            return payload
            payload["neo4j"] = "down"
            payload["status"] = "unhealthy" # Or a more descriptive status
            return JSONResponse(status_code=500, content=payload)

    # Include routers
    app.include_router(mcp_router, prefix="/mcp", tags=["MCP"])
    logger.info("API routers included. MCP router mounted at /mcp.")

    logger.info(
        "{} v{} application instance created.", settings.app.name, settings.app.version
    )
    return app
