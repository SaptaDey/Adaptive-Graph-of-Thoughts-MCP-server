"""Asynchronous PostgreSQL database utilities."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg


class DatabaseManager:
    """Manage a connection pool and provide async query helpers."""

    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self, database_url: str) -> None:
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=5,
                command_timeout=10,
                server_settings={"application_name": "adaptive-graph-mcp"},
            )
            logging.info("Database pool initialized")
        except Exception as exc:  # pragma: no cover - rare
            logging.error(f"Database initialization failed: {exc}")
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Yield a connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        async with self.pool.acquire() as connection:
            yield connection

    async def fetch_evidence(self, query: str) -> list:
        """Run an evidence query with timeout handling."""
        try:
            async with self.get_connection() as conn:
                result = await asyncio.wait_for(
                    conn.fetch(
                        "SELECT * FROM evidence WHERE query_match($1)", query
                    ),
                    timeout=5.0,
                )
                return list(result)
        except asyncio.TimeoutError:
            logging.warning("Database query timed out")
            return []
