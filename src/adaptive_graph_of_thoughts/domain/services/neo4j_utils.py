from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

_driver: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    """Return a singleton Neo4j driver instance."""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        logger.debug(f"Connecting to Neo4j at {uri} as {user}.")
        try:
            _driver = GraphDatabase.driver(uri, auth=(user, password))
            _driver.verify_connectivity()
        except Exception as e:  # pragma: no cover - just log
            logger.error(f"Failed to connect to Neo4j: {e}")
            if _driver is not None:
                _driver.close()
            _driver = None
            raise ServiceUnavailable("Could not connect to Neo4j")
    return _driver


def close_neo4j_driver() -> None:
    """Close the global Neo4j driver if it exists."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


async def execute_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    database: Optional[str] = None,
    tx_type: str = "read",
) -> List[Any]:
    """Execute a Cypher query asynchronously."""
    driver = get_neo4j_driver()
    session_kwargs = {"database": database} if database else {}
    async with driver.session(**session_kwargs) as session:
        if tx_type == "read":
            result = await session.execute_read(lambda tx: tx.run(query, parameters).data())
        elif tx_type == "write":
            result = await session.execute_write(lambda tx: tx.run(query, parameters).data())
        else:
            raise ValueError("tx_type must be 'read' or 'write'")
    return result
