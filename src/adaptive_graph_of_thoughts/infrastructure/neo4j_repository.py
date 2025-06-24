"""Neo4j graph repository implementation."""
from typing import Any, Optional

from ..domain.interfaces import GraphRepository
from . import neo4j_utils


class Neo4jGraphRepository(GraphRepository):
    """GraphRepository backed by Neo4j."""

    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        database: Optional[str] = None,
        tx_type: str = "read",
    ) -> Any:
        return await neo4j_utils.execute_query(
            query, parameters=parameters, database=database, tx_type=tx_type
        )
