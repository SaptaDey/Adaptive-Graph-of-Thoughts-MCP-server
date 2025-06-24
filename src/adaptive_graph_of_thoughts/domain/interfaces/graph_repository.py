"""Abstract graph database repository used by domain logic."""
from typing import Any, Protocol, Optional


class GraphRepository(Protocol):
    """Interface for graph database access."""

    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        database: Optional[str] = None,
        tx_type: str = "read",
    ) -> Any:
        """Execute a Cypher query and return driver records."""
        raise NotImplementedError
