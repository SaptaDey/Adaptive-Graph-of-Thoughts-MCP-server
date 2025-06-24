"""Interface for evidence search providers."""
from typing import Protocol, Any


class EvidenceProvider(Protocol):
    """A search provider used to gather evidence articles."""

    async def search(self, query: str, num_results: int = 10) -> list[Any]:
        """Return search results for the given query."""
        ...
