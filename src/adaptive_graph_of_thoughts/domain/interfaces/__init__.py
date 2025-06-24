"""Domain interfaces for dependency inversion."""

from .graph_repository import GraphRepository
from .evidence_provider import EvidenceProvider

__all__ = ["GraphRepository", "EvidenceProvider"]
