# This file makes 'adaptive_graph_of_thoughts' a Python package.
# You can define package-level imports or metadata here if needed later.

__version__ = "0.1.0"  # Should match pyproject.toml

from .async_server import AdaptiveGraphServer

__all__ = ["AdaptiveGraphServer"]
