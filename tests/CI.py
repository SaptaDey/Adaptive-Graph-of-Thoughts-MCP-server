import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.adaptive_graph_of_thoughts.config import RuntimeSettings
from src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils import Neo4jConnection

@pytest.fixture
def mock_neo4j():
    return MagicMock(spec=Neo4jConnection)

@pytest.fixture
def test_settings():
    return RuntimeSettings(
        app={"name": "test", "version": "0.1.0"},
        neo4j={"uri": "bolt://localhost:7687", "user": "test", "password": "test"}
    )
