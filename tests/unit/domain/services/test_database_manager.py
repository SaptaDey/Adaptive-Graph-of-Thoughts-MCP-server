import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.adaptive_graph_of_thoughts.domain.services.database_manager import DatabaseManager


@pytest.fixture
def db_manager() -> DatabaseManager:
    return DatabaseManager()


@pytest.mark.asyncio
async def test_initialize_creates_pool(db_manager):
    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        await db_manager.initialize("postgresql://localhost/test")
        mock_create.assert_awaited_once()
        assert db_manager.pool is mock_create.return_value


@pytest.mark.asyncio
async def test_get_connection_yields_connection(db_manager):
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    db_manager.pool = mock_pool

    async with db_manager.get_connection() as conn:
        assert conn is mock_conn
    mock_pool.acquire.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_evidence_handles_timeout(db_manager):
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_conn.fetch.side_effect = asyncio.TimeoutError
    db_manager.pool = mock_pool

    result = await db_manager.fetch_evidence("test")
    assert result == []
