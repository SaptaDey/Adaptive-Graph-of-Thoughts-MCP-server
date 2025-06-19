import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Any, Optional, Dict, List

# Import Neo4j related modules
from neo4j import Driver, GraphDatabase, Record, Result, Transaction
from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError, ClientError
from loguru import logger

# Import the module under test
from src.adaptive_graph_of_thoughts.domain.services.neo4j_utils import (
    Neo4jSettings,
    GlobalSettings,
    get_neo4j_settings,
    get_neo4j_driver,
    close_neo4j_driver,
    execute_query,
    _neo4j_settings,
    _driver
)

# Import config dependencies for mocking
from src.adaptive_graph_of_thoughts.config import runtime_settings

@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test to ensure test isolation."""
    import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
    neo4j_utils._neo4j_settings = None
    neo4j_utils._driver = None
    yield
    # Cleanup after test
    neo4j_utils._neo4j_settings = None
    if neo4j_utils._driver and not neo4j_utils._driver.closed:
        neo4j_utils._driver.close()
    neo4j_utils._driver = None

@pytest.fixture
def mock_runtime_settings():
    """Mock runtime settings with test configuration."""
    with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.runtime_settings') as mock_settings:
        mock_settings.neo4j.uri = "bolt://test:7687"
        mock_settings.neo4j.user = "test_user"
        mock_settings.neo4j.password = "test_password"
        mock_settings.neo4j.database = "test_db"
        yield mock_settings

@pytest.fixture
def mock_driver():
    """Mock Neo4j driver for testing."""
    driver = Mock(spec=Driver)
    driver.closed = False
    driver.verify_connectivity.return_value = None
    driver.close.return_value = None
    return driver

@pytest.fixture
def mock_session():
    """Mock Neo4j session for testing."""
    session = Mock()
    return session

@pytest.fixture
def mock_transaction():
    """Mock Neo4j transaction for testing."""
    transaction = Mock(spec=Transaction)
    return transaction

@pytest.fixture
def mock_result():
    """Mock Neo4j result for testing."""
    result = Mock(spec=Result)
    return result

@pytest.fixture
def sample_records():
    """Sample Neo4j records for testing."""
    record1 = Mock(spec=Record)
    record1.__getitem__ = Mock(side_effect=lambda key: f"value_{key}")
    record1.data.return_value = {"node_count": 42}

    record2 = Mock(spec=Record)
    record2.__getitem__ = Mock(side_effect=lambda key: f"value2_{key}")
    record2.data.return_value = {"name": "test_node"}

    return [record1, record2]

@pytest.fixture
def query_parameters():
    """Sample query parameters for testing."""
    return {
        "node_id": "test-node-123",
        "name": "Test Node",
        "value": 42,
        "active": True
    }

class TestNeo4jSettings:
    """Test cases for Neo4jSettings class."""

    def test_neo4j_settings_initialization_with_runtime_settings(self, mock_runtime_settings):
        """Test Neo4jSettings initialization using runtime_settings."""
        settings = Neo4jSettings()

        assert settings.uri == "bolt://test:7687"
        assert settings.user == "test_user"
        assert settings.password == "test_password"
        assert settings.database == "test_db"

    def test_neo4j_settings_attributes_access(self, mock_runtime_settings):
        """Test that all required attributes are accessible."""
        settings = Neo4jSettings()

        assert isinstance(settings.uri, str)
        assert isinstance(settings.user, str)
        assert isinstance(settings.password, str)
        assert isinstance(settings.database, str)

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.runtime_settings')
    def test_neo4j_settings_with_none_values(self, mock_settings):
        """Test Neo4jSettings when runtime_settings has None values."""
        mock_settings.neo4j.uri = None
        mock_settings.neo4j.user = None
        mock_settings.neo4j.password = None
        mock_settings.neo4j.database = None

        settings = Neo4jSettings()

        assert settings.uri is None
        assert settings.user is None
        assert settings.password is None
        assert settings.database is None

class TestGlobalSettings:
    """Test cases for GlobalSettings class."""

    def test_global_settings_initialization(self, mock_runtime_settings):
        """Test GlobalSettings initialization."""
        global_settings = GlobalSettings()

        assert global_settings.neo4j is not None
        assert isinstance(global_settings.neo4j, Neo4jSettings)

    def test_global_settings_neo4j_attribute_access(self, mock_runtime_settings):
        """Test accessing Neo4j settings through GlobalSettings."""
        global_settings = GlobalSettings()

        assert global_settings.neo4j.uri == "bolt://test:7687"
        assert global_settings.neo4j.user == "test_user"
        assert global_settings.neo4j.password == "test_password"
        assert global_settings.neo4j.database == "test_db"

    def test_multiple_global_settings_instances(self, mock_runtime_settings):
        """Test that multiple GlobalSettings instances work independently."""
        settings1 = GlobalSettings()
        settings2 = GlobalSettings()

        assert settings1.neo4j.uri == settings2.neo4j.uri
        assert settings1 is not settings2
        assert settings1.neo4j is not settings2.neo4j

class TestGetNeo4jSettings:
    """Test cases for get_neo4j_settings function."""

    def test_get_neo4j_settings_first_call_initialization(self, mock_runtime_settings):
        """Test first call to get_neo4j_settings initializes settings."""
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            settings = get_neo4j_settings()

            assert settings is not None
            assert isinstance(settings, GlobalSettings)
            assert isinstance(settings.neo4j, Neo4jSettings)

            mock_logger.info.assert_called_with("Initializing Neo4j settings.")
            mock_logger.debug.assert_called_once()
            debug_call = mock_logger.debug.call_args[0][0]
            assert "Neo4j Settings loaded" in debug_call
            assert "bolt://test:7687" in debug_call
            assert "test_user" in debug_call
            assert "test_db" in debug_call

    def test_get_neo4j_settings_singleton_behavior(self, mock_runtime_settings):
        """Test that get_neo4j_settings returns the same instance on subsequent calls."""
        settings1 = get_neo4j_settings()
        settings2 = get_neo4j_settings()
        assert settings1 is settings2

    def test_get_neo4j_settings_logging_only_on_first_call(self, mock_runtime_settings):
        """Test that logging only occurs on first initialization."""
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            get_neo4j_settings()
            mock_logger.reset_mock()
            get_neo4j_settings()
            mock_logger.info.assert_not_called()
            mock_logger.debug.assert_not_called()

    def test_get_neo4j_settings_with_missing_config(self):
        """Test get_neo4j_settings behavior when runtime_settings is unavailable."""
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.runtime_settings', side_effect=AttributeError("No runtime_settings")):
            with pytest.raises(AttributeError):
                get_neo4j_settings()

    def test_get_neo4j_settings_thread_safety(self, mock_runtime_settings):
        """Test that get_neo4j_settings is thread-safe."""
        import threading
        results = []

        def fetch():
            results.append(get_neo4j_settings())

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is results[0] for r in results)
        assert len(set(id(r) for r in results)) == 1

class TestGetNeo4jDriver:
    """Test cases for get_neo4j_driver function."""

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver')
    def test_get_neo4j_driver_successful_initialization(self, mock_graph_driver, mock_runtime_settings, mock_driver):
        """Test successful Neo4j driver initialization."""
        mock_graph_driver.return_value = mock_driver
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            driver = get_neo4j_driver()
            assert driver is mock_driver
            mock_graph_driver.assert_called_once_with(
                "bolt://test:7687",
                auth=("test_user", "test_password")
            )
            mock_driver.verify_connectivity.assert_called_once()
            mock_logger.info.assert_any_call("Initializing Neo4j driver for URI: bolt://test:7687")
            mock_logger.info.assert_any_call("Neo4j driver initialized and connectivity verified.")

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver')
    def test_get_neo4j_driver_singleton_behavior(self, mock_graph_driver, mock_runtime_settings, mock_driver):
        """Test that get_neo4j_driver returns the same instance on subsequent calls."""
        mock_graph_driver.return_value = mock_driver
        d1 = get_neo4j_driver()
        d2 = get_neo4j_driver()
        assert d1 is d2
        mock_graph_driver.assert_called_once()

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver')
    def test_get_neo4j_driver_reinitialize_after_close(self, mock_graph_driver, mock_runtime_settings):
        """Test driver reinitialization when previous driver was closed."""
        d1 = Mock(spec=Driver)
        d1.closed = False
        d1.verify_connectivity.return_value = None
        d2 = Mock(spec=Driver)
        d2.closed = False
        d2.verify_connectivity.return_value = None
        mock_graph_driver.side_effect = [d1, d2]

        first = get_neo4j_driver()
        assert first is d1
        d1.closed = True
        second = get_neo4j_driver()
        assert second is d2
        assert mock_graph_driver.call_count == 2

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver')
    def test_get_neo4j_driver_missing_neo4j_config(self, mock_graph_driver, mock_runtime_settings):
        """Test driver initialization when Neo4j config is missing."""
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_settings') as mock_get:
            mock_cfg = Mock()
            mock_cfg.neo4j = None
            mock_get.return_value = mock_cfg
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                with pytest.raises(ServiceUnavailable, match="Neo4j configuration is not available"):
                    get_neo4j_driver()
                mock_logger.error.assert_called_with("Neo4j configuration is missing in global settings.")

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver')
    def test_get_neo4j_driver_missing_connection_details(self, mock_graph_driver, mock_runtime_settings):
        """Test driver initialization with missing connection details."""
        mock_runtime_settings.neo4j.uri = ""
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            with pytest.raises(ServiceUnavailable, match="Neo4j connection details are incomplete"):
                get_neo4j_driver()
            mock_logger.error.assert_called_with("Neo4j URI, username, or password missing in configuration.")

        mock_runtime_settings.neo4j.uri = "bolt://test:7687"
        mock_runtime_settings.neo4j.user = ""
        with pytest.raises(ServiceUnavailable):
            get_neo4j_driver()

        mock_runtime_settings.neo4j.user = "test_user"
        mock_runtime_settings.neo4j.password = ""
        with pytest.raises(ServiceUnavailable):
            get_neo4j_driver()

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver')
    def test_get_neo4j_driver_connection_failure(self, mock_graph_driver, mock_runtime_settings):
        """Test driver initialization when connection fails."""
        drv = Mock(spec=Driver)
        drv.verify_connectivity.side_effect = ServiceUnavailable("Connection failed")
        mock_graph_driver.return_value = drv
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            with pytest.raises(ServiceUnavailable, match="Connection failed"):
                get_neo4j_driver()
            mock_logger.error.assert_called_with("Failed to connect to Neo4j at bolt://test:7687: Connection failed")

    @patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver')
    def test_get_neo4j_driver_unexpected_error(self, mock_graph_driver, mock_runtime_settings):
        """Test driver initialization with unexpected error."""
        mock_graph_driver.side_effect = RuntimeError("Unexpected error")
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            with pytest.raises(RuntimeError, match="Unexpected error"):
                get_neo4j_driver()
            mock_logger.error.assert_called_with("An unexpected error occurred while initializing Neo4j driver: Unexpected error")

    def test_get_neo4j_driver_global_state_cleanup_on_error(self, mock_runtime_settings):
        """Test that global driver state is cleaned up on error."""
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver', side_effect=ServiceUnavailable("Connection failed")):
            import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
            neo4j_utils._driver = None
            with pytest.raises(ServiceUnavailable):
                get_neo4j_driver()
            assert neo4j_utils._driver is None

class TestCloseNeo4jDriver:
    """Test cases for close_neo4j_driver function."""

    def test_close_neo4j_driver_with_open_driver(self, mock_runtime_settings):
        """Test closing an open Neo4j driver."""
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        mock_drv = Mock(spec=Driver)
        mock_drv.closed = False
        neo4j_utils._driver = mock_drv
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            close_neo4j_driver()
            mock_drv.close.assert_called_once()
            assert neo4j_utils._driver is None
            mock_logger.info.assert_called_with("Closing Neo4j driver.")

    def test_close_neo4j_driver_with_closed_driver(self, mock_runtime_settings):
        """Test closing an already closed Neo4j driver."""
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        mock_drv = Mock(spec=Driver)
        mock_drv.closed = True
        neo4j_utils._driver = mock_drv
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            close_neo4j_driver()
            mock_drv.close.assert_not_called()
            mock_logger.info.assert_called_with("Neo4j driver is already closed or not initialized.")

    def test_close_neo4j_driver_with_none_driver(self):
        """Test closing when no driver exists."""
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        neo4j_utils._driver = None
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
            close_neo4j_driver()
            mock_logger.info.assert_called_with("Neo4j driver is already closed or not initialized.")

    def test_close_neo4j_driver_exception_handling(self, mock_runtime_settings):
        """Test close_neo4j_driver handles exceptions gracefully."""
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        mock_drv = Mock(spec=Driver)
        mock_drv.closed = False
        mock_drv.close.side_effect = Exception("Close failed")
        neo4j_utils._driver = mock_drv
        close_neo4j_driver()
        mock_drv.close.assert_called_once()

    def test_close_neo4j_driver_thread_safety(self, mock_runtime_settings):
        """Test that close_neo4j_driver is thread-safe."""
        import threading
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        mock_drv = Mock(spec=Driver)
        mock_drv.closed = False
        neo4j_utils._driver = mock_drv

        def runner():
            close_neo4j_driver()

        threads = [threading.Thread(target=runner) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert neo4j_utils._driver is None
        assert mock_drv.close.call_count >= 1

class TestExecuteQuery:
    """Test cases for execute_query async function."""

    @pytest.mark.asyncio
    async def test_execute_query_successful_read(self, mock_runtime_settings, sample_records):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)

        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter(sample_records)
        mock_tx.run.return_value = mock_res

        def fake_read(fn):
            return list(fn(mock_tx))
        mock_sess.execute_read = fake_read

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                query = "MATCH (n) RETURN count(n) as node_count"
                params = {"limit": 100}

                result = await execute_query(query, params, database="test_db", tx_type="read")
                assert result == sample_records
                mock_drv.session.assert_called_once_with(database="test_db")
                assert any("Executing query on database 'test_db' with type 'read'" in str(c) for c in mock_logger.debug.call_args_list)
                assert any("Query executed successfully on database 'test_db'. Fetched 2 records." in str(c) for c in mock_logger.info.call_args_list)

    @pytest.mark.asyncio
    async def test_execute_query_successful_write(self, mock_runtime_settings, sample_records):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)

        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter(sample_records)
        mock_tx.run.return_value = mock_res

        def fake_write(fn):
            return list(fn(mock_tx))
        mock_sess.execute_write = fake_write

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            query = "CREATE (n:TestNode {name: $name}) RETURN n"
            params = {"name": "test"}

            result = await execute_query(query, params, tx_type="write")
            assert result == sample_records
            mock_tx.run.assert_called_once_with(query, params)

    @pytest.mark.asyncio
    async def test_execute_query_default_database(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)

        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            query = "MATCH (n) RETURN n"
            await execute_query(query)
            mock_drv.session.assert_called_once_with(database="test_db")

    @pytest.mark.asyncio
    async def test_execute_query_no_parameters(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)

        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            query = "MATCH (n) RETURN n"
            await execute_query(query)
            mock_tx.run.assert_called_once_with(query, None)

    @pytest.mark.asyncio
    async def test_execute_query_invalid_transaction_type(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                with pytest.raises(ValueError, match="Invalid transaction type: invalid. Must be 'read' or 'write'."):
                    await execute_query("MATCH (n) RETURN n", tx_type="invalid")
                mock_logger.error.assert_called_with("Invalid transaction type: invalid. Must be 'read' or 'write'.")
    
    @pytest.mark.asyncio
    async def test_execute_query_driver_not_available(self, mock_runtime_settings):
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=None):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                with pytest.raises(ServiceUnavailable, match="Neo4j driver not initialized or connection failed"):
                    await execute_query("MATCH (n) RETURN n")
                mock_logger.error.assert_called_with("Neo4j driver not available. Cannot execute query.")

    @pytest.mark.asyncio
    async def test_execute_query_neo4j_error(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        err = Neo4jError("Query failed")
        def fake_read(tx): raise err
        mock_sess.execute_read = fake_read

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                with pytest.raises(Neo4jError, match="Query failed"):
                    await execute_query("INVALID", {"test": "value"}, database="test_db")
                mock_logger.error.assert_any_call("Neo4j error executing Cypher query on database 'test_db': Query failed")
                mock_logger.error.assert_any_call(f"Query: INVALID, Parameters: {{'test': 'value'}}")

    @pytest.mark.asyncio
    async def test_execute_query_service_unavailable_error(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        err = ServiceUnavailable("Service unavailable")
        def fake_read(tx): raise err
        mock_sess.execute_read = fake_read

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                with pytest.raises(ServiceUnavailable, match="Service unavailable"):
                    await execute_query("MATCH (n) RETURN n", database="test_db")
                mock_logger.error.assert_called_with("Neo4j service became unavailable while attempting to execute query on 'test_db'.")

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_error(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        err = RuntimeError("Unexpected error")
        def fake_read(tx): raise err
        mock_sess.execute_read = fake_read

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                with pytest.raises(RuntimeError, match="Unexpected error"):
                    await execute_query("MATCH (n) RETURN n", {"test": "value"}, database="test_db")
                mock_logger.error.assert_any_call("Unexpected error executing Cypher query on database 'test_db': Unexpected error")
                mock_logger.error.assert_any_call(f"Query: MATCH (n) RETURN n, Parameters: {{'test': 'value'}}")

    @pytest.mark.asyncio
    async def test_execute_query_empty_result(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)

        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                result = await execute_query("MATCH (n:NonExistent) RETURN n")
                assert result == []
                assert any("Fetched 0 records" in str(c) for c in mock_logger.info.call_args_list)

class TestExecuteQueryIntegration:
    """Integration-style tests for execute_query function."""

    @pytest.mark.asyncio
    async def test_execute_query_full_workflow(self, mock_runtime_settings):
        mock_graph_driver = Mock()
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        mock_record = Mock(spec=Record)
        mock_record.data.return_value = {"count": 42}

        mock_graph_driver.return_value = mock_drv
        mock_drv.closed = False
        mock_drv.verify_connectivity.return_value = None
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([mock_record])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.GraphDatabase.driver', mock_graph_driver):
            result = await execute_query("MATCH (n) RETURN count(n) as count")
            assert result[0] is mock_record
            mock_graph_driver.assert_called_once()
            mock_drv.verify_connectivity.assert_called_once()
            mock_drv.session.assert_called_once()
            mock_tx.run.assert_called_once_with("MATCH (n) RETURN count(n) as count", None)

    @pytest.mark.asyncio
    async def test_execute_query_concurrent_execution(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            queries = [f"MATCH (n:Test{i}) RETURN n" for i in range(5)]
            results = await asyncio.gather(*(execute_query(q) for q in queries))
            assert all(r == [] for r in results)
            assert mock_tx.run.call_count == 5

    @pytest.mark.asyncio
    async def test_execute_query_with_complex_parameters(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_write = lambda fn: list(fn(mock_tx))

        complex_params = {
            "string_param": "test_string",
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"},
            "none_param": None
        }

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            await execute_query("CREATE (n:ComplexNode) SET n = $dict_param RETURN n", complex_params, tx_type="write")
            mock_tx.run.assert_called_once_with("CREATE (n:ComplexNode) SET n = $dict_param RETURN n", complex_params)

class TestEdgeCasesAndErrorRecovery:
    """Test edge cases and error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_execute_query_very_long_query(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        long_query = "MATCH " + " ".join(f"(n{i}:Node{i})" for i in range(100)) + " RETURN count(*)"
        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.logger') as mock_logger:
                await execute_query(long_query)
                assert any("..." in str(c) for c in mock_logger.debug.call_args_list)

    @pytest.mark.asyncio
    async def test_execute_query_special_characters(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_write = lambda fn: list(fn(mock_tx))

        special_params = {
            "unicode_text": "Hello 世界 🌍",
            "special_chars": "!@#$%^&*()[]{}|;':\",./<>?",
            "newlines": "Line 1\nLine 2\rLine 3\r\n",
            "tabs": "Column1\tColumn2\tColumn3"
        }

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            await execute_query("CREATE (n:SpecialNode) SET n = $unicode_text RETURN n", special_params, tx_type="write")
            mock_tx.run.assert_called_once_with("CREATE (n:SpecialNode) SET n = $unicode_text RETURN n", special_params)

    @pytest.mark.asyncio
    async def test_execute_query_memory_intensive(self, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        large_records = []
        for i in range(10000):
            rec = Mock(spec=Record)
            rec.data.return_value = {"id": f"node-{i}", "value": i}
            large_records.append(rec)

        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter(large_records)
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            result = await execute_query("MATCH (n) RETURN n")
            assert len(result) == 10000
            assert result[0] is large_records[0]
            assert result[-1] is large_records[-1]

class TestParameterizedScenarios:
    """Parameterized tests for various scenarios."""

    @pytest.mark.parametrize("tx_type,expected", [
        ("read", "execute_read"),
        ("write", "execute_write")
    ])
    @pytest.mark.asyncio
    async def test_execute_query_transaction_types(self, tx_type, expected, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        setattr(mock_sess, expected, lambda fn: list(fn(mock_tx)))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            await execute_query("MATCH (n) RETURN n", tx_type=tx_type)
            assert hasattr(mock_sess, expected)

    @pytest.mark.parametrize("database_name", [
        "neo4j",
        "custom_db",
        "test-database",
        "database_with_underscores",
        None
    ])
    @pytest.mark.asyncio
    async def test_execute_query_different_databases(self, database_name, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_tx = Mock(spec=Transaction)
        mock_res = Mock(spec=Result)
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        mock_res.__iter__.return_value = iter([])
        mock_tx.run.return_value = mock_res
        mock_sess.execute_read = lambda fn: list(fn(mock_tx))

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            await execute_query("MATCH (n) RETURN n", database=database_name)
            expected_db = database_name if database_name else "test_db"
            mock_drv.session.assert_called_once_with(database=expected_db)

    @pytest.mark.parametrize("err_type,err_msg", [
        (Neo4jError, "Generic Neo4j error"),
        (ClientError, "Client error occurred"),
        (TransientError, "Transient error occurred"),
        (ServiceUnavailable, "Service is unavailable")
    ])
    @pytest.mark.asyncio
    async def test_execute_query_various_neo4j_errors(self, err_type, err_msg, mock_runtime_settings):
        mock_drv = Mock(spec=Driver)
        mock_sess = Mock()
        mock_drv.session.return_value.__enter__.return_value = mock_sess
        error = err_type(err_msg)
        mock_sess.execute_read = lambda fn: (_ for _ in ()).throw(error)

        with patch('src.adaptive_graph_of_thoughts.domain.services.neo4j_utils.get_neo4j_driver', return_value=mock_drv):
            with pytest.raises(err_type, match=err_msg):
                await execute_query("MATCH (n) RETURN n")

class TestGlobalStateManagement:
    """Test global state management and cleanup."""

    def test_global_state_isolation_between_tests(self):
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        assert neo4j_utils._neo4j_settings is None
        assert neo4j_utils._driver is None

    def test_module_level_variables_exist(self):
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        assert hasattr(neo4j_utils, '_neo4j_settings')
        assert hasattr(neo4j_utils, '_driver')
        assert hasattr(neo4j_utils, 'Neo4jSettings')
        assert hasattr(neo4j_utils, 'GlobalSettings')

    def test_function_signatures(self):
        import inspect
        import src.adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils
        sig = inspect.signature(neo4j_utils.get_neo4j_settings)
        assert len(sig.parameters) == 0
        sig = inspect.signature(neo4j_utils.get_neo4j_driver)
        assert len(sig.parameters) == 0
        sig = inspect.signature(neo4j_utils.close_neo4j_driver)
        assert len(sig.parameters) == 0
        sig = inspect.signature(neo4j_utils.execute_query)
        expected = ["query", "parameters", "database", "tx_type"]
        assert list(sig.parameters.keys()) == expected

# Test markers for categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.neo4j,
    pytest.mark.database,
    pytest.mark.asyncio
]

# Additional configuration for async tests
pytest_plugins = ("pytest_asyncio",)