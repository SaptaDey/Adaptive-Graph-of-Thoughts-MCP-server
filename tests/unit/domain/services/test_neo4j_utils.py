"""
Comprehensive unit tests for neo4j_utils module.
Testing framework: pytest with pytest-asyncio for async support.
"""

import asyncio
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from typing import Any, Dict, List, Optional

import pytest
from neo4j import Driver, Record, Result, Transaction
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils import (
    Neo4jSettings,
    GlobalSettings,
    get_neo4j_settings,
    get_neo4j_driver,
    close_neo4j_driver,
    execute_query,
    _neo4j_settings,
    _driver,
)


@pytest.fixture
def mock_runtime_settings():
    """Mock runtime settings for Neo4j configuration."""
    with patch(
        "src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils.runtime_settings"
    ) as mock_settings:
        mock_settings.neo4j.uri = "bolt://localhost:7687"
        mock_settings.neo4j.user = "neo4j"
        mock_settings.neo4j.password = "password"
        mock_settings.neo4j.database = "neo4j"
        yield mock_settings


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j GraphDatabase driver."""
    with patch(
        "src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver"
    ) as mock_driver:
        mock_instance = Mock(spec=Driver)
        mock_instance.closed = False
        mock_instance.verify_connectivity.return_value = None
        mock_instance.close.return_value = None
        mock_driver.return_value = mock_instance
        yield mock_driver, mock_instance


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    import src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils as neo4j_utils

    neo4j_utils._neo4j_settings = None
    neo4j_utils._driver = None
    yield
    neo4j_utils._neo4j_settings = None
    neo4j_utils._driver = None


@pytest.fixture
def mock_logger():
    """Mock logger to avoid log output during tests."""
    with patch(
        "src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils.logger"
    ) as mock_log:
        yield mock_log


class TestNeo4jSettings:
    """Test cases for Neo4jSettings class."""

    def test_neo4j_settings_initialization(self, mock_runtime_settings):
        """Test successful initialization of Neo4jSettings."""
        settings = Neo4jSettings()
        assert settings.uri == "bolt://localhost:7687"
        assert settings.user == "neo4j"
        assert settings.password == "password"
        assert settings.database == "neo4j"

    def test_neo4j_settings_with_custom_values(self, mock_runtime_settings):
        """Test Neo4jSettings with custom runtime settings."""
        mock_runtime_settings.neo4j.uri = "bolt://custom:7687"
        mock_runtime_settings.neo4j.user = "custom_user"
        mock_runtime_settings.neo4j.password = "custom_pass"
        mock_runtime_settings.neo4j.database = "custom_db"

        settings = Neo4jSettings()
        assert settings.uri == "bolt://custom:7687"
        assert settings.user == "custom_user"
        assert settings.password == "custom_pass"
        assert settings.database == "custom_db"


class TestGlobalSettings:
    """Test cases for GlobalSettings class."""

    def test_global_settings_initialization(self, mock_runtime_settings):
        """Test successful initialization of GlobalSettings."""
        settings = GlobalSettings()
        assert isinstance(settings.neo4j, Neo4jSettings)
        assert settings.neo4j.uri == "bolt://localhost:7687"

    def test_global_settings_neo4j_attribute(self, mock_runtime_settings):
        """Test that GlobalSettings properly initializes neo4j attribute."""
        settings = GlobalSettings()
        assert hasattr(settings, "neo4j")
        assert isinstance(settings.neo4j, Neo4jSettings)


class TestGetNeo4jSettings:
    """Test cases for get_neo4j_settings function."""

    def test_get_neo4j_settings_first_call(self, mock_runtime_settings, mock_logger):
        """Test first call to get_neo4j_settings initializes settings."""
        settings = get_neo4j_settings()

        assert isinstance(settings, GlobalSettings)
        assert isinstance(settings.neo4j, Neo4jSettings)
        mock_logger.info.assert_called_with("Initializing Neo4j settings.")
        mock_logger.debug.assert_called_once()

    def test_get_neo4j_settings_singleton_behavior(
        self, mock_runtime_settings, mock_logger
    ):
        """Test that get_neo4j_settings returns the same instance on subsequent calls."""
        settings1 = get_neo4j_settings()
        settings2 = get_neo4j_settings()

        assert settings1 is settings2
        assert mock_logger.info.call_count == 1  # only first call logs init

    def test_get_neo4j_settings_global_state(self, mock_runtime_settings):
        """Test that get_neo4j_settings properly sets global state."""
        import src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils as neo4j_utils

        assert neo4j_utils._neo4j_settings is None

        settings = get_neo4j_settings()
        assert neo4j_utils._neo4j_settings is settings
        assert neo4j_utils._neo4j_settings is not None


class TestGetNeo4jDriver:
    """Test cases for get_neo4j_driver function."""

    def test_get_neo4j_driver_success(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test successful driver initialization."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        driver = get_neo4j_driver()

        assert driver is mock_driver_instance
        mock_driver_class.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
        )
        mock_driver_instance.verify_connectivity.assert_called_once()
        mock_logger.info.assert_any_call(
            "Initializing Neo4j driver for URI: bolt://localhost:7687"
        )
        mock_logger.info.assert_any_call(
            "Neo4j driver initialized and connectivity verified."
        )

    def test_get_neo4j_driver_singleton_behavior(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test that get_neo4j_driver returns the same instance on subsequent calls."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        driver1 = get_neo4j_driver()
        driver2 = get_neo4j_driver()

        assert driver1 is driver2
        assert driver1 is mock_driver_instance
        assert mock_driver_class.call_count == 1  # only first call creates driver

    def test_get_neo4j_driver_closed_driver_reinitializes(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test that a closed driver gets reinitialized."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        driver1 = get_neo4j_driver()
        mock_driver_instance.closed = True

        driver2 = get_neo4j_driver()
        assert mock_driver_class.call_count == 2
        assert driver2 is mock_driver_instance

    def test_get_neo4j_driver_service_unavailable_on_connectivity_check(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test ServiceUnavailable exception during connectivity verification."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        mock_driver_instance.verify_connectivity.side_effect = ServiceUnavailable(
            "Connection failed"
        )

        with pytest.raises(ServiceUnavailable):
            get_neo4j_driver()

        mock_logger.error.assert_called_with(
            "Failed to connect to Neo4j at bolt://localhost:7687: Connection failed"
        )
        import src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils as neo4j_utils

        assert neo4j_utils._driver is None

    def test_get_neo4j_driver_missing_neo4j_config(
        self, mock_runtime_settings, mock_logger
    ):
        """Test ServiceUnavailable when Neo4j config is missing."""
        with patch(
            "src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils.get_neo4j_settings"
        ) as mock_get_settings:
            mock_settings = Mock()
            mock_settings.neo4j = None
            mock_get_settings.return_value = mock_settings

            with pytest.raises(
                ServiceUnavailable, match="Neo4j configuration is not available"
            ):
                get_neo4j_driver()

            mock_logger.error.assert_called_with(
                "Neo4j configuration is missing in global settings."
            )

    @pytest.mark.parametrize(
        "missing_field,uri,username,password",
        [
            ("uri", "", "neo4j", "password"),
            ("username", "bolt://localhost:7687", "", "password"),
            ("password", "bolt://localhost:7687", "neo4j", ""),
            ("uri", None, "neo4j", "password"),
            ("username", "bolt://localhost:7687", None, "password"),
            ("password", "bolt://localhost:7687", "neo4j", None),
        ],
    )
    def test_get_neo4j_driver_incomplete_config(
        self, mock_runtime_settings, mock_logger, missing_field, uri, username, password
    ):
        """Test ServiceUnavailable when connection details are incomplete."""
        mock_runtime_settings.neo4j.uri = uri
        mock_runtime_settings.neo4j.user = username
        mock_runtime_settings.neo4j.password = password

        with pytest.raises(
            ServiceUnavailable, match="Neo4j connection details are incomplete"
        ):
            get_neo4j_driver()

        mock_logger.error.assert_called_with(
            "Neo4j URI, username, or password missing in configuration."
        )


class TestCloseNeo4jDriver:
    """Test cases for close_neo4j_driver function."""

    def test_close_neo4j_driver_success(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test successful driver closure."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        driver = get_neo4j_driver()
        assert driver is mock_driver_instance

        close_neo4j_driver()
        mock_driver_instance.close.assert_called_once()
        mock_logger.info.assert_any_call("Closing Neo4j driver.")

        import src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils as neo4j_utils

        assert neo4j_utils._driver is None

    def test_close_neo4j_driver_already_closed(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test closing already closed driver."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        driver = get_neo4j_driver()
        mock_driver_instance.closed = True

        close_neo4j_driver()
        mock_driver_instance.close.assert_not_called()
        mock_logger.info.assert_any_call(
            "Neo4j driver is already closed or not initialized."
        )

    def test_close_neo4j_driver_not_initialized(self, mock_logger):
        """Test closing driver when not initialized."""
        close_neo4j_driver()
        mock_logger.info.assert_called_with(
            "Neo4j driver is already closed or not initialized."
        )

    def test_close_neo4j_driver_none_driver(self, mock_logger):
        """Test closing when driver is None."""
        import src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils as neo4j_utils

        neo4j_utils._driver = None

        close_neo4j_driver()
        mock_logger.info.assert_called_with(
            "Neo4j driver is already closed or not initialized."
        )


class TestExecuteQuery:
    """Test cases for execute_query async function."""

    @pytest.mark.asyncio
    async def test_execute_query_read_success(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test successful read query execution."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None

        mock_record = Mock(spec=Record)
        mock_session.execute_read.return_value = [mock_record]

        result = await execute_query("MATCH (n) RETURN n", tx_type="read")

        assert result == [mock_record]
        mock_driver_instance.session.assert_called_once_with(database="neo4j")
        mock_session.execute_read.assert_called_once()
        mock_logger.info.assert_any_call(
            "Query executed successfully on database 'neo4j'. Fetched 1 records."
        )

    @pytest.mark.asyncio
    async def test_execute_query_write_success(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test successful write query execution."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None

        mock_record = Mock(spec=Record)
        mock_session.execute_write.return_value = [mock_record]

        result = await execute_query("CREATE (n:Test) RETURN n", tx_type="write")

        assert result == [mock_record]
        mock_session.execute_write.assert_called_once()
        mock_logger.info.assert_any_call(
            "Query executed successfully on database 'neo4j'. Fetched 1 records."
        )

    @pytest.mark.asyncio
    async def test_execute_query_with_parameters(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test query execution with parameters."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []

        parameters = {"name": "test", "age": 30}
        await execute_query(
            "MATCH (n) WHERE n.name = $name RETURN n", parameters=parameters
        )

        mock_session.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_custom_database(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test query execution with custom database."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []

        await execute_query("MATCH (n) RETURN n", database="custom_db")

        mock_driver_instance.session.assert_called_once_with(database="custom_db")

    @pytest.mark.asyncio
    async def test_execute_query_invalid_tx_type(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test query execution with invalid transaction type."""
        with pytest.raises(ValueError, match="Invalid transaction type: invalid"):
            await execute_query("MATCH (n) RETURN n", tx_type="invalid")

        mock_logger.error.assert_called_with(
            "Invalid transaction type: invalid. Must be 'read' or 'write'."
        )

    @pytest.mark.asyncio
    async def test_execute_query_neo4j_error(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test query execution with Neo4j error."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = Neo4jError("Query failed")

        with pytest.raises(Neo4jError):
            await execute_query("INVALID QUERY")

        mock_logger.error.assert_any_call(
            "Neo4j error executing Cypher query on database 'neo4j': Query failed"
        )

    @pytest.mark.asyncio
    async def test_execute_query_service_unavailable(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test query execution when service becomes unavailable."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = ServiceUnavailable(
            "Service unavailable"
        )

        with pytest.raises(ServiceUnavailable):
            await execute_query("MATCH (n) RETURN n")

        mock_logger.error.assert_called_with(
            "Neo4j service became unavailable while attempting to execute query on 'neo4j'."
        )

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_error(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test query execution with unexpected error."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception):
            await execute_query("MATCH (n) RETURN n")

        mock_logger.error.assert_any_call(
            "Unexpected error executing Cypher query on database 'neo4j': Unexpected error"
        )

    @pytest.mark.asyncio
    async def test_execute_query_empty_result(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test query execution returning empty results."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []

        result = await execute_query("MATCH (n) WHERE 1=0 RETURN n")

        assert result == []
        mock_logger.info.assert_any_call(
            "Query executed successfully on database 'neo4j'. Fetched 0 records."
        )

    @pytest.mark.asyncio
    async def test_execute_query_driver_not_available(self, mock_logger):
        """Test query execution when driver is not available."""
        with patch(
            "src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils.get_neo4j_driver"
        ) as mock_get_driver:
            mock_get_driver.return_value = None

            with pytest.raises(
                ServiceUnavailable, match="Neo4j driver not initialized"
            ):
                await execute_query("MATCH (n) RETURN n")

            mock_logger.error.assert_called_with(
                "Neo4j driver not available. Cannot execute query."
            )


class TestIntegrationScenarios:
    """Integration-style tests for complex scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow_read_query(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test complete workflow from settings to query execution."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None

        mock_record = Mock(spec=Record)
        mock_record.__getitem__.return_value = 5
        mock_session.execute_read.return_value = [mock_record]

        result = await execute_query("MATCH (n) RETURN count(n) as count")

        assert len(result) == 1
        assert result[0] is mock_record
        mock_driver_class.assert_called_once()
        mock_driver_instance.verify_connectivity.assert_called_once()
        mock_session.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_driver_reinitialization_after_close(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test that driver can be reinitialized after being closed."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        driver1 = get_neo4j_driver()
        close_neo4j_driver()
        driver2 = get_neo4j_driver()

        assert mock_driver_class.call_count == 2
        assert driver2 is mock_driver_instance

    @pytest.mark.asyncio
    async def test_concurrent_query_execution(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test concurrent query execution using the same driver."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session1 = Mock()
        mock_session2 = Mock()
        mock_driver_instance.session.side_effect = [
            mock_session1.__enter__.return_value,
            mock_session2.__enter__.return_value,
        ]

        mock_session1.__enter__.return_value = mock_session1
        mock_session1.__exit__.return_value = None
        mock_session1.execute_read.return_value = [Mock(spec=Record)]

        mock_session2.__enter__.return_value = mock_session2
        mock_session2.__exit__.return_value = None
        mock_session2.execute_read.return_value = [Mock(spec=Record)]

        tasks = [
            execute_query("MATCH (n:User) RETURN n"),
            execute_query("MATCH (n:Product) RETURN n"),
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_input,expected_error",
        [
            (None, AttributeError),
            ("", None),
            ("   ", None),
        ],
    )
    async def test_execute_query_invalid_inputs(
        self, mock_runtime_settings, mock_neo4j_driver, query_input, expected_error
    ):
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []

        if expected_error:
            with pytest.raises(expected_error):
                await execute_query(query_input)
        else:
            result = await execute_query(query_input)
            assert result == []

    @pytest.mark.asyncio
    async def test_execute_query_large_parameter_set(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []

        large_params = {f"param_{i}": f"value_{i}" for i in range(100)}
        result = await execute_query("MATCH (n) RETURN n", parameters=large_params)

        assert result == []
        mock_session.execute_read.assert_called_once()

    def test_settings_with_special_characters(self, mock_runtime_settings):
        mock_runtime_settings.neo4j.uri = "bolt://localhost:7687"
        mock_runtime_settings.neo4j.user = "user@domain.com"
        mock_runtime_settings.neo4j.password = "pass!@#$%^&*()"
        mock_runtime_settings.neo4j.database = "test-db_123"

        settings = get_neo4j_settings()

        assert settings.neo4j.uri == "bolt://localhost:7687"
        assert settings.neo4j.user == "user@domain.com"
        assert settings.neo4j.password == "pass!@#$%^&*()"
        assert settings.neo4j.database == "test-db_123"

    @pytest.mark.asyncio
    async def test_transaction_timeout_handling(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        mock_driver_class, mock_driver_instance = mock_neo4j_driver

        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = Neo4jError("Transaction timeout")

        with pytest.raises(Neo4jError):
            await execute_query("MATCH (n) RETURN n")

        mock_logger.error.assert_any_call(
            "Neo4j error executing Cypher query on database 'neo4j': Transaction timeout"
        )


class TestNeo4jDriverConnectionManagement:
    """Test advanced connection management scenarios."""

    def test_get_neo4j_driver_with_custom_connection_params(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test driver initialization with custom connection parameters."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        # Modify settings to test custom parameters
        mock_runtime_settings.neo4j.uri = "bolt+s://secure.neo4j.com:7687"
        
        driver = get_neo4j_driver()
        
        mock_driver_class.assert_called_once_with(
            "bolt+s://secure.neo4j.com:7687",
            auth=("neo4j", "password"),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
        )

    def test_get_neo4j_driver_connection_pool_exhaustion(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test behavior when connection pool is exhausted."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        mock_driver_instance.verify_connectivity.side_effect = [
            None,  # First call succeeds
            ServiceUnavailable("Connection pool exhausted")  # Second call fails
        ]
        
        # First call should succeed
        driver1 = get_neo4j_driver()
        assert driver1 is mock_driver_instance
        
        # Reset driver to force reconnection
        import src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils as neo4j_utils
        neo4j_utils._driver = None
        
        # Second call should raise exception
        with pytest.raises(ServiceUnavailable):
            get_neo4j_driver()

    def test_get_neo4j_driver_authentication_failure(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test authentication failure during driver initialization."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        mock_driver_instance.verify_connectivity.side_effect = Neo4jError("Authentication failed")
        
        with pytest.raises(Neo4jError):
            get_neo4j_driver()

    @pytest.mark.parametrize(
        "uri_format",
        [
            "neo4j://localhost:7687",
            "neo4j+s://localhost:7687", 
            "bolt://localhost:7687",
            "bolt+s://localhost:7687",
            "bolt+ssc://localhost:7687"
        ]
    )
    def test_get_neo4j_driver_various_uri_formats(
        self, mock_runtime_settings, mock_neo4j_driver, uri_format
    ):
        """Test driver initialization with various URI formats."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        mock_runtime_settings.neo4j.uri = uri_format
        
        driver = get_neo4j_driver()
        
        mock_driver_class.assert_called_once()
        args, kwargs = mock_driver_class.call_args
        assert args[0] == uri_format


class TestQueryExecutionAdvancedScenarios:
    """Advanced test scenarios for query execution."""

    @pytest.mark.asyncio
    async def test_execute_query_with_complex_parameters(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test query execution with complex parameter types."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        
        complex_params = {
            "list_param": [1, 2, 3, "string"],
            "dict_param": {"nested": {"value": 42}},
            "none_param": None,
            "bool_param": True,
            "float_param": 3.14159,
        }
        
        result = await execute_query(
            "MATCH (n) WHERE n.id IN $list_param RETURN n", 
            parameters=complex_params
        )
        
        assert result == []
        mock_session.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_transaction_rollback_simulation(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test behavior during transaction rollback scenarios."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_write.side_effect = Neo4jError("Transaction was rolled back")
        
        with pytest.raises(Neo4jError):
            await execute_query(
                "CREATE (n:Test {name: $name}) RETURN n", 
                parameters={"name": "test"}, 
                tx_type="write"
            )
        
        mock_logger.error.assert_any_call(
            "Neo4j error executing Cypher query on database 'neo4j': Transaction was rolled back"
        )

    @pytest.mark.asyncio
    async def test_execute_query_constraint_violation(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test behavior when query violates database constraints."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_write.side_effect = Neo4jError("Node already exists with label `User` and property `email`")
        
        with pytest.raises(Neo4jError):
            await execute_query(
                "CREATE (u:User {email: $email}) RETURN u", 
                parameters={"email": "duplicate@example.com"}, 
                tx_type="write"
            )

    @pytest.mark.asyncio
    async def test_execute_query_memory_error_simulation(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test behavior when query causes memory issues."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = Neo4jError("There is not enough memory to perform the current task")
        
        with pytest.raises(Neo4jError):
            await execute_query("MATCH (n)-[*1..10]-(m) RETURN n, m")

    @pytest.mark.asyncio
    async def test_execute_query_syntax_error(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test behavior with malformed Cypher queries."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = Neo4jError("Invalid input 'INVALID'")
        
        with pytest.raises(Neo4jError):
            await execute_query("INVALID CYPHER SYNTAX")

    @pytest.mark.asyncio
    async def test_execute_query_very_long_query(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test execution of very long Cypher queries."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        
        # Create a very long query
        long_query = "MATCH (n) WHERE " + " OR ".join([f"n.prop{i} = $param{i}" for i in range(100)]) + " RETURN n"
        long_params = {f"param{i}": f"value{i}" for i in range(100)}
        
        result = await execute_query(long_query, parameters=long_params)
        
        assert result == []
        mock_session.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_unicode_handling(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test query execution with Unicode characters."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        
        unicode_params = {
            "chinese": "你好世界",
            "emoji": "🚀🌟💻",
            "arabic": "مرحبا بالعالم",
            "special_chars": "àáâãäåæçèéêë"
        }
        
        result = await execute_query(
            "MATCH (n) WHERE n.name = $chinese RETURN n", 
            parameters=unicode_params
        )
        
        assert result == []


class TestConnectionRecoveryAndResilience:
    """Test connection recovery and resilience scenarios."""

    @pytest.mark.asyncio
    async def test_execute_query_connection_lost_during_execution(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test behavior when connection is lost during query execution."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = ServiceUnavailable("Connection to the database terminated")
        
        with pytest.raises(ServiceUnavailable):
            await execute_query("MATCH (n) RETURN n")

    def test_driver_reinitialization_multiple_times(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test multiple driver reinitializations."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        # Initialize and close driver multiple times
        for i in range(3):
            driver = get_neo4j_driver()
            assert driver is mock_driver_instance
            close_neo4j_driver()
        
        assert mock_driver_class.call_count == 3

    def test_concurrent_driver_access(self, mock_runtime_settings, mock_neo4j_driver):
        """Test concurrent access to driver initialization."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        # Simulate concurrent calls
        drivers = []
        for _ in range(5):
            drivers.append(get_neo4j_driver())
        
        # All should return the same instance (singleton behavior)
        assert all(driver is mock_driver_instance for driver in drivers)
        assert mock_driver_class.call_count == 1


class TestConfigurationValidation:
    """Test configuration validation scenarios."""

    @pytest.mark.parametrize(
        "invalid_uri",
        [
            "invalid-protocol://localhost:7687",
            "bolt://",
            "://localhost:7687",
            "bolt://localhost",
            "bolt://localhost:invalid-port",
            "bolt://localhost:-1",
            "bolt://localhost:99999",
        ]
    )
    def test_neo4j_settings_invalid_uri_formats(self, mock_runtime_settings, invalid_uri):
        """Test Neo4jSettings with various invalid URI formats."""
        mock_runtime_settings.neo4j.uri = invalid_uri
        
        # The settings should still initialize (validation happens at driver level)
        settings = Neo4jSettings()
        assert settings.uri == invalid_uri

    def test_neo4j_settings_missing_runtime_settings(self):
        """Test Neo4jSettings when runtime_settings is not available."""
        with patch(
            "src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils.runtime_settings",
            side_effect=AttributeError("runtime_settings not available")
        ):
            with pytest.raises(AttributeError):
                Neo4jSettings()

    def test_global_settings_with_none_neo4j_config(self, mock_runtime_settings):
        """Test GlobalSettings when neo4j config is None."""
        mock_runtime_settings.neo4j = None
        
        with pytest.raises(AttributeError):
            GlobalSettings()


class TestPerformanceAndStress:
    """Performance and stress testing scenarios."""

    @pytest.mark.asyncio
    async def test_execute_query_rapid_succession(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test rapid succession of query executions."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        
        # Execute many queries in rapid succession
        tasks = []
        for i in range(20):
            tasks.append(execute_query(f"MATCH (n) WHERE n.id = {i} RETURN n"))
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 20
        assert all(result == [] for result in results)

    @pytest.mark.asyncio
    async def test_execute_query_large_result_set_simulation(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test handling of large result sets."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        
        # Simulate large result set
        large_result = [Mock(spec=Record) for _ in range(10000)]
        mock_session.execute_read.return_value = large_result
        
        result = await execute_query("MATCH (n) RETURN n")
        
        assert len(result) == 10000
        mock_logger.info.assert_any_call(
            "Query executed successfully on database 'neo4j'. Fetched 10000 records."
        )


class TestErrorRecoveryAndCleanup:
    """Test error recovery and resource cleanup scenarios."""

    def test_close_neo4j_driver_exception_during_close(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test driver closure when close() raises an exception."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        driver = get_neo4j_driver()
        mock_driver_instance.close.side_effect = Exception("Error during close")
        
        # Should not raise exception, should handle gracefully
        close_neo4j_driver()
        
        mock_logger.info.assert_any_call("Closing Neo4j driver.")

    @pytest.mark.asyncio
    async def test_execute_query_session_creation_failure(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test behavior when session creation fails."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        mock_driver_instance.session.side_effect = ServiceUnavailable("Cannot create session")
        
        with pytest.raises(ServiceUnavailable):
            await execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_execute_query_session_context_manager_failure(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test behavior when session context manager fails."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value = mock_session
        mock_session.__enter__.side_effect = Exception("Context manager failed")
        
        with pytest.raises(Exception):
            await execute_query("MATCH (n) RETURN n")


class TestDatabaseSwitching:
    """Test database switching scenarios."""

    @pytest.mark.asyncio
    async def test_execute_query_multiple_databases(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test query execution across multiple databases."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        
        databases = ["neo4j", "analytics", "staging", "test_db"]
        
        for db in databases:
            await execute_query("MATCH (n) RETURN n", database=db)
            
        # Verify session was created with each database
        expected_calls = [Mock(database=db) for db in databases]
        assert mock_driver_instance.session.call_count == len(databases)

    @pytest.mark.asyncio
    async def test_execute_query_nonexistent_database(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test query execution on non-existent database."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.side_effect = Neo4jError("Database does not exist. Database name: 'nonexistent'")
        
        with pytest.raises(Neo4jError):
            await execute_query("MATCH (n) RETURN n", database="nonexistent")


class TestAdvancedMockingScenarios:
    """Test advanced mocking and edge case scenarios."""

    def test_global_state_isolation_between_tests(self):
        """Verify that global state is properly isolated between tests."""
        import src.adaptive_graph_of_thoughts.infrastructure.neo4j_utils as neo4j_utils
        
        # This test verifies the reset_global_state fixture works correctly
        assert neo4j_utils._neo4j_settings is None
        assert neo4j_utils._driver is None

    @pytest.mark.asyncio
    async def test_execute_query_partial_session_mock_failure(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test behavior when session mock is partially configured."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        # Incomplete session mock - missing __exit__
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        # Intentionally not setting __exit__ to test error handling
        mock_driver_instance.session.return_value.__exit__ = None
        
        with pytest.raises(AttributeError):
            await execute_query("MATCH (n) RETURN n")

    def test_neo4j_settings_repr_and_str(self, mock_runtime_settings):
        """Test string representations of Neo4jSettings."""
        settings = Neo4jSettings()
        
        # Test that str() and repr() don't expose sensitive information
        str_repr = str(settings)
        repr_str = repr(settings)
        
        # Password should not be in string representation
        assert "password" not in str_repr.lower()
        assert "password" not in repr_str.lower()

    @pytest.mark.parametrize(
        "tx_type_variant",
        ["READ", "Read", "rEaD", "WRITE", "Write", "wRiTe"]
    )
    @pytest.mark.asyncio
    async def test_execute_query_case_insensitive_tx_type(
        self, mock_runtime_settings, mock_neo4j_driver, tx_type_variant
    ):
        """Test that transaction type is case-insensitive."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        mock_session.execute_write.return_value = []
        
        result = await execute_query("MATCH (n) RETURN n", tx_type=tx_type_variant)
        
        assert result == []
        # Either execute_read or execute_write should be called based on normalized type
        assert (mock_session.execute_read.called or mock_session.execute_write.called)


class TestExtremeEdgeCases:
    """Test extreme edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_execute_query_with_none_database(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test query execution with None as database parameter."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        
        result = await execute_query("MATCH (n) RETURN n", database=None)
        
        assert result == []
        # Should use default database from settings when None is passed
        mock_driver_instance.session.assert_called_once_with(database="neo4j")

    @pytest.mark.asyncio
    async def test_execute_query_with_empty_parameters_dict(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test query execution with explicitly empty parameters."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_read.return_value = []
        
        result = await execute_query("MATCH (n) RETURN n", parameters={})
        
        assert result == []
        mock_session.execute_read.assert_called_once()

    def test_neo4j_settings_attribute_access_after_initialization(self, mock_runtime_settings):
        """Test that Neo4jSettings attributes are accessible after initialization."""
        settings = Neo4jSettings()
        
        # Test attribute access doesn't raise exceptions
        assert hasattr(settings, 'uri')
        assert hasattr(settings, 'user')
        assert hasattr(settings, 'password')
        assert hasattr(settings, 'database')
        
        # Test attributes are not None
        assert settings.uri is not None
        assert settings.user is not None
        assert settings.password is not None
        assert settings.database is not None

    @pytest.mark.asyncio
    async def test_execute_query_timeout_scenarios(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test various timeout scenarios during query execution."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        
        timeout_errors = [
            "Connection acquisition timed out in 60 seconds",
            "Transaction execution timed out",
            "Query execution timed out after 30 seconds"
        ]
        
        for timeout_error in timeout_errors:
            mock_session.execute_read.side_effect = Neo4jError(timeout_error)
            
            with pytest.raises(Neo4jError):
                await execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_execute_query_concurrent_modification_error(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test handling of concurrent modification errors."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_write.side_effect = Neo4jError("Unable to acquire exclusive schema lock")
        
        with pytest.raises(Neo4jError):
            await execute_query(
                "CREATE CONSTRAINT ON (n:User) ASSERT n.email IS UNIQUE", 
                tx_type="write"
            )

    @pytest.mark.asyncio
    async def test_execute_query_quota_exceeded_error(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test handling of quota exceeded errors."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        mock_session.execute_write.side_effect = Neo4jError("Database quota exceeded")
        
        with pytest.raises(Neo4jError):
            await execute_query(
                "CREATE (n:LargeNode {data: $data})", 
                parameters={"data": "x" * 10000}, 
                tx_type="write"
            )


class TestResultProcessingAndValidation:
    """Test result processing and validation scenarios."""

    @pytest.mark.asyncio
    async def test_execute_query_result_record_processing(
        self, mock_runtime_settings, mock_neo4j_driver
    ):
        """Test processing of different record types and structures."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        
        # Create complex mock records
        mock_record1 = Mock(spec=Record)
        mock_record1.keys.return_value = ["id", "name", "properties"]
        mock_record1.values.return_value = [1, "test", {"key": "value"}]
        
        mock_record2 = Mock(spec=Record)
        mock_record2.keys.return_value = ["count"]
        mock_record2.values.return_value = [42]
        
        mock_session.execute_read.return_value = [mock_record1, mock_record2]
        
        result = await execute_query("MATCH (n) RETURN n.id, n.name, n")
        
        assert len(result) == 2
        assert result[0] is mock_record1
        assert result[1] is mock_record2

    @pytest.mark.asyncio
    async def test_execute_query_mixed_success_failure_batch(
        self, mock_runtime_settings, mock_neo4j_driver, mock_logger
    ):
        """Test batch operations with mixed success and failure scenarios."""
        mock_driver_class, mock_driver_instance = mock_neo4j_driver
        
        mock_session = Mock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_instance.session.return_value.__exit__.return_value = None
        
        # Simulate partial success in batch operations
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [Mock(spec=Record)]
            elif call_count[0] == 2:
                raise Neo4jError("Batch item 2 failed")
            else:
                return []
        
        mock_session.execute_write.side_effect = side_effect
        
        # First call succeeds
        result1 = await execute_query("CREATE (n:Test1) RETURN n", tx_type="write")
        assert len(result1) == 1
        
        # Second call fails
        with pytest.raises(Neo4jError):
            await execute_query("CREATE (n:Test2) RETURN n", tx_type="write")
        
        # Third call succeeds
        result3 = await execute_query("CREATE (n:Test3) RETURN n", tx_type="write")
        assert len(result3) == 0

