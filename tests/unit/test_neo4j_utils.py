import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from neo4j import GraphDatabase, Driver, Session, Record, Result
from neo4j.exceptions import ServiceUnavailable, ClientError, TransientError
import logging
from typing import Dict, List, Any
import json
from datetime import datetime

# Import the module under test
from adaptive_graph_of_thoughts.infrastructure.neo4j_utils import (
    Neo4jConnection,
    execute_query,
    create_node,
    update_node,
    delete_node,
    find_nodes,
    create_relationship,
    get_database_info,
    validate_connection,
    bulk_create_nodes,
    execute_cypher_file
)

@pytest.fixture
def mock_driver():
    """Mock Neo4j driver for testing."""
    driver = Mock(spec=Driver)
    return driver

@pytest.fixture
def mock_session():
    """Mock Neo4j session for testing."""
    session = Mock(spec=Session)
    return session

@pytest.fixture
def mock_result():
    """Mock Neo4j result for testing."""
    result = Mock(spec=Result)
    return result

@pytest.fixture
def mock_record():
    """Mock Neo4j record for testing."""
    record = Mock(spec=Record)
    return record

@pytest.fixture
def sample_node_data():
    """Sample node data for testing."""
    return {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com",
        "created_at": datetime.now().isoformat()
    }

@pytest.fixture
def sample_relationship_data():
    """Sample relationship data for testing."""
    return {
        "since": "2020-01-01",
        "strength": 0.8,
        "type": "friend"
    }

@pytest.fixture
def neo4j_connection():
    """Fixture for Neo4jConnection instance with mocked driver."""
    with patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver') as mock_driver_creator:
        mock_driver = Mock(spec=Driver)
        mock_session = Mock(spec=Session)
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.run.return_value = Mock(spec=Result)
        mock_driver.session.return_value = mock_session
        mock_driver_creator.return_value = mock_driver

        connection = Neo4jConnection("bolt://localhost:7687", "neo4j", "password", "testdb")
        connection._driver = mock_driver
        return connection