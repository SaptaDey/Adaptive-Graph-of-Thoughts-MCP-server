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

class TestNeo4jConnection:
    """Test suite for Neo4jConnection class."""
    
    def test_init_with_valid_parameters(self, mock_driver):
        """Test Neo4jConnection initialization with valid parameters."""
        with patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver') as mock_driver_creator:
            mock_driver_creator.return_value = mock_driver
            
            connection = Neo4jConnection("bolt://localhost:7687", "neo4j", "password", "testdb")
            
            assert connection.uri == "bolt://localhost:7687"
            assert connection.user == "neo4j"
            assert connection.password == "password"
            assert connection.database == "testdb"
            mock_driver_creator.assert_called_once_with("bolt://localhost:7687", auth=("neo4j", "password"))

    def test_init_with_default_database(self, mock_driver):
        """Test Neo4jConnection initialization with default database."""
        with patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver') as mock_driver_creator:
            mock_driver_creator.return_value = mock_driver
            
            connection = Neo4jConnection("bolt://localhost:7687", "neo4j", "password")
            
            assert connection.database == "neo4j"

    def test_init_connection_failure(self):
        """Test Neo4jConnection initialization with connection failure."""
        with patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver') as mock_driver_creator:
            mock_driver_creator.side_effect = ServiceUnavailable("Database unavailable")
            
            with pytest.raises(ServiceUnavailable):
                Neo4jConnection("bolt://localhost:7687", "neo4j", "password")

    def test_close_connection(self, neo4j_connection):
        """Test closing Neo4j connection."""
        neo4j_connection.close()
        neo4j_connection._driver.close.assert_called_once()

    def test_context_manager_usage(self, mock_driver):
        """Test Neo4jConnection as context manager."""
        with patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver') as mock_driver_creator:
            mock_driver_creator.return_value = mock_driver
            
            with Neo4jConnection("bolt://localhost:7687", "neo4j", "password") as conn:
                assert conn._driver is not None
            
            mock_driver.close.assert_called_once()

    def test_get_session(self, neo4j_connection, mock_session):
        """Test getting session from connection."""
        neo4j_connection._driver.session.return_value = mock_session
        
        session = neo4j_connection.get_session()
        
        assert session == mock_session
        neo4j_connection._driver.session.assert_called_once_with(database="testdb")

    def test_get_session_with_different_database(self, neo4j_connection, mock_session):
        """Test getting session with different database."""
        neo4j_connection._driver.session.return_value = mock_session
        
        session = neo4j_connection.get_session(database="different_db")
        
        neo4j_connection._driver.session.assert_called_once_with(database="different_db")

    def test_invalid_uri_format(self):
        """Test Neo4jConnection with invalid URI format."""
        with patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver') as mock_driver_creator:
            mock_driver_creator.side_effect = ValueError("Invalid URI format")
            
            with pytest.raises(ValueError):
                Neo4jConnection("invalid://uri", "neo4j", "password")

    def test_authentication_failure(self):
        """Test Neo4jConnection with authentication failure."""
        with patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.GraphDatabase.driver') as mock_driver_creator:
            mock_driver_creator.side_effect = ClientError("Authentication failed")
            
            with pytest.raises(ClientError):
                Neo4jConnection("bolt://localhost:7687", "wrong_user", "wrong_pass")


class TestExecuteQuery:
    """Test suite for execute_query function."""
    
    def test_execute_query_success(self, neo4j_connection, mock_session, mock_result):
        """Test successful query execution."""
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        query = "MATCH (n) RETURN n LIMIT 10"
        parameters = {"limit": 10}
        
        result = execute_query(neo4j_connection, query, parameters)
        
        assert result == mock_result
        mock_session.run.assert_called_once_with(query, parameters)

    def test_execute_query_with_no_parameters(self, neo4j_connection, mock_session, mock_result):
        """Test query execution without parameters."""
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        query = "MATCH (n) RETURN count(n)"
        
        result = execute_query(neo4j_connection, query)
        
        mock_session.run.assert_called_once_with(query, {})

    def test_execute_query_client_error(self, neo4j_connection, mock_session):
        """Test query execution with client error."""
        mock_session.run.side_effect = ClientError("Invalid query")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            execute_query(neo4j_connection, "INVALID QUERY")

    def test_execute_query_transient_error_retry(self, neo4j_connection, mock_session, mock_result):
        """Test query execution with transient error and retry."""
        mock_session.run.side_effect = [TransientError("Temporary failure"), mock_result]
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = execute_query(neo4j_connection, "MATCH (n) RETURN n", retries=2)
        
        assert result == mock_result
        assert mock_session.run.call_count == 2

    def test_execute_query_max_retries_exceeded(self, neo4j_connection, mock_session):
        """Test query execution exceeding max retries."""
        mock_session.run.side_effect = TransientError("Persistent failure")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(TransientError):
            execute_query(neo4j_connection, "MATCH (n) RETURN n", retries=2)
        
        assert mock_session.run.call_count == 3  # Initial + 2 retries

    def test_execute_query_empty_query(self, neo4j_connection, mock_session):
        """Test query execution with empty query string."""
        mock_session.run.side_effect = ClientError("Empty query")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            execute_query(neo4j_connection, "")

    def test_execute_query_with_complex_parameters(self, neo4j_connection, mock_session, mock_result):
        """Test query execution with complex parameter types."""
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        complex_params = {
            "string_list": ["a", "b", "c"],
            "nested_dict": {"key": {"nested": "value"}},
            "datetime": datetime.now().isoformat(),
            "boolean": True,
            "null_value": None
        }
        
        result = execute_query(neo4j_connection, "MATCH (n) WHERE n.id IN $string_list RETURN n", complex_params)
        
        assert result == mock_result
        mock_session.run.assert_called_once_with("MATCH (n) WHERE n.id IN $string_list RETURN n", complex_params)


class TestCreateNode:
    """Test suite for create_node function."""
    
    def test_create_node_success(self, neo4j_connection, mock_session, mock_result, sample_node_data):
        """Test successful node creation."""
        mock_record = Mock()
        mock_record.get.return_value = "node123"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        node_id = create_node(neo4j_connection, "Person", sample_node_data)
        
        assert node_id == "node123"
        mock_session.run.assert_called_once()
        query_call = mock_session.run.call_args[0][0]
        assert "CREATE" in query_call
        assert "Person" in query_call

    def test_create_node_with_empty_properties(self, neo4j_connection, mock_session, mock_result):
        """Test creating node with empty properties."""
        mock_record = Mock()
        mock_record.get.return_value = "node456"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        node_id = create_node(neo4j_connection, "EmptyNode", {})
        
        assert node_id == "node456"

    def test_create_node_with_none_properties(self, neo4j_connection, mock_session, mock_result):
        """Test creating node with None properties."""
        mock_record = Mock()
        mock_record.get.return_value = "node789"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        node_id = create_node(neo4j_connection, "NullNode", None)
        
        assert node_id == "node789"

    def test_create_node_creation_failure(self, neo4j_connection, mock_session):
        """Test node creation failure."""
        mock_session.run.side_effect = ClientError("Constraint violation")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            create_node(neo4j_connection, "Person", {"name": "John"})

    def test_create_node_invalid_label(self, neo4j_connection, mock_session):
        """Test creating node with invalid label."""
        mock_session.run.side_effect = ClientError("Invalid label")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            create_node(neo4j_connection, "", {"name": "John"})

    def test_create_node_with_multiple_labels(self, neo4j_connection, mock_session, mock_result):
        """Test creating node with multiple labels."""
        mock_record = Mock()
        mock_record.get.return_value = "multi_label_node"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        node_id = create_node(neo4j_connection, ["Person", "Employee"], {"name": "John"})
        
        assert node_id == "multi_label_node"
        query_call = mock_session.run.call_args[0][0]
        assert "Person" in query_call and "Employee" in query_call

    def test_create_node_with_special_characters(self, neo4j_connection, mock_session, mock_result):
        """Test creating node with special characters in properties."""
        mock_record = Mock()
        mock_record.get.return_value = "special_node"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        special_data = {
            "name": "John O'Reilly",
            "description": 'Contains "quotes" and \'apostrophes\'',
            "unicode": "José García 中文 😀"
        }
        
        node_id = create_node(neo4j_connection, "SpecialNode", special_data)
        
        assert node_id == "special_node"


class TestUpdateNode:
    """Test suite for update_node function."""
    
    def test_update_node_success(self, neo4j_connection, mock_session, mock_result, sample_node_data):
        """Test successful node update."""
        mock_result.consume.return_value = Mock(counters=Mock(properties_set=2))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = update_node(neo4j_connection, "node123", sample_node_data)
        
        assert result is True
        mock_session.run.assert_called_once()
        query_call = mock_session.run.call_args[0][0]
        assert "MATCH" in query_call
        assert "SET" in query_call

    def test_update_node_not_found(self, neo4j_connection, mock_session, mock_result):
        """Test updating non-existent node."""
        mock_result.consume.return_value = Mock(counters=Mock(properties_set=0))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = update_node(neo4j_connection, "nonexistent", {"name": "New Name"})
        
        assert result is False

    def test_update_node_with_empty_properties(self, neo4j_connection, mock_session, mock_result):
        """Test updating node with empty properties."""
        mock_result.consume.return_value = Mock(counters=Mock(properties_set=0))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = update_node(neo4j_connection, "node123", {})
        
        assert result is True  # Should succeed even with empty properties

    def test_update_node_constraint_violation(self, neo4j_connection, mock_session):
        """Test node update with constraint violation."""
        mock_session.run.side_effect = ClientError("Constraint violation")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            update_node(neo4j_connection, "node123", {"email": "duplicate@email.com"})

    def test_update_node_partial_update(self, neo4j_connection, mock_session, mock_result):
        """Test partial node property update."""
        mock_result.consume.return_value = Mock(counters=Mock(properties_set=1))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = update_node(neo4j_connection, "node123", {"age": 31})
        
        assert result is True
        query_call = mock_session.run.call_args[0][0]
        assert "age" in query_call

    def test_update_node_remove_property(self, neo4j_connection, mock_session, mock_result):
        """Test removing a property by setting it to None."""
        mock_result.consume.return_value = Mock(counters=Mock(properties_set=1))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = update_node(neo4j_connection, "node123", {"old_property": None})
        
        assert result is True


class TestDeleteNode:
    """Test suite for delete_node function."""
    
    def test_delete_node_success(self, neo4j_connection, mock_session, mock_result):
        """Test successful node deletion."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_deleted=1))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = delete_node(neo4j_connection, "node123")
        
        assert result is True
        mock_session.run.assert_called_once()
        query_call = mock_session.run.call_args[0][0]
        assert "MATCH" in query_call
        assert "DELETE" in query_call

    def test_delete_node_not_found(self, neo4j_connection, mock_session, mock_result):
        """Test deleting non-existent node."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_deleted=0))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = delete_node(neo4j_connection, "nonexistent")
        
        assert result is False

    def test_delete_node_with_relationships(self, neo4j_connection, mock_session, mock_result):
        """Test deleting node and its relationships."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_deleted=1, relationships_deleted=3))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = delete_node(neo4j_connection, "node123", delete_relationships=True)
        
        assert result is True
        query_call = mock_session.run.call_args[0][0]
        assert "DETACH DELETE" in query_call

    def test_delete_node_constraint_error(self, neo4j_connection, mock_session):
        """Test node deletion with constraint error."""
        mock_session.run.side_effect = ClientError("Cannot delete node with relationships")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            delete_node(neo4j_connection, "node123")

    def test_delete_multiple_nodes_by_id_list(self, neo4j_connection, mock_session, mock_result):
        """Test deleting multiple nodes by providing a list of IDs."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_deleted=3))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        result = delete_node(neo4j_connection, ["node1", "node2", "node3"])
        
        assert result is True
        query_call = mock_session.run.call_args[0][0]
        assert "WHERE" in query_call and "IN" in query_call


class TestFindNodes:
    """Test suite for find_nodes function."""
    
    def test_find_nodes_success(self, neo4j_connection, mock_session, mock_result):
        """Test successful node finding."""
        mock_records = [
            Mock(data={"n": {"id": "node1", "name": "John"}}),
            Mock(data={"n": {"id": "node2", "name": "Jane"}})
        ]
        mock_result.list.return_value = mock_records
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes = find_nodes(neo4j_connection, "Person", {"age": 30})
        
        assert len(nodes) == 2
        assert nodes[0]["id"] == "node1"
        assert nodes[1]["id"] == "node2"
        mock_session.run.assert_called_once()

    def test_find_nodes_no_results(self, neo4j_connection, mock_session, mock_result):
        """Test finding nodes with no results."""
        mock_result.list.return_value = []
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes = find_nodes(neo4j_connection, "Person", {"name": "Nonexistent"})
        
        assert nodes == []

    def test_find_nodes_with_limit(self, neo4j_connection, mock_session, mock_result):
        """Test finding nodes with limit."""
        mock_records = [Mock(data={"n": {"id": "node1"}})]
        mock_result.list.return_value = mock_records
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes = find_nodes(neo4j_connection, "Person", {}, limit=5)
        
        query_call = mock_session.run.call_args[0][0]
        assert "LIMIT" in query_call

    def test_find_nodes_all_labels(self, neo4j_connection, mock_session, mock_result):
        """Test finding nodes without label filter."""
        mock_records = [Mock(data={"n": {"id": "node1"}})]
        mock_result.list.return_value = mock_records
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes = find_nodes(neo4j_connection, None, {})
        
        query_call = mock_session.run.call_args[0][0]
        assert "MATCH (n)" in query_call

    def test_find_nodes_with_complex_filter(self, neo4j_connection, mock_session, mock_result):
        """Test finding nodes with complex filter conditions."""
        mock_records = [Mock(data={"n": {"id": "node1", "age": 25, "active": True}})]
        mock_result.list.return_value = mock_records
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        complex_filter = {
            "age": {"$gte": 18, "$lt": 65},
            "active": True,
            "department": {"$in": ["IT", "Engineering"]}
        }
        
        nodes = find_nodes(neo4j_connection, "Employee", complex_filter)
        
        assert len(nodes) == 1

    def test_find_nodes_with_sorting(self, neo4j_connection, mock_session, mock_result):
        """Test finding nodes with sorting."""
        mock_records = [
            Mock(data={"n": {"id": "node1", "name": "Alice"}}),
            Mock(data={"n": {"id": "node2", "name": "Bob"}})
        ]
        mock_result.list.return_value = mock_records
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes = find_nodes(neo4j_connection, "Person", {}, order_by="name")
        
        query_call = mock_session.run.call_args[0][0]
        assert "ORDER BY" in query_call


class TestCreateRelationship:
    """Test suite for create_relationship function."""
    
    def test_create_relationship_success(self, neo4j_connection, mock_session, mock_result, sample_relationship_data):
        """Test successful relationship creation."""
        mock_record = Mock()
        mock_record.get.return_value = "rel123"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        rel_id = create_relationship(neo4j_connection, "node1", "node2", "KNOWS", sample_relationship_data)
        
        assert rel_id == "rel123"
        mock_session.run.assert_called_once()
        query_call = mock_session.run.call_args[0][0]
        assert "MATCH" in query_call
        assert "CREATE" in query_call
        assert "KNOWS" in query_call

    def test_create_relationship_nodes_not_found(self, neo4j_connection, mock_session, mock_result):
        """Test relationship creation when nodes don't exist."""
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ValueError, match="One or both nodes not found"):
            create_relationship(neo4j_connection, "nonexistent1", "nonexistent2", "KNOWS", {})

    def test_create_relationship_empty_properties(self, neo4j_connection, mock_session, mock_result):
        """Test creating relationship with empty properties."""
        mock_record = Mock()
        mock_record.get.return_value = "rel456"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        rel_id = create_relationship(neo4j_connection, "node1", "node2", "LIKES", {})
        
        assert rel_id == "rel456"

    def test_create_relationship_duplicate_error(self, neo4j_connection, mock_session):
        """Test relationship creation with duplicate constraint error."""
        mock_session.run.side_effect = ClientError("Duplicate relationship")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            create_relationship(neo4j_connection, "node1", "node2", "KNOWS", {})

    def test_create_bidirectional_relationship(self, neo4j_connection, mock_session, mock_result):
        """Test creating bidirectional relationship."""
        mock_record = Mock()
        mock_record.get.return_value = "rel_bidirectional"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        rel_id = create_relationship(neo4j_connection, "node1", "node2", "FRIENDS", {"mutual": True}, bidirectional=True)
        
        assert rel_id == "rel_bidirectional"
        # Should create two relationships
        assert mock_session.run.call_count >= 1

    def test_create_relationship_with_weight(self, neo4j_connection, mock_session, mock_result):
        """Test creating weighted relationship."""
        mock_record = Mock()
        mock_record.get.return_value = "weighted_rel"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        rel_props = {"weight": 0.85, "confidence": 0.92}
        rel_id = create_relationship(neo4j_connection, "node1", "node2", "SIMILAR_TO", rel_props)
        
        assert rel_id == "weighted_rel"


class TestGetDatabaseInfo:
    """Test suite for get_database_info function."""
    
    def test_get_database_info_success(self, neo4j_connection, mock_session, mock_result):
        """Test successful database info retrieval."""
        mock_records = [
            Mock(data={"name": "neo4j", "status": "online", "role": "primary"}),
            Mock(data={"name": "system", "status": "online", "role": "primary"})
        ]
        mock_result.list.return_value = mock_records
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        info = get_database_info(neo4j_connection)
        
        assert len(info) == 2
        assert info[0]["name"] == "neo4j"
        assert info[1]["name"] == "system"

    def test_get_database_info_empty_result(self, neo4j_connection, mock_session, mock_result):
        """Test database info retrieval with empty result."""
        mock_result.list.return_value = []
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        info = get_database_info(neo4j_connection)
        
        assert info == []

    def test_get_database_info_access_denied(self, neo4j_connection, mock_session):
        """Test database info retrieval with access denied."""
        mock_session.run.side_effect = ClientError("Access denied")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            get_database_info(neo4j_connection)

    def test_get_database_info_with_statistics(self, neo4j_connection, mock_session, mock_result):
        """Test database info retrieval including statistics."""
        mock_records = [
            Mock(data={
                "name": "neo4j", 
                "status": "online", 
                "node_count": 1000, 
                "relationship_count": 2500,
                "size_mb": 45.2
            })
        ]
        mock_result.list.return_value = mock_records
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        info = get_database_info(neo4j_connection, include_statistics=True)
        
        assert info[0]["node_count"] == 1000
        assert info[0]["relationship_count"] == 2500


class TestValidateConnection:
    """Test suite for validate_connection function."""
    
    def test_validate_connection_success(self, neo4j_connection, mock_session, mock_result):
        """Test successful connection validation."""
        mock_record = Mock()
        mock_record.get.return_value = 1
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        is_valid = validate_connection(neo4j_connection)
        
        assert is_valid is True
        mock_session.run.assert_called_once_with("RETURN 1 as test")

    def test_validate_connection_failure(self, neo4j_connection, mock_session):
        """Test connection validation failure."""
        mock_session.run.side_effect = ServiceUnavailable("Database unavailable")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        is_valid = validate_connection(neo4j_connection)
        
        assert is_valid is False

    def test_validate_connection_unexpected_result(self, neo4j_connection, mock_session, mock_result):
        """Test connection validation with unexpected result."""
        mock_record = Mock()
        mock_record.get.return_value = None
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        is_valid = validate_connection(neo4j_connection)
        
        assert is_valid is False

    def test_validate_connection_timeout(self, neo4j_connection, mock_session):
        """Test connection validation with timeout."""
        mock_session.run.side_effect = TransientError("Timeout")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        is_valid = validate_connection(neo4j_connection)
        
        assert is_valid is False

    def test_validate_connection_with_custom_query(self, neo4j_connection, mock_session, mock_result):
        """Test connection validation with custom validation query."""
        mock_record = Mock()
        mock_record.get.return_value = "online"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        is_valid = validate_connection(neo4j_connection, validation_query="CALL db.ping() YIELD status RETURN status")
        
        assert is_valid is True

    def test_validate_connection_permission_check(self, neo4j_connection, mock_session, mock_result):
        """Test connection validation with permission verification."""
        mock_record = Mock()
        mock_record.get.return_value = ["read", "write"]
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        is_valid = validate_connection(neo4j_connection, check_permissions=True)
        
        assert is_valid is True


class TestBulkCreateNodes:
    """Test suite for bulk_create_nodes function."""
    
    def test_bulk_create_nodes_success(self, neo4j_connection, mock_session, mock_result):
        """Test successful bulk node creation."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=3))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes_data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ]
        
        count = bulk_create_nodes(neo4j_connection, "Person", nodes_data)
        
        assert count == 3
        mock_session.run.assert_called_once()
        query_call = mock_session.run.call_args[0][0]
        assert "UNWIND" in query_call
        assert "CREATE" in query_call

    def test_bulk_create_nodes_empty_list(self, neo4j_connection, mock_session, mock_result):
        """Test bulk node creation with empty list."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=0))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        count = bulk_create_nodes(neo4j_connection, "Person", [])
        
        assert count == 0

    def test_bulk_create_nodes_with_batch_size(self, neo4j_connection, mock_session, mock_result):
        """Test bulk node creation with custom batch size."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=2))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes_data = [{"name": f"Person{i}"} for i in range(5)]
        
        count = bulk_create_nodes(neo4j_connection, "Person", nodes_data, batch_size=2)
        
        assert count == 6  # 3 batches * 2 nodes per batch
        assert mock_session.run.call_count == 3  # Should be called 3 times for 3 batches

    def test_bulk_create_nodes_constraint_violation(self, neo4j_connection, mock_session):
        """Test bulk node creation with constraint violation."""
        mock_session.run.side_effect = ClientError("Constraint violation")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes_data = [{"name": "Duplicate"}]
        
        with pytest.raises(ClientError):
            bulk_create_nodes(neo4j_connection, "Person", nodes_data)

    def test_bulk_create_nodes_large_dataset(self, neo4j_connection, mock_session, mock_result):
        """Test bulk node creation with large dataset."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=1000))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        # Create 10,000 nodes
        nodes_data = [{"id": i, "name": f"Node{i}"} for i in range(10000)]
        
        count = bulk_create_nodes(neo4j_connection, "TestNode", nodes_data, batch_size=1000)
        
        assert count == 10000  # 10 batches * 1000 nodes per batch
        assert mock_session.run.call_count == 10

    def test_bulk_create_nodes_with_relationships(self, neo4j_connection, mock_session, mock_result):
        """Test bulk node creation that also creates relationships."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=3, relationships_created=2))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes_data = [
            {"name": "Parent", "type": "parent"},
            {"name": "Child1", "type": "child", "parent_name": "Parent"},
            {"name": "Child2", "type": "child", "parent_name": "Parent"}
        ]
        
        count = bulk_create_nodes(neo4j_connection, "Family", nodes_data, create_relationships=True)
        
        assert count == 3

    def test_bulk_create_nodes_with_validation(self, neo4j_connection, mock_session, mock_result):
        """Test bulk node creation with data validation."""
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=2))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes_data = [
            {"name": "Valid Node", "age": 25},
            {"name": "Another Valid", "age": 30},
            {"invalid": "missing required fields"}  # This should be filtered out
        ]
        
        validation_schema = {"required": ["name", "age"]}
        count = bulk_create_nodes(neo4j_connection, "Person", nodes_data, validation_schema=validation_schema)
        
        assert count == 2  # Only 2 valid nodes created


class TestExecuteCypherFile:
    """Test suite for execute_cypher_file function."""
    
    def test_execute_cypher_file_success(self, neo4j_connection, mock_session, mock_result):
        """Test successful Cypher file execution."""
        cypher_content = """
        CREATE (n:Person {name: 'Test'})
        RETURN n;
        """
        
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=1))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with patch('builtins.open', mock_open(read_data=cypher_content)):
            results = execute_cypher_file(neo4j_connection, "test.cypher")
        
        assert len(results) == 2  # Two statements
        assert mock_session.run.call_count == 2

    def test_execute_cypher_file_with_comments(self, neo4j_connection, mock_session, mock_result):
        """Test Cypher file execution with comments."""
        cypher_content = """
        // This is a comment
        CREATE (n:Person {name: 'Test'})
        /* Multi-line
           comment */
        RETURN n;
        """
        
        mock_result.consume.return_value = Mock()
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with patch('builtins.open', mock_open(read_data=cypher_content)):
            results = execute_cypher_file(neo4j_connection, "test.cypher")
        
        assert len(results) == 2  # Comments should be filtered out

    def test_execute_cypher_file_not_found(self, neo4j_connection):
        """Test Cypher file execution with file not found."""
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                execute_cypher_file(neo4j_connection, "nonexistent.cypher")

    def test_execute_cypher_file_empty_file(self, neo4j_connection, mock_session):
        """Test Cypher file execution with empty file."""
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with patch('builtins.open', mock_open(read_data="")):
            results = execute_cypher_file(neo4j_connection, "empty.cypher")
        
        assert results == []
        mock_session.run.assert_not_called()

    def test_execute_cypher_file_syntax_error(self, neo4j_connection, mock_session):
        """Test Cypher file execution with syntax error."""
        cypher_content = "INVALID CYPHER SYNTAX;"
        
        mock_session.run.side_effect = ClientError("Syntax error")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with patch('builtins.open', mock_open(read_data=cypher_content)):
            with pytest.raises(ClientError):
                execute_cypher_file(neo4j_connection, "invalid.cypher")

    def test_execute_cypher_file_with_parameters(self, neo4j_connection, mock_session, mock_result):
        """Test Cypher file execution with parameters."""
        cypher_content = "CREATE (n:Person {name: $name}) RETURN n;"
        
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=1))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        parameters = {"name": "John Doe"}
        
        with patch('builtins.open', mock_open(read_data=cypher_content)):
            results = execute_cypher_file(neo4j_connection, "test.cypher", parameters)
        
        mock_session.run.assert_called_once_with("CREATE (n:Person {name: $name}) RETURN n", parameters)

    def test_execute_cypher_file_transaction_handling(self, neo4j_connection, mock_session, mock_result):
        """Test Cypher file execution with transaction management."""
        cypher_content = """
        CREATE (n1:Person {name: 'Alice'});
        CREATE (n2:Person {name: 'Bob'});
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b);
        """
        
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=2, relationships_created=1))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with patch('builtins.open', mock_open(read_data=cypher_content)):
            results = execute_cypher_file(neo4j_connection, "transaction.cypher", use_transaction=True)
        
        assert len(results) == 3

    def test_execute_cypher_file_with_imports(self, neo4j_connection, mock_session, mock_result):
        """Test Cypher file execution with LOAD CSV or other imports."""
        cypher_content = """
        LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row
        CREATE (n:Person {name: row.name, age: toInteger(row.age)});
        """
        
        mock_result.consume.return_value = Mock(counters=Mock(nodes_created=100))
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with patch('builtins.open', mock_open(read_data=cypher_content)):
            results = execute_cypher_file(neo4j_connection, "import.cypher")
        
        assert len(results) == 1
        query_call = mock_session.run.call_args[0][0]
        assert "LOAD CSV" in query_call


class TestErrorHandlingAndEdgeCases:
    """Test suite for comprehensive error handling and edge cases."""
    
    def test_connection_lost_during_operation(self, neo4j_connection, mock_session):
        """Test handling connection loss during operation."""
        mock_session.run.side_effect = ServiceUnavailable("Connection lost")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ServiceUnavailable):
            execute_query(neo4j_connection, "MATCH (n) RETURN n")

    def test_large_dataset_handling(self, neo4j_connection, mock_session, mock_result):
        """Test handling large datasets."""
        # Simulate large result set
        large_dataset = [Mock(data={"n": {"id": f"node{i}"}}) for i in range(10000)]
        mock_result.list.return_value = large_dataset
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        nodes = find_nodes(neo4j_connection, "Person", {})
        
        assert len(nodes) == 10000

    def test_unicode_and_special_characters(self, neo4j_connection, mock_session, mock_result):
        """Test handling Unicode and special characters."""
        unicode_data = {
            "name": "José García",
            "emoji": "😀🎉",
            "chinese": "中文测试",
            "special": "\"'\\`${}[]",
            "cyrillic": "Привет мир",
            "arabic": "مرحبا بالعالم"
        }
        
        mock_record = Mock()
        mock_record.get.return_value = "unicode_node"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        node_id = create_node(neo4j_connection, "UnicodeTest", unicode_data)
        
        assert node_id == "unicode_node"

    def test_concurrent_access_simulation(self, neo4j_connection, mock_session):
        """Test simulation of concurrent access scenarios."""
        mock_session.run.side_effect = [
            TransientError("Deadlock detected"),
            Mock(spec=Result)  # Success on retry
        ]
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        # Should succeed after retry
        result = execute_query(neo4j_connection, "MATCH (n) RETURN n", retries=1)
        assert result is not None
        assert mock_session.run.call_count == 2

    def test_memory_management_large_properties(self, neo4j_connection, mock_session, mock_result):
        """Test handling nodes with large property values."""
        large_text = "x" * 1000000  # 1MB string
        large_data = {
            "large_text": large_text,
            "large_list": list(range(10000)),
            "nested_data": {"level1": {"level2": {"level3": "deep"}}},
            "binary_data": b"binary content here" * 1000
        }
        
        mock_record = Mock()
        mock_record.get.return_value = "large_node"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        node_id = create_node(neo4j_connection, "LargeNode", large_data)
        
        assert node_id == "large_node"

    @patch('adaptive_graph_of_thoughts.infrastructure.neo4j_utils.logging')
    def test_logging_integration(self, mock_logging, neo4j_connection, mock_session):
        """Test logging integration during operations."""
        mock_session.run.side_effect = ClientError("Test error for logging")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ClientError):
            execute_query(neo4j_connection, "INVALID QUERY")

    def test_json_serialization_edge_cases(self, neo4j_connection, mock_session, mock_result):
        """Test JSON serialization edge cases."""
        edge_case_data = {
            "datetime": datetime.now(),
            "none_value": None,
            "boolean": True,
            "float": 3.14159,
            "empty_string": "",
            "zero": 0,
            "negative": -42,
            "scientific_notation": 1.23e-4,
            "infinity": float('inf') if hasattr(float, 'inf') else 999999999,
        }
        
        mock_record = Mock()
        mock_record.get.return_value = "json_node"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        # This should handle type conversion appropriately
        node_id = create_node(neo4j_connection, "JsonTest", edge_case_data)
        
        assert node_id == "json_node"

    def test_malformed_query_handling(self, neo4j_connection, mock_session):
        """Test handling of malformed queries."""
        malformed_queries = [
            "MATCH (n RETURN n",  # Missing closing parenthesis
            "CREATE (n:Person {name: 'John', age: }",  # Missing value
            "MATCH (n) WHERE n.age > RETURN n",  # Missing comparison value
            "",  # Empty query
            "   ",  # Whitespace only
            None,  # None query
        ]
        
        mock_session.run.side_effect = ClientError("Syntax error")
        neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
        
        for query in malformed_queries:
            with pytest.raises((ClientError, ValueError, TypeError)):
                execute_query(neo4j_connection, query)

    def test_network_timeout_scenarios(self, neo4j_connection, mock_session):
        """Test various network timeout scenarios."""
        timeout_errors = [
            TransientError("Read timeout"),
            TransientError("Write timeout"),
            TransientError("Connection timeout"),
            ServiceUnavailable("Network unreachable")
        ]
        
        for error in timeout_errors:
            mock_session.run.side_effect = error
            neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
            
            with pytest.raises((TransientError, ServiceUnavailable)):
                execute_query(neo4j_connection, "MATCH (n) RETURN n")

    def test_constraint_violation_scenarios(self, neo4j_connection, mock_session):
        """Test various constraint violation scenarios."""
        constraint_errors = [
            ClientError("Node(0) already exists with label `Person` and property `email` = 'test@example.com'"),
            ClientError("Cannot delete node<0>, because it still has relationships."),
            ClientError("Property key does not exist: invalidProperty"),
            ClientError("Type mismatch: expected Integer but was String")
        ]
        
        for error in constraint_errors:
            mock_session.run.side_effect = error
            neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
            
            with pytest.raises(ClientError):
                execute_query(neo4j_connection, "CREATE (n:Person {email: 'test@example.com'})")

    def test_resource_exhaustion_scenarios(self, neo4j_connection, mock_session):
        """Test resource exhaustion scenarios."""
        resource_errors = [
            TransientError("There is not enough memory to perform the current task"),
            TransientError("Transaction timeout"),
            ClientError("Too many connections"),
            ServiceUnavailable("Database is temporarily unavailable")
        ]
        
        for error in resource_errors:
            mock_session.run.side_effect = error
            neo4j_connection._driver.session.return_value.__enter__.return_value = mock_session
            
            with pytest.raises((TransientError, ClientError, ServiceUnavailable)):
                execute_query(neo4j_connection, "MATCH (n) RETURN n")
