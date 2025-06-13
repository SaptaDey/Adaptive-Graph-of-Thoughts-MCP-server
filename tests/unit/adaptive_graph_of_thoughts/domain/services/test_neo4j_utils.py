import pytest
from unittest.mock import patch, MagicMock
from neo4j.exceptions import Neo4jError

import adaptive_graph_of_thoughts.domain.services.neo4j_utils as neo4j_utils

@pytest.fixture
def mock_driver():
    driver = MagicMock(name="driver")
    session = MagicMock(name="session")
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    return driver

def test_connect_success(monkeypatch, mock_driver):
    with patch("neo4j.GraphDatabase.driver", return_value=mock_driver) as mock_graph_driver:
        driver = neo4j_utils.connect("bolt://localhost:7687", auth=("neo4j", "test"))
        assert driver is mock_driver
        mock_graph_driver.assert_called_once_with("bolt://localhost:7687", auth=("neo4j", "test"))

def test_connect_failure():
    with patch("neo4j.GraphDatabase.driver", side_effect=Neo4jError("boom")):
        with pytest.raises(Neo4jError):
            neo4j_utils.connect("bolt://badhost:7687", auth=("neo4j", "bad"))

def test_execute_query_success(monkeypatch, mock_driver):
    sample_records = [{"name": "Alice"}, {"name": "Bob"}]
    mock_result = MagicMock()
    mock_result.data.return_value = sample_records
    mock_driver.session.return_value.__enter__.return_value.run.return_value = mock_result

    monkeypatch.setattr(neo4j_utils, "_driver", mock_driver)
    data = neo4j_utils.execute_query("MATCH (n) RETURN n.name AS name")
    assert data == sample_records
    mock_driver.session.return_value.__enter__.return_value.run.assert_called_once()

def test_execute_query_error(monkeypatch, mock_driver):
    mock_driver.session.return_value.__enter__.return_value.run.side_effect = Neo4jError("query error")
    monkeypatch.setattr(neo4j_utils, "_driver", mock_driver)
    with pytest.raises(Neo4jError):
        neo4j_utils.execute_query("BAD CYPHER")

def test_with_transaction_context_manager(monkeypatch, mock_driver):
    monkeypatch.setattr(neo4j_utils, "_driver", mock_driver)
    with neo4j_utils.transaction() as session:
        session.run("RETURN 1")
    mock_driver.session.assert_called_once()

def test_format_result_empty():
    assert neo4j_utils._format_result([]) == []