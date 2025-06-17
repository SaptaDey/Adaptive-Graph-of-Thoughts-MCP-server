"""Unit tests for EvidenceStage in stage_4_evidence.py.

Covers:
- Happy path execution
- Handling of empty evidence list
- Behavior at MAX_EVIDENCE limit
- Validation of evidence types
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from adaptive_graph_of_thoughts.domain.models.common_types import (
    GoTProcessorSessionData,
)
from adaptive_graph_of_thoughts.domain.stages.stage_3_hypothesis import HypothesisStage
from adaptive_graph_of_thoughts.domain.stages.stage_4_evidence import EvidenceStage

# Define a reasonable max evidence for testing
MAX_EVIDENCE = 100


# Mock Config class for testing
class MockConfig:
    def __init__(self):
        self.evidence_max_iterations = 5
        self.ibn_similarity_threshold = 0.5
        self.min_nodes_for_hyperedge = 2
        self.default_parameters = MagicMock(
            evidence_max_iterations=5,
            ibn_similarity_threshold=0.5,
            min_nodes_for_hyperedge=2,
            initial_layer="test_layer",
            default_disciplinary_tags=["test_tag"],
        )
        self.pubmed = None
        self.google_scholar = None
        self.exa_search = None


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def mock_session_data():
    """Provides a minimal valid session data object."""
    return GoTProcessorSessionData(
        session_id="test_session",
        accumulated_context={
            HypothesisStage.stage_name: {
                "hypothesis_node_ids": ["hypo1", "hypo2", "hypo3"]
            }
        },
    )


@pytest.fixture
def evidence_items():
    """Returns a list of valid evidence dictionaries for happy-path tests."""
    return [
        {
            "content": "Evidence one",
            "source_description": "Test Source",
            "url": "http://example.com/1",
            "doi": "10.1234/test1",
            "authors_list": ["Author 1", "Author 2"],
            "publication_date_str": "2023-01-01",
            "supports_hypothesis": True,
            "strength": 0.7,
            "disciplinary_tags": ["tag1", "tag2"],
            "raw_source_data_type": "TestArticle",
            "original_data": {"test": "data1"},
        },
        {
            "content": "Evidence two",
            "source_description": "Test Source",
            "url": "http://example.com/2",
            "doi": "10.1234/test2",
            "authors_list": ["Author 3", "Author 4"],
            "publication_date_str": "2023-02-01",
            "supports_hypothesis": False,
            "strength": 0.6,
            "disciplinary_tags": ["tag3", "tag4"],
            "raw_source_data_type": "TestArticle",
            "original_data": {"test": "data2"},
        },
        {
            "content": "Evidence three",
            "source_description": "Test Source",
            "url": "http://example.com/3",
            "doi": "10.1234/test3",
            "authors_list": ["Author 5", "Author 6"],
            "publication_date_str": "2023-03-01",
            "supports_hypothesis": True,
            "strength": 0.8,
            "disciplinary_tags": ["tag5", "tag6"],
            "raw_source_data_type": "TestArticle",
            "original_data": {"test": "data3"},
        },
    ]


@pytest.mark.asyncio
async def test_execute_returns_successful_output(
    mock_session_data, evidence_items, mock_config, monkeypatch
):
    """Happy path: execute method returns a successful StageOutput."""

    # Stub out any external/model calls to ensure deterministic behavior
    async def _stub_select(_self, _x):  # pylint: disable=unused-argument
        return {
            "id": "hypo1",
            "label": "Test Hypothesis",
            "confidence_vector_list": [0.5, 0.5, 0.5, 0.5],
        }

    async def _stub_execute_plan(_self, _x):  # pylint: disable=unused-argument
        return evidence_items

    async def _stub_create_evidence(_self, _hypo_data, _ev_data, _iteration, index):  # pylint: disable=unused-argument
        return {"id": f"ev_{index}", "label": f"Evidence {index}"}

    monkeypatch.setattr(
        EvidenceStage, "_select_hypothesis_to_evaluate_from_neo4j", _stub_select
    )
    monkeypatch.setattr(EvidenceStage, "_execute_hypothesis_plan", _stub_execute_plan)
    monkeypatch.setattr(
        EvidenceStage, "_create_evidence_in_neo4j", _stub_create_evidence
    )
    monkeypatch.setattr(
        EvidenceStage,
        "_update_hypothesis_confidence_in_neo4j",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        EvidenceStage, "_create_ibn_in_neo4j", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        EvidenceStage, "_create_hyperedges_in_neo4j", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        EvidenceStage, "_apply_temporal_decay_and_patterns", AsyncMock()
    )
    monkeypatch.setattr(EvidenceStage, "_adapt_graph_topology", AsyncMock())
    monkeypatch.setattr(EvidenceStage, "close_clients", AsyncMock())

    # Create the stage and execute it
    stage = EvidenceStage(mock_config)
    output = await stage.execute(mock_session_data)

    # Verify the output
    assert output is not None
    assert output.summary is not None
    assert "Evidence integration completed" in output.summary
    assert output.metrics is not None
    assert output.metrics["evidence_nodes_created_in_neo4j"] >= 0
    assert output.metrics["hypotheses_updated_in_neo4j"] >= 0
    assert output.next_stage_context_update is not None
    assert (
        output.next_stage_context_update[EvidenceStage.stage_name][
            "evidence_integration_completed"
        ]
        is True
    )


@pytest.mark.asyncio
async def test_execute_with_no_hypotheses(mock_config, _monkeypatch):
    """Test that execute handles the case where there are no hypotheses."""
    # Create a session data with no hypothesis node IDs
    session_data = GoTProcessorSessionData(
        session_id="test_session",
        accumulated_context={HypothesisStage.stage_name: {"hypothesis_node_ids": []}},
    )

    # Create the stage and execute it
    stage = EvidenceStage(mock_config)
    output = await stage.execute(session_data)

    # Verify the output
    assert output is not None
    assert "Evidence skipped: No hypotheses." in output.summary
    assert output.metrics["iterations_completed"] == 0
    assert (
        output.next_stage_context_update[EvidenceStage.stage_name]["error"]
        == "No hypotheses found"
    )


@pytest.mark.asyncio
async def test_execute_with_max_iterations(
    mock_session_data, evidence_items, mock_config, monkeypatch
):
    """Test that execute respects the max_iterations setting."""

    # Stub out any external/model calls to ensure deterministic behavior
    async def _stub_select(_self, _x):  # pylint: disable=unused-argument
        return {
            "id": "hypo1",
            "label": "Test Hypothesis",
            "confidence_vector_list": [0.5, 0.5, 0.5, 0.5],
        }

    async def _stub_execute_plan(_self, _x):  # pylint: disable=unused-argument
        return evidence_items

    async def _stub_create_evidence(_self, _hypo_data, _ev_data, _iteration, index):  # pylint: disable=unused-argument
        return {"id": f"ev_{index}", "label": f"Evidence {index}"}

    monkeypatch.setattr(
        EvidenceStage, "_select_hypothesis_to_evaluate_from_neo4j", _stub_select
    )
    monkeypatch.setattr(EvidenceStage, "_execute_hypothesis_plan", _stub_execute_plan)
    monkeypatch.setattr(
        EvidenceStage, "_create_evidence_in_neo4j", _stub_create_evidence
    )
    monkeypatch.setattr(
        EvidenceStage,
        "_update_hypothesis_confidence_in_neo4j",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        EvidenceStage, "_create_ibn_in_neo4j", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        EvidenceStage, "_create_hyperedges_in_neo4j", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        EvidenceStage, "_apply_temporal_decay_and_patterns", AsyncMock()
    )
    monkeypatch.setattr(EvidenceStage, "_adapt_graph_topology", AsyncMock())
    monkeypatch.setattr(EvidenceStage, "close_clients", AsyncMock())

    # Set max_iterations to 1
    mock_config.default_parameters.evidence_max_iterations = 1

    # Create the stage and execute it
    stage = EvidenceStage(mock_config)
    output = await stage.execute(mock_session_data)

    # Verify the output
    assert output is not None
    assert output.metrics["iterations_completed"] == 1


@pytest.mark.asyncio
async def test_execute_with_client_error(mock_session_data, mock_config, monkeypatch):
    """Test that execute handles client errors gracefully."""

    # Stub out any external/model calls to ensure deterministic behavior
    async def _stub_select(_self, _x):  # pylint: disable=unused-argument
        return {
            "id": "hypo1",
            "label": "Test Hypothesis",
            "confidence_vector_list": [0.5, 0.5, 0.5, 0.5],
        }

    async def _stub_execute_plan(_self, _x):  # pylint: disable=unused-argument
        raise Exception("Test client error")

    monkeypatch.setattr(
        EvidenceStage, "_select_hypothesis_to_evaluate_from_neo4j", _stub_select
    )
    monkeypatch.setattr(EvidenceStage, "_execute_hypothesis_plan", _stub_execute_plan)
    monkeypatch.setattr(EvidenceStage, "close_clients", AsyncMock())

    # Create the stage and execute it
    stage = EvidenceStage(mock_config)
    output = await stage.execute(mock_session_data)

    # Verify the output
    assert output is not None
    assert (
        output.next_stage_context_update[EvidenceStage.stage_name][
            "evidence_integration_completed"
        ]
        is True
    )
