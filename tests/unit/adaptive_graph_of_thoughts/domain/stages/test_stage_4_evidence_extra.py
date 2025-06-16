"""Unit tests for EvidenceStage in stage_4_evidence.py.

Covers:
- Happy path execution
- Handling of empty evidence list
- Behavior at MAX_EVIDENCE limit
- Validation of evidence types
"""
import pytest
from adaptive_graph_of_thoughts.domain.stages.stage_4_evidence import EvidenceStage

# Define a reasonable max evidence for testing
MAX_EVIDENCE = 100

# Mock Config class for testing
class MockConfig:
    def __init__(self):
        self.evidence_max_iterations = 5
        self.ibn_similarity_threshold = 0.5
        self.min_nodes_for_hyperedge = 2

@pytest.fixture
def mock_config():
    return MockConfig()

class MinimalState:
    """Minimal stub for conversation/pipeline state required by EvidenceStage."""
    def __init__(self):
        self.step = 0
        self.evidence = []
        self.data = {}

@pytest.fixture
def minimal_state():
    """Provides a minimal valid state object."""
    state = MinimalState()
    state.step = 4
    return state

@pytest.fixture
def evidence_items():
    """Returns a list of valid evidence strings for happy-path tests."""
    return ["evidence one", "evidence two", "evidence three"]

def test_run_returns_successful_state(minimal_state, evidence_items, mock_config, monkeypatch):
    """Happy path: state updated with provided evidence and success flag True."""
    # Stub out any external/model calls to ensure deterministic behavior
    monkeypatch.setattr(EvidenceStage, "_fetch_additional_info", lambda self, x: {})
    stage = EvidenceStage(mock_config)
    updated_state, success = stage.run(minimal_state, evidence_items, config={})
    assert success is True
    assert updated_state is not None
    assert updated_state.step == 4
    assert hasattr(updated_state, "evidence")
    assert updated_state.evidence == evidence_items

def test_run_with_empty_evidence_list(minimal_state, mock_config):
    """Passing empty evidence should raise ValueError."""
    stage = EvidenceStage(mock_config)
    with pytest.raises(ValueError):
        stage.run(minimal_state, [], config={})

def test_run_with_max_evidence(minimal_state, mock_config):
    """Ensure that providing more than MAX_EVIDENCE items is truncated correctly."""
    stage = EvidenceStage(mock_config)
    items = [f"evidence {i}" for i in range(MAX_EVIDENCE + 2)]
    updated_state, success = stage.run(minimal_state, items, config={})
    assert success is True
    assert len(updated_state.evidence) == MAX_EVIDENCE
    assert updated_state.evidence == items[:MAX_EVIDENCE]

def test_run_with_invalid_evidence_type(minimal_state, mock_config):
    """Passing non-string evidence items should raise TypeError."""
    stage = EvidenceStage(mock_config)
    invalid_items = [1, 2, 3]
    with pytest.raises(TypeError):
        stage.run(minimal_state, invalid_items, config={})