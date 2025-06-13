"""Unit tests for Stage4Evidence in stage_4_evidence.py.

Covers:
- Happy path execution
- Handling of empty evidence list
- Behavior at MAX_EVIDENCE limit
- Validation of evidence types
"""
import pytest
from adaptive_graph_of_thoughts.domain.stages.stage_4_evidence import Stage4Evidence, MAX_EVIDENCE

class MinimalState:
    """Minimal stub for conversation/pipeline state required by Stage4Evidence."""
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

def test_run_returns_successful_state(minimal_state, evidence_items, monkeypatch):
    """Happy path: state updated with provided evidence and success flag True."""
    # Stub out any external/model calls to ensure deterministic behavior
    monkeypatch.setattr(Stage4Evidence, "_fetch_additional_info", lambda self, x: {})
    stage = Stage4Evidence()
    updated_state, success = stage.run(minimal_state, evidence_items, config={})
    assert success is True
    assert updated_state is not None
    assert updated_state.step == 4
    assert hasattr(updated_state, "evidence")
    assert updated_state.evidence == evidence_items

def test_run_with_empty_evidence_list(minimal_state):
    """Passing empty evidence should raise ValueError."""
    stage = Stage4Evidence()
    with pytest.raises(ValueError):
        stage.run(minimal_state, [], config={})

def test_run_with_max_evidence(minimal_state):
    """Ensure that providing more than MAX_EVIDENCE items is truncated correctly."""
    stage = Stage4Evidence()
    items = [f"evidence {i}" for i in range(MAX_EVIDENCE + 2)]
    updated_state, success = stage.run(minimal_state, items, config={})
    assert success is True
    assert len(updated_state.evidence) == MAX_EVIDENCE
    assert updated_state.evidence == items[:MAX_EVIDENCE]

def test_run_with_invalid_evidence_type(minimal_state):
    """Passing non-string evidence items should raise TypeError."""
    stage = Stage4Evidence()
    invalid_items = [1, 2, 3]
    with pytest.raises(TypeError):
        stage.run(minimal_state, invalid_items, config={})
def test_run_with_duplicate_evidence(minimal_state, monkeypatch):
    """Duplicate evidence strings should be preserved (business rule: duplicates allowed)."""
    monkeypatch.setattr(Stage4Evidence, "_fetch_additional_info", lambda self, x: {})
    stage = Stage4Evidence()
    items = ["e1", "e2", "e1"]  # duplicate 'e1'
    updated_state, success = stage.run(minimal_state, items, config={})
    assert success is True
    assert updated_state.evidence == items  # duplicates kept

def test_run_with_none_evidence(minimal_state):
    """Passing None instead of a list should raise TypeError."""
    stage = Stage4Evidence()
    with pytest.raises(TypeError):
        stage.run(minimal_state, None, config={})

def test_run_with_wrong_step(monkeypatch):
    """Stage4Evidence expects state.step == 4; otherwise ValueError is raised."""
    wrong_state = MinimalState()
    wrong_state.step = 3
    stage = Stage4Evidence()
    with pytest.raises(ValueError):
        stage.run(wrong_state, ["e1"], config={})

def test_fetch_additional_info_merges_into_state(minimal_state, monkeypatch):
    """_fetch_additional_info return dict should be merged into state.data."""
    extra = {"foo": "bar", "num": 42}
    monkeypatch.setattr(Stage4Evidence, "_fetch_additional_info", lambda self, x: extra)
    stage = Stage4Evidence()
    updated_state, success = stage.run(minimal_state, ["e1"], config={})
    assert success is True
    for k, v in extra.items():
        assert updated_state.data[k] == v

def test_evidence_whitespace_trim(minimal_state, monkeypatch):
    """Evidence strings should be trimmed of leading/trailing whitespace."""
    monkeypatch.setattr(Stage4Evidence, "_fetch_additional_info", lambda self, x: {})
    stage = Stage4Evidence()
    items = ["  ev1  ", "\tev2\n"]
    updated_state, _ = stage.run(minimal_state, items, config={})
    assert updated_state.evidence == ["ev1", "ev2"]

def test_run_idempotent(minimal_state, monkeypatch):
    """Running Stage4Evidence twice with same evidence should yield identical state."""
    monkeypatch.setattr(Stage4Evidence, "_fetch_additional_info", lambda self, x: {})
    stage = Stage4Evidence()
    items = ["ev1"]
    state1, _ = stage.run(minimal_state, items, config={})
    state2, _ = stage.run(state1, items, config={})
    assert state1.evidence == state2.evidence
    assert state1.data == state2.data