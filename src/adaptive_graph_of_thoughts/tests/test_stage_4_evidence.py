# test_stage_4_evidence.py
# Pytest-based tests for stage_4_evidence module.
# Confirmed pytest is used in this repo via pytest.ini and pytest markers below.

import pytest

from adaptive_graph_of_thoughts.stage_4_evidence import (
    gather_evidence,
    score_evidence,
    select_best_evidence,
    Evidence,
)

# --- Fixtures --------------------------------------------------------------

@pytest.fixture
def sample_prompt():
    return "Test prompt for evidence gathering."

@pytest.fixture
def sample_thoughts():
    return ["Thought A", "Thought B"]

@pytest.fixture
def sample_evidence():
    return [
        Evidence(id="e1", text="Evidence 1", score=5.0),
        Evidence(id="e2", text="Evidence 2", score=3.0),
        Evidence(id="e3", text="Evidence 3", score=4.0),
    ]

# --- Fake LLM stub ---------------------------------------------------------

class FakeChoice:
    def __init__(self, content):
        self.message = {"content": content}
        self.finish_reason = "stop"

class FakeResponse:
    def __init__(self, contents):
        self.choices = [FakeChoice(c) for c in contents]

# --- Tests for gather_evidence ---------------------------------------------

def test_gather_evidence_happy_path(monkeypatch, sample_prompt, sample_thoughts):
    # Stub the OpenAI ChatCompletion.create to return deterministic fake responses
    fake_contents = ["1|First evidence text", "2|Second evidence text"]
    import adaptive_graph_of_thoughts.stage_4_evidence as s4
    monkeypatch.setattr(
        s4.openai.ChatCompletion,
        "create",
        lambda **kwargs: FakeResponse(fake_contents),
    )

    result = gather_evidence(sample_prompt, sample_thoughts)
    assert isinstance(result, list)
    assert len(result) == len(fake_contents)
    for ev, content in zip(result, fake_contents):
        expected_id, expected_text = content.split("|", 1)
        assert ev.id == expected_id
        assert ev.text == expected_text

def test_gather_evidence_missing_thoughts(sample_prompt):
    with pytest.raises(ValueError):
        gather_evidence(sample_prompt, [])

def test_gather_evidence_invalid_thoughts_type(sample_prompt):
    with pytest.raises(TypeError):
        gather_evidence(sample_prompt, "not a list")

def test_gather_evidence_deterministic_multiple_calls(monkeypatch, sample_prompt, sample_thoughts):
    fake_contents = ["A|Alpha", "B|Beta"]
    import adaptive_graph_of_thoughts.stage_4_evidence as s4
    monkeypatch.setattr(
        s4.openai.ChatCompletion,
        "create",
        lambda **kwargs: FakeResponse(fake_contents),
    )
    first = gather_evidence(sample_prompt, sample_thoughts)
    second = gather_evidence(sample_prompt, sample_thoughts)
    assert first == second

# --- Tests for score_evidence ---------------------------------------------

@pytest.mark.parametrize(
    "scores,expected_scores",
    [
        ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
        ([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]),
    ],
)
def test_score_evidence_order_and_ties(scores, expected_scores):
    evs = [Evidence(id=f"e{i+1}", text="", score=s) for i, s in enumerate(scores)]
    sorted_evs = score_evidence(evs)
    assert [ev.score for ev in sorted_evs] == expected_scores

def test_score_evidence_ties_preserve_order():
    evs = [Evidence(id="a", text="", score=1.0), Evidence(id="b", text="", score=1.0)]
    sorted_evs = score_evidence(evs)
    assert [ev.id for ev in sorted_evs] == ["a", "b"]

def test_score_evidence_invalid_input_type():
    with pytest.raises(TypeError):
        score_evidence("invalid input")

# --- Tests for select_best_evidence ----------------------------------------

def test_select_best_evidence_top_n(sample_evidence):
    best = select_best_evidence(sample_evidence, max_items=2)
    expected_top = sorted(sample_evidence, key=lambda e: e.score, reverse=True)[:2]
    assert [e.id for e in best] == [e.id for e in expected_top]

def test_select_best_evidence_empty_list():
    with pytest.raises(ValueError):
        select_best_evidence([], max_items=1)

def test_select_best_evidence_zero_max_items(sample_evidence):
    # Boundary: zero max_items should return empty list
    result = select_best_evidence(sample_evidence, max_items=0)
    assert result == []

# --- Property-based test for idempotency ----------------------------------

pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

@given(
    st.lists(
        st.builds(
            lambda id, text, score: Evidence(id=id, text=text, score=score),
            id=st.text(min_size=1),
            text=st.text(),
            score=st.floats(allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
    )
)
def test_select_best_evidence_idempotent(evidence_list):
    once = select_best_evidence(evidence_list, max_items=len(evidence_list))
    twice = select_best_evidence(once, max_items=len(once))
    assert once == twice
# --- Additional edge-case tests --------------------------------------------

def test_select_best_evidence_max_exceeds_list(sample_evidence):
    # If max_items larger than list, should return full sorted list without error
    bigger = select_best_evidence(sample_evidence, max_items=99)
    expected = sorted(sample_evidence, key=lambda e: e.score, reverse=True)
    assert bigger == expected

def test_select_best_evidence_negative_max_items(sample_evidence):
    # Negative max_items should raise a ValueError
    with pytest.raises(ValueError):
        select_best_evidence(sample_evidence, max_items=-1)

def test_score_evidence_handles_negative_scores():
    # Ensure descending sort even when scores include negatives and zero
    evs = [
        Evidence(id="neg1", text="neg", score=-1.0),
        Evidence(id="zero", text="zero", score=0.0),
        Evidence(id="pos", text="pos", score=1.0),
    ]
    sorted_evs = score_evidence(evs)
    assert [ev.score for ev in sorted_evs] == [1.0, 0.0, -1.0]

def test_gather_evidence_invalid_prompt_type(sample_thoughts):
    # Non-string prompt should raise a TypeError
    with pytest.raises(TypeError):
        gather_evidence(123, sample_thoughts)