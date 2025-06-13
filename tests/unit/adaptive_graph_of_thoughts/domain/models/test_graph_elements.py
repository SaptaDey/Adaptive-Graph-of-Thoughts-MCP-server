import sys
import pytest
import uuid
from datetime import datetime
from hypothesis import given, strategies as st

from adaptive_graph_of_thoughts.domain.models.graph_elements import GraphElement

# Hypothesis strategies for GraphElement parameters
valid_uuid = st.uuids()
short_label = st.text(min_size=0, max_size=50)
long_label = st.text(min_size=51, max_size=500)
valid_weight = st.floats(allow_nan=False, allow_infinity=False)
edge_weights = st.one_of(
    st.just(0.0),
    st.just(sys.float_info.epsilon),
    st.just(-sys.float_info.epsilon),
    st.just(sys.float_info.max),
    st.just(float("inf")),
    st.just(float("-inf")),
    valid_weight
)

@pytest.fixture
def canonical_elem():
    """
    Pytest fixture that returns a canonical GraphElement instance with fixed attributes.
    
    Returns:
        GraphElement: An instance with node_id set to UUID(int=0), label "canonical", and weight 1.0.
    """
    return GraphElement(node_id=uuid.UUID(int=0), label="canonical", weight=1.0)

class TestGraphElementConstruction:
    """Tests for GraphElement initialization with valid inputs."""

    @given(node_id=valid_uuid, label=short_label, weight=valid_weight)
    def test_happy_path(self, node_id, label, weight):
        """
        Tests that a GraphElement is correctly initialized with valid node_id, label, and weight.
        
        Asserts that the constructed object's attributes match the provided values.
        """
        elem = GraphElement(node_id=node_id, label=label, weight=weight)
        assert elem.node_id == node_id
        assert elem.label == label
        assert elem.weight == weight

class TestGraphElementEdgeCases:
    """Edge-case tests for GraphElement parameters."""

    @pytest.mark.parametrize("label", ["", "a" * 1000, "测试中文标签"])
    @pytest.mark.parametrize("weight", [0.0, sys.float_info.max, sys.float_info.epsilon, -sys.float_info.epsilon])
    def test_edge_values(self, label, weight):
        """
        Tests that GraphElement correctly handles edge-case values for label and weight.
        
        Verifies that a GraphElement instance can be created with the provided label and weight,
        and that its attributes match the input values.
        """
        elem = GraphElement(node_id=uuid.uuid4(), label=label, weight=weight)
        assert elem.label == label
        assert elem.weight == weight

class TestGraphElementInvalidInputs:
    """Failure condition tests for invalid GraphElement inputs."""

    @pytest.mark.parametrize(
        "node_id, label, weight",
        [
            (None, "label", 1.0),
            ("not-uuid", "label", 1.0),
            (uuid.uuid4(), None, 1.0),
            (uuid.uuid4(), "label", None),
            (uuid.uuid4(), "a" * 1001, 1.0),
            (uuid.uuid4(), "label", "not-a-float"),
            (uuid.uuid4(), datetime.now(), 1.0),
            (uuid.uuid4(), "label", datetime.now()),
        ],
    )
    def test_invalid_inputs(self, node_id, label, weight):
        """
        Tests that constructing a GraphElement with invalid inputs raises ValueError or TypeError.
        
        Verifies that improper node_id, label, or weight values result in the appropriate exception.
        """
        with pytest.raises((ValueError, TypeError)):
            GraphElement(node_id=node_id, label=label, weight=weight)

class TestGraphElementEqualityHashing:
    """Tests for equality and hashing behavior of GraphElement."""

    def test_equality_and_hash(self):
        """
        Tests that two GraphElement instances with identical attributes are equal and have the same hash value.
        """
        uid = uuid.uuid4()
        elem1 = GraphElement(node_id=uid, label="test", weight=1.23)
        elem2 = GraphElement(node_id=uid, label="test", weight=1.23)
        assert elem1 == elem2
        assert hash(elem1) == hash(elem2)

    def test_inequality(self):
        """
        Tests that two GraphElement instances with the same node_id but different labels are not equal.
        """
        elem1 = GraphElement(node_id=uuid.uuid4(), label="a", weight=1.0)
        elem2 = GraphElement(node_id=elem1.node_id, label="b", weight=1.0)
        assert elem1 != elem2

class TestGraphElementRepresentation:
    """Tests for __repr__ and __str__ methods of GraphElement."""

    def test_repr_contains_class_and_id(self):
        """
        Tests that the repr() output of a GraphElement includes the class name and its UUID.
        """
        uid = uuid.uuid4()
        elem = GraphElement(node_id=uid, label="repr", weight=0.0)
        rep = repr(elem)
        assert "GraphElement" in rep
        assert str(uid) in rep

    def test_str_contains_label(self):
        """
        Tests that the string representation of a GraphElement includes its label.
        """
        elem = GraphElement(node_id=uuid.uuid4(), label="labelled", weight=2.0)
        assert "labelled" in str(elem)

class TestGraphElementSerialization:
    """Tests for serialization round-trip if supported."""

    @pytest.mark.skipif(
        not (hasattr(GraphElement, "to_dict") and hasattr(GraphElement, "from_dict")),
        reason="to_dict/from_dict not implemented",
    )
    def test_dict_round_trip(self):
        """
        Tests that serializing a GraphElement to a dictionary and deserializing it returns an equal object.
        """
        elem = GraphElement(node_id=uuid.uuid4(), label="round", weight=2.34)
        assert GraphElement.from_dict(elem.to_dict()) == elem

    @pytest.mark.skipif(
        not (hasattr(GraphElement, "to_json") and hasattr(GraphElement, "from_json")),
        reason="to_json/from_json not implemented",
    )
    def test_json_round_trip(self):
        """
        Tests that serializing and deserializing a GraphElement to and from JSON preserves equality.
        """
        elem = GraphElement(node_id=uuid.uuid4(), label="json", weight=3.45)
        assert GraphElement.from_json(elem.to_json()) == elem

class TestGraphElementFixture:
    """Tests using the canonical fixture GraphElement instance."""

    def test_canonical_attributes(self, canonical_elem):
        """
        Validates that the canonical GraphElement fixture has expected default attributes.
        
        Asserts that the node_id is a UUID, the label is "canonical", and the weight is 1.0.
        """
        assert isinstance(canonical_elem.node_id, uuid.UUID)
        assert canonical_elem.label == "canonical"
        assert canonical_elem.weight == 1.0