import sys
import pytest
import uuid
import math
import threading
import time
import gc
import unicodedata
from datetime import datetime
from hypothesis import given, strategies as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from adaptive_graph_of_thoughts.domain.models.graph_elements import GraphElement

# Enhanced Hypothesis strategies for GraphElement parameters
valid_uuid = st.uuids()
short_label = st.text(min_size=0, max_size=50)
medium_label = st.text(min_size=51, max_size=1000)
unicode_label = st.text(alphabet=st.characters(min_codepoint=0x0100, max_codepoint=0x017F))
emoji_label = st.text(alphabet="🌟✨🎉💫⭐🌈🎈🎊🎁🎀", min_size=1, max_size=10)
mixed_label = st.one_of(short_label, medium_label, unicode_label, emoji_label)

valid_weight = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
special_weights = st.one_of(
    st.just(0.0),
    st.just(-0.0),
    st.just(sys.float_info.epsilon),
    st.just(-sys.float_info.epsilon),
    st.just(sys.float_info.min),
    st.just(sys.float_info.max),
    st.just(1e-308),
    st.just(1e308),
)
edge_weights = st.one_of(
    st.just(0.0),
    st.just(sys.float_info.epsilon),
    st.just(-sys.float_info.epsilon),
    st.just(sys.float_info.max),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.just(float("nan")),
    valid_weight,
    special_weights
)

@pytest.fixture
def canonical_elem():
    """Provides a canonical GraphElement instance for reuse."""
    return GraphElement(node_id=uuid.UUID(int=0), label="canonical", weight=1.0)

class TestGraphElementConstruction:
    """Tests for GraphElement initialization with valid inputs."""

    @given(node_id=valid_uuid, label=short_label, weight=valid_weight)
    def test_happy_path(self, node_id, label, weight):
        elem = GraphElement(node_id=node_id, label=label, weight=weight)
        assert elem.node_id == node_id
        assert elem.label == label
        assert elem.weight == weight

class TestGraphElementEdgeCases:
    """Edge-case tests for GraphElement parameters."""

    @pytest.mark.parametrize("label", ["", "a" * 1000, "测试中文标签"])
    @pytest.mark.parametrize("weight", [0.0, sys.float_info.max, sys.float_info.epsilon, -sys.float_info.epsilon])
    def test_edge_values(self, label, weight):
        elem = GraphElement(node_id=uuid.uuid4(), label=label, weight=weight)
        assert elem.label == label
        assert elem.weight == weight

class TestGraphElementInvalidInputs:
    """Failure condition tests for invalid GraphElement inputs."""

    @pytest.mark.parametrize(
        "node_id, label, weight, expected_exception",
        [
            (None, "label", 1.0, TypeError),
            ("not-uuid", "label", 1.0, TypeError),
            (uuid.uuid4(), None, 1.0, TypeError),
            (uuid.uuid4(), "label", None, TypeError),
            (uuid.uuid4(), "a" * 1001, 1.0, ValueError),  # Exceeds 1000 char limit
            (uuid.uuid4(), "label", "not-a-float", TypeError),
            (uuid.uuid4(), 123, 1.0, TypeError),  # label must be string
            (uuid.uuid4(), "label", complex(1, 2), TypeError),  # weight must be number
            (uuid.uuid4(), [], 1.0, TypeError),  # label must be string
            (uuid.uuid4(), "label", [1.0], TypeError),  # weight must be number
        ],
    )
    def test_invalid_inputs(self, node_id, label, weight, expected_exception):
        with pytest.raises(expected_exception):
            GraphElement(node_id=node_id, label=label, weight=weight)

class TestGraphElementEqualityHashing:
    """Tests for equality and hashing behavior of GraphElement."""

    def test_equality_and_hash(self):
        uid = uuid.uuid4()
        elem1 = GraphElement(node_id=uid, label="test", weight=1.23)
        elem2 = GraphElement(node_id=uid, label="test", weight=1.23)
        assert elem1 == elem2
        assert hash(elem1) == hash(elem2)

    def test_inequality(self):
        elem1 = GraphElement(node_id=uuid.uuid4(), label="a", weight=1.0)
        elem2 = GraphElement(node_id=elem1.node_id, label="b", weight=1.0)
        assert elem1 != elem2

class TestGraphElementRepresentation:
    """Tests for __repr__ and __str__ methods of GraphElement."""

    def test_repr_contains_class_and_id(self):
        uid = uuid.uuid4()
        elem = GraphElement(node_id=uid, label="repr", weight=0.0)
        rep = repr(elem)
        assert "GraphElement" in rep
        assert str(uid) in rep

    def test_str_contains_label(self):
        elem = GraphElement(node_id=uuid.uuid4(), label="labelled", weight=2.0)
        assert "labelled" in str(elem)

class TestGraphElementSerialization:
    """Tests for serialization round-trip if supported."""

    @pytest.mark.skipif(
        not (hasattr(GraphElement, "to_dict") and hasattr(GraphElement, "from_dict")),
        reason="to_dict/from_dict not implemented",
    )
    def test_dict_round_trip(self):
        elem = GraphElement(node_id=uuid.uuid4(), label="round", weight=2.34)
        assert GraphElement.from_dict(elem.to_dict()) == elem

    @pytest.mark.skipif(
        not (hasattr(GraphElement, "to_json") and hasattr(GraphElement, "from_json")),
        reason="to_json/from_json not implemented",
    )
    def test_json_round_trip(self):
        elem = GraphElement(node_id=uuid.uuid4(), label="json", weight=3.45)
        assert GraphElement.from_json(elem.to_json()) == elem

class TestGraphElementFixture:
    """Tests using the canonical fixture GraphElement instance."""

    def test_canonical_attributes(self, canonical_elem):
        assert isinstance(canonical_elem.node_id, uuid.UUID)
        assert canonical_elem.label == "canonical"
        assert canonical_elem.weight == 1.0

class TestGraphElementThreadSafety:
    """Tests for thread safety and concurrent access to GraphElement."""
    
    def test_concurrent_creation(self):
        """Test that multiple threads can create GraphElements concurrently."""
        def create_element(index):
            return GraphElement(
                node_id=uuid.uuid4(), 
                label=f"thread_{index}", 
                weight=float(index)
            )
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_element, i) for i in range(100)]
            elements = [future.result() for future in as_completed(futures)]
        
        assert len(elements) == 100
        assert len(set(elem.node_id for elem in elements)) == 100  # All unique IDs
        
    def test_concurrent_hash_computation(self):
        """Test that hash computation is thread-safe."""
        elem = GraphElement(node_id=uuid.uuid4(), label="concurrent", weight=1.0)
        hashes = []
        
        def compute_hash():
            hashes.append(hash(elem))
        
        threads = [threading.Thread(target=compute_hash) for _ in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(set(hashes)) == 1  # All hashes should be identical
        assert all(h == hash(elem) for h in hashes)

    def test_concurrent_equality_checks(self):
        """Test concurrent equality comparisons."""
        elem1 = GraphElement(node_id=uuid.uuid4(), label="test", weight=1.0)
        elem2 = GraphElement(node_id=elem1.node_id, label="test", weight=1.0)
        results = []
        
        def check_equality():
            results.append(elem1 == elem2)
        
        threads = [threading.Thread(target=check_equality) for _ in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert all(results)  # All should be True
        assert len(results) == 50

class TestGraphElementImmutability:
    """Tests for immutability of GraphElement (frozen dataclass)."""
    
    def test_node_id_immutable(self):
        """Test that node_id cannot be modified after creation."""
        elem = GraphElement(node_id=uuid.uuid4(), label="test", weight=1.0)
        original_id = elem.node_id
        
        with pytest.raises(AttributeError):
            elem.node_id = uuid.uuid4()
        
        assert elem.node_id == original_id
        
    def test_label_immutable(self):
        """Test that label cannot be modified after creation."""
        elem = GraphElement(node_id=uuid.uuid4(), label="original", weight=1.0)
        
        with pytest.raises(AttributeError):
            elem.label = "modified"
        
        assert elem.label == "original"
        
    def test_weight_immutable(self):
        """Test that weight cannot be modified after creation."""
        elem = GraphElement(node_id=uuid.uuid4(), label="test", weight=1.0)
        
        with pytest.raises(AttributeError):
            elem.weight = 2.0
        
        assert elem.weight == 1.0
        
    def test_no_dynamic_attributes(self):
        """Test that new attributes cannot be added dynamically."""
        elem = GraphElement(node_id=uuid.uuid4(), label="test", weight=1.0)
        
        with pytest.raises(AttributeError):
            elem.new_attribute = "should fail"
            
    def test_frozen_dataclass_behavior(self):
        """Test that the dataclass is properly frozen."""
        elem = GraphElement(node_id=uuid.uuid4(), label="test", weight=1.0)
        
        # Should not be able to use __setattr__
        with pytest.raises(AttributeError):
            elem.__setattr__("node_id", uuid.uuid4())

class TestGraphElementFloatingPointPrecision:
    """Tests for floating-point precision and special floating-point values."""
    
    @pytest.mark.parametrize("weight", [
        float('inf'),
        float('-inf'),
        float('nan'),
        1e-308,  # Very small positive number
        1e308,   # Very large positive number
        -1e308,  # Very large negative number
        sys.float_info.min,
        sys.float_info.max,
        sys.float_info.epsilon,
        -sys.float_info.epsilon,
        0.0,
        -0.0,
    ])
    def test_special_float_values(self, weight):
        """Test creation with special floating-point values."""
        elem = GraphElement(node_id=uuid.uuid4(), label="special", weight=weight)
        
        if math.isnan(weight):
            assert math.isnan(elem.weight)
        elif math.isinf(weight):
            assert math.isinf(elem.weight)
            assert math.copysign(1.0, elem.weight) == math.copysign(1.0, weight)
        else:
            assert elem.weight == weight
            
    def test_floating_point_precision_comparison(self):
        """Test that floating-point precision affects equality as expected."""
        node_id = uuid.uuid4()
        weight1 = 0.1 + 0.2  # 0.30000000000000004
        weight2 = 0.3        # 0.3
        
        elem1 = GraphElement(node_id=node_id, label="test", weight=weight1)
        elem2 = GraphElement(node_id=node_id, label="test", weight=weight2)
        
        # Should be different due to floating-point precision
        assert elem1 != elem2
        assert hash(elem1) != hash(elem2)
        
    def test_zero_variations(self):
        """Test different representations of zero."""
        node_id = uuid.uuid4()
        
        elem1 = GraphElement(node_id=node_id, label="test", weight=0.0)
        elem2 = GraphElement(node_id=node_id, label="test", weight=-0.0)
        
        # Both should be considered equal (0.0 == -0.0 in Python)
        assert elem1 == elem2
        assert hash(elem1) == hash(elem2)
        
    def test_nan_equality_behavior(self):
        """Test NaN equality behavior (NaN != NaN)."""
        node_id = uuid.uuid4()
        
        elem1 = GraphElement(node_id=node_id, label="test", weight=float('nan'))
        elem2 = GraphElement(node_id=node_id, label="test", weight=float('nan'))
        
        # NaN != NaN, so elements should not be equal
        assert elem1 != elem2
        # Hash of NaN elements might differ

class TestGraphElementUnicodeAndEncoding:
    """Tests for Unicode handling and various character encodings."""
    
    @pytest.mark.parametrize("label", [
        "简体中文",
        "العربية", 
        "русский",
        "ひらがな",
        "한국어",
        "🌟✨🎉",  # Emoji
        "café",
        "naïve",
        "\u0001",  # Control character
        "\u001f",  # More control characters
        "a\nb\tc",  # Whitespace characters
        "  leading and trailing spaces  ",
        "",  # Empty string
        " ",  # Single space
        "\u00A0",  # Non-breaking space
        "αβγδε",  # Greek letters
        "∑∏∆√∞",  # Mathematical symbols
    ])
    def test_unicode_labels(self, label):
        """Test GraphElement creation with various Unicode labels."""
        elem = GraphElement(node_id=uuid.uuid4(), label=label, weight=1.0)
        assert elem.label == label
        assert isinstance(elem.label, str)
        
    def test_label_normalization(self):
        """Test that Unicode normalization doesn't affect labels."""
        # Composed vs decomposed characters
        label1 = "café"  # é as single character
        label2 = "cafe\u0301"  # e + combining acute accent
        
        elem1 = GraphElement(node_id=uuid.uuid4(), label=label1, weight=1.0)
        elem2 = GraphElement(node_id=uuid.uuid4(), label=label2, weight=1.0)
        
        # Should be different unless explicitly normalized
        assert elem1.label != elem2.label
        assert elem1 != elem2
        
    def test_very_long_unicode_label(self):
        """Test with long Unicode labels up to the limit."""
        long_label = "测试" * 500  # 1000 Unicode characters (at the limit)
        elem = GraphElement(node_id=uuid.uuid4(), label=long_label, weight=1.0)
        assert elem.label == long_label
        assert len(elem.label) == 1000
        
    def test_unicode_label_exceeds_limit(self):
        """Test that Unicode labels exceeding 1000 chars raise ValueError."""
        too_long_label = "测试" * 501  # 1002 Unicode characters
        with pytest.raises(ValueError, match="label cannot exceed 1000 characters"):
            GraphElement(node_id=uuid.uuid4(), label=too_long_label, weight=1.0)

    def test_mixed_unicode_ascii(self):
        """Test labels mixing ASCII and Unicode characters."""
        mixed_label = "Hello 世界! 🌍"
        elem = GraphElement(node_id=uuid.uuid4(), label=mixed_label, weight=1.0)
        assert elem.label == mixed_label

class TestGraphElementValidationAndEdgeCases:
    """Comprehensive validation and edge case tests."""
    
    def test_type_validation_comprehensive(self):
        """Comprehensive type validation for all parameters."""
        invalid_node_ids = [
            "string", 123, [], {}, set(), object(), lambda x: x, True, False
        ]
        
        for invalid_id in invalid_node_ids:
            with pytest.raises(TypeError, match="node_id must be a UUID instance"):
                GraphElement(node_id=invalid_id, label="test", weight=1.0)
                
        invalid_labels = [
            123, [], {}, set(), object(), lambda x: x, True, False, uuid.uuid4()
        ]
        
        for invalid_label in invalid_labels:
            with pytest.raises(TypeError, match="label must be a string"):
                GraphElement(node_id=uuid.uuid4(), label=invalid_label, weight=1.0)
                
        invalid_weights = [
            "string", [], {}, set(), object(), lambda x: x, complex(1, 2), True, False
        ]
        
        for invalid_weight in invalid_weights:
            with pytest.raises(TypeError, match="weight must be a number"):
                GraphElement(node_id=uuid.uuid4(), label="test", weight=invalid_weight)
                
    def test_boundary_label_lengths(self):
        """Test label lengths at boundaries."""
        # Test exactly at the limit
        limit_label = "x" * 1000
        elem = GraphElement(node_id=uuid.uuid4(), label=limit_label, weight=1.0)
        assert len(elem.label) == 1000
        
        # Test one character over the limit
        over_limit_label = "x" * 1001
        with pytest.raises(ValueError, match="label cannot exceed 1000 characters"):
            GraphElement(node_id=uuid.uuid4(), label=over_limit_label, weight=1.0)
            
    def test_uuid_boundary_values(self):
        """Test with UUID boundary values."""
        # Test with UUID containing all zeros
        zero_uuid = uuid.UUID(int=0)
        elem1 = GraphElement(node_id=zero_uuid, label="zero", weight=1.0)
        assert elem1.node_id == zero_uuid
        
        # Test with UUID containing all ones (maximum value)
        max_uuid = uuid.UUID(int=2**128 - 1)
        elem2 = GraphElement(node_id=max_uuid, label="max", weight=1.0)
        assert elem2.node_id == max_uuid
        
    def test_comparison_with_none_and_other_types(self):
        """Test comparison with None and different object types."""
        elem = GraphElement(node_id=uuid.uuid4(), label="test", weight=1.0)
        
        assert elem != None
        assert not (elem == None)
        assert elem != "string"
        assert elem != 123
        assert elem != []
        assert elem != {}
        assert elem != set()
        assert elem != uuid.uuid4()
        
    def test_hash_immutability(self):
        """Test that hash remains constant for the same object."""
        elem = GraphElement(node_id=uuid.uuid4(), label="test", weight=1.0)
        original_hash = hash(elem)
        
        # Hash should remain the same across multiple calls
        for _ in range(100):
            assert hash(elem) == original_hash
            
    def test_repr_and_str_comprehensive(self):
        """Comprehensive tests for string representations."""
        uid = uuid.uuid4()
        elem = GraphElement(node_id=uid, label="test_repr", weight=1.23)
        
        # Test __repr__
        repr_str = repr(elem)
        assert "GraphElement" in repr_str
        assert str(uid) in repr_str
        assert "test_repr" in repr_str
        assert "1.23" in repr_str
        
        # Test __str__ (dataclass uses __repr__ by default)
        str_str = str(elem)
        assert str_str == repr_str
        
    def test_int_weight_conversion(self):
        """Test that integer weights are accepted and work correctly."""
        elem = GraphElement(node_id=uuid.uuid4(), label="int_weight", weight=42)
        assert elem.weight == 42
        assert isinstance(elem.weight, int)  # Should preserve int type
        
        # Test equality with float equivalent
        elem_float = GraphElement(node_id=elem.node_id, label="int_weight", weight=42.0)
        assert elem == elem_float  # 42 == 42.0 in Python

class TestGraphElementPropertyBased:
    """Comprehensive property-based tests using Hypothesis."""
    
    @given(node_id=valid_uuid, label=mixed_label, weight=valid_weight)
    def test_property_creation_round_trip(self, node_id, label, weight):
        """Property-based test for creation and attribute access."""
        # Skip if label is too long
        if len(label) > 1000:
            return
            
        elem = GraphElement(node_id=node_id, label=label, weight=weight)
        assert elem.node_id == node_id
        assert elem.label == label
        assert elem.weight == weight
        
    @given(node_id=valid_uuid, label=short_label, weight=edge_weights)
    def test_property_hash_consistency(self, node_id, label, weight):
        """Property-based test for hash consistency."""
        # Skip NaN values as they don't hash consistently with themselves
        if math.isnan(weight):
            return
            
        elem1 = GraphElement(node_id=node_id, label=label, weight=weight)
        elem2 = GraphElement(node_id=node_id, label=label, weight=weight)
        
        assert elem1 == elem2
        assert hash(elem1) == hash(elem2)
        
    @given(
        node_id1=valid_uuid, 
        node_id2=valid_uuid,
        label1=short_label,
        label2=short_label,
        weight1=valid_weight,
        weight2=valid_weight
    )
    def test_property_inequality_conditions(self, node_id1, node_id2, label1, label2, weight1, weight2):
        """Property-based test for inequality conditions."""
        elem1 = GraphElement(node_id=node_id1, label=label1, weight=weight1)
        elem2 = GraphElement(node_id=node_id2, label=label2, weight=weight2)
        
        if node_id1 != node_id2 or label1 != label2 or weight1 != weight2:
            assert elem1 != elem2
        else:
            assert elem1 == elem2
            
    @given(node_id=valid_uuid, label=short_label, weight=valid_weight)
    def test_property_immutability(self, node_id, label, weight):
        """Property-based test for immutability."""
        elem = GraphElement(node_id=node_id, label=label, weight=weight)
        
        # All attempts to modify should fail
        with pytest.raises(AttributeError):
            elem.node_id = uuid.uuid4()
        with pytest.raises(AttributeError):
            elem.label = "modified"
        with pytest.raises(AttributeError):
            elem.weight = 999.0
            
        # Original values should remain unchanged
        assert elem.node_id == node_id
        assert elem.label == label
        assert elem.weight == weight

    @given(node_id=valid_uuid, label=st.text(max_size=1000), weight=valid_weight)
    def test_property_dataclass_behavior(self, node_id, label, weight):
        """Property-based test for dataclass behavior."""
        elem = GraphElement(node_id=node_id, label=label, weight=weight)
        
        # Test that all attributes are accessible
        assert hasattr(elem, 'node_id')
        assert hasattr(elem, 'label')
        assert hasattr(elem, 'weight')
        
        # Test that the element is hashable
        hash_val = hash(elem)
        assert isinstance(hash_val, int)
        
        # Test that it can be used in sets and as dict keys
        element_set = {elem}
        assert elem in element_set
        
        element_dict = {elem: "value"}
        assert element_dict[elem] == "value"

class TestGraphElementPerformance:
    """Performance and memory usage tests."""
    
    def test_creation_performance(self):
        """Test performance of GraphElement creation."""
        start_time = time.time()
        elements = []
        
        for i in range(10000):
            elements.append(GraphElement(
                node_id=uuid.uuid4(),
                label=f"perf_test_{i}",
                weight=float(i)
            ))
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 10000 elements in reasonable time (less than 2 seconds)
        assert creation_time < 2.0
        assert len(elements) == 10000
        
    def test_hash_performance(self):
        """Test performance of hash computation."""
        elem = GraphElement(node_id=uuid.uuid4(), label="hash_test", weight=1.0)
        
        start_time = time.time()
        hashes = [hash(elem) for _ in range(100000)]
        end_time = time.time()
        
        # Should compute 100000 hashes quickly
        assert end_time - start_time < 1.0
        assert len(set(hashes)) == 1  # All hashes should be identical
        
    def test_equality_performance(self):
        """Test performance of equality comparisons."""
        elem1 = GraphElement(node_id=uuid.uuid4(), label="perf_test", weight=1.0)
        elem2 = GraphElement(node_id=elem1.node_id, label="perf_test", weight=1.0)
        
        start_time = time.time()
        results = [elem1 == elem2 for _ in range(100000)]
        end_time = time.time()
        
        # Should perform 100000 equality checks quickly
        assert end_time - start_time < 1.0
        assert all(results)  # All should be True
        
    @pytest.mark.slow
    def test_memory_usage_estimation(self):
        """Estimate memory usage of GraphElements."""
        # Create a baseline measurement
        gc.collect()  # Clean up before measurement
        
        elements = []
        for i in range(1000):
            elements.append(GraphElement(
                node_id=uuid.uuid4(),
                label=f"memory_test_{i}",
                weight=float(i)
            ))
        
        # Basic check that we can create many elements without issues
        assert len(elements) == 1000
        assert all(isinstance(elem, GraphElement) for elem in elements)
        
        # Clean up
        del elements
        gc.collect()