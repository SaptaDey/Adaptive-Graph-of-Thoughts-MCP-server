# NOTE: This test module uses pytest and pytest-asyncio fixtures as the primary test framework.

import datetime
import pytest

from adaptive_graph_of_thoughts.domain.services.neo4j_utils import (
    sanitize_property_key,
    serialize_value,
    build_cypher_match_clause,
    build_cypher_return_clause,
)

# sanitize_property_key tests
@pytest.mark.parametrize(
    "key,expected",
    [
        ("simple key", "`simple key`"),
        ("name.surname", "`name.surname`"),
        ("hyphen-key", "`hyphen-key`"),
    ],
)
def test_sanitize_property_key_valid(key, expected):
    """Valid keys containing spaces, dots, or hyphens should be backticked."""
    assert sanitize_property_key(key) == expected

def test_sanitize_property_key_already_backticked():
    """Keys already enclosed in backticks should remain unchanged."""
    key = "`already_backticked`"
    assert sanitize_property_key(key) == key

@pytest.mark.parametrize("invalid_key", ["inva`lid", ""])
def test_sanitize_property_key_invalid(invalid_key):
    """Keys containing backticks or empty strings should raise ValueError."""
    with pytest.raises(ValueError):
        sanitize_property_key(invalid_key)


# serialize_value tests
@pytest.mark.parametrize(
    "value,expected",
    [
        (123, 123),
        (45.6, 45.6),
        ("test", "test"),
        (True, True),
    ],
)
def test_serialize_value_primitives(value, expected):
    """Primitive types should serialize to themselves."""
    assert serialize_value(value) == expected

def test_serialize_value_datetime_and_date():
    """datetime and date objects should serialize to ISO 8601 strings."""
    dt = datetime.datetime(2021, 12, 31, 23, 59, 59)
    d = datetime.date(2021, 12, 31)
    assert serialize_value(dt) == dt.isoformat()
    assert serialize_value(d) == d.isoformat()

@pytest.mark.parametrize("lst", [[1, 2, "three"], []])
def test_serialize_value_list_of_primitives(lst):
    """Lists of primitives should serialize unchanged."""
    assert serialize_value(lst) == lst

def test_serialize_value_list_with_unsupported():
    """Lists containing unsupported types should raise TypeError."""
    with pytest.raises(TypeError):
        serialize_value([1, object(), 3])

def test_serialize_value_dict_nested():
    """Dictionaries should serialize nested values recursively."""
    data = {"int": 1, "date": datetime.date(2021, 1, 1), "nested": {"flag": False}}
    expected = {
        "int": 1,
        "date": data["date"].isoformat(),
        "nested": {"flag": False},
    }
    assert serialize_value(data) == expected

def test_serialize_value_unsupported_type():
    """Unsupported types should raise TypeError."""
    with pytest.raises(TypeError):
        serialize_value(object())


# build_cypher_match_clause tests
@pytest.mark.parametrize(
    "labels,props,expected",
    [
        ("Label", {"prop": "value"}, "MATCH (n:Label {prop:$prop})"),
        (["A", "B"], {}, "MATCH (n:A:B)"),
        ([], {"x": "y"}, "MATCH (n {x:$x})"),
        ([], {}, "MATCH (n)"),
    ],
)
def test_build_cypher_match_clause_various(labels, props, expected):
    """Generate MATCH clause with different label and property combinations."""
    assert build_cypher_match_clause(labels, props) == expected


# build_cypher_return_clause tests
@pytest.mark.parametrize(
    "props,expected",
    [
        (["prop1", "prop2"], "RETURN n.prop1, n.prop2"),
        ([], "RETURN n"),
        (["*"], "RETURN n"),
        (["dup", "dup"], "RETURN n.dup"),
    ],
)
def test_build_cypher_return_clause_various(props, expected):
    """Generate RETURN clause handling wildcards, duplicates, and empty lists."""
    assert build_cypher_return_clause(props) == expected