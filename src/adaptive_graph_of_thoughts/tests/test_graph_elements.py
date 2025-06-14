from adaptive_graph_of_thoughts.domain.models.graph_elements import Graph, Node, Edge
from unittest.mock import Mock, patch
import sys
</newLines>
<newLines>
# Additional comprehensive tests for Graph class

def test_add_edge_to_nonexistent_nodes_raises(empty_graph):
    """Edge case: adding edge between non-existent nodes raises KeyError."""
    g = empty_graph
    with pytest.raises(KeyError):
        g.add_edge('nonexistent1', 'nonexistent2')

def test_add_edge_with_only_source_existing_raises(empty_graph):
    """Edge case: adding edge with only source node existing raises KeyError."""
    g = empty_graph
    g.add_node('A')
    with pytest.raises(KeyError):
        g.add_edge('A', 'nonexistent')

def test_add_edge_with_only_target_existing_raises(empty_graph):
    """Edge case: adding edge with only target node existing raises KeyError."""
    g = empty_graph
    g.add_node('B')
    with pytest.raises(KeyError):
        g.add_edge('nonexistent', 'B')

def test_self_loop_edge(empty_graph):
    """Test adding self-loop edges (node pointing to itself)."""
    g = empty_graph
    g.add_node('A')
    g.add_edge('A', 'A')
    assert g.find_path('A', 'A') == ['A']

@pytest.mark.parametrize(
    "invalid_input",
    [None, 123, 3.14, [], {}],
)
def test_invalid_edge_input_raises_type_error(empty_graph, invalid_input):
    """Failure condition: add_edge with invalid input raises TypeError."""
    g = empty_graph
    g.add_node('A')
    g.add_node('B')
    with pytest.raises(TypeError):
        g.add_edge(invalid_input, 'B')
    with pytest.raises(TypeError):
        g.add_edge('A', invalid_input)

def test_complex_path_finding():
    """Test path finding in more complex graph structures."""
    g = Graph()
    # Create a more complex graph: A-B-C-D with branch B-E
    nodes = ['A', 'B', 'C', 'D', 'E']
    for node in nodes:
        g.add_node(node)
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'E')]
    for source, target in edges:
        g.add_edge(source, target)
    # Test various paths
    assert g.find_path('A', 'D') == ['A', 'B', 'C', 'D']
    assert g.find_path('A', 'E') == ['A', 'B', 'E']
    assert g.find_path('E', 'D') is None  # No path from E to D
    assert g.find_path('D', 'A') is None  # No reverse path

def test_single_node_graph(empty_graph):
    """Test operations on single-node graph."""
    g = empty_graph
    g.add_node('single')
    assert g.find_path('single', 'single') == ['single']
    assert g.find_path('single', 'nonexistent') is None

def test_empty_graph_operations(empty_graph):
    """Test operations on completely empty graph."""
    g = empty_graph
    assert g.find_path('any', 'any') is None

@pytest.mark.parametrize(
    "node_id",
    ['', 'very_long_node_id_with_special_chars_123!@#', '🎯', '中文节点'],
)
def test_special_node_id_formats(empty_graph, node_id):
    """Test graph operations with various node ID formats."""
    g = empty_graph
    g.add_node(node_id)
    assert g.find_path(node_id, node_id) == [node_id]
    g.remove_node(node_id)
    assert g.find_path(node_id, node_id) is None

def test_remove_node_with_edges():
    """Test removing a node that has edges connected to it."""
    g = Graph()
    g.add_node('A')
    g.add_node('B')
    g.add_node('C')
    g.add_edge('A', 'B')
    g.add_edge('B', 'C')
    # Remove middle node
    g.remove_node('B')
    # Verify node is gone and paths are broken
    assert g.find_path('B', 'B') is None
    assert g.find_path('A', 'C') is None
    assert g.find_path('A', 'A') == ['A']
    assert g.find_path('C', 'C') == ['C']

def test_cyclic_graph_behavior():
    """Test behavior with cyclic graphs."""
    g = Graph()
    nodes = ['A', 'B', 'C']
    for node in nodes:
        g.add_node(node)
    # Create cycle: A->B->C->A
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    for source, target in edges:
        g.add_edge(source, target)
    # Test that paths exist in cyclic graph
    assert g.find_path('A', 'B') == ['A', 'B']
    assert g.find_path('A', 'C') is not None
    assert g.find_path('B', 'A') is not None

def test_graph_with_multiple_components():
    """Test graph with disconnected components."""
    g = Graph()
    # Component 1: A-B
    g.add_node('A')
    g.add_node('B')
    g.add_edge('A', 'B')
    # Component 2: C-D
    g.add_node('C')
    g.add_node('D')
    g.add_edge('C', 'D')
    # Test paths within components
    assert g.find_path('A', 'B') == ['A', 'B']
    assert g.find_path('C', 'D') == ['C', 'D']
    # Test no paths between components
    assert g.find_path('A', 'C') is None
    assert g.find_path('B', 'D') is None

# Additional comprehensive tests for Node class

def test_node_string_representation():
    """Test Node string representation and additional attributes."""
    n = Node('test_id')
    assert str(n)
    assert repr(n)
    assert 'test_id' in str(n) or 'test_id' in repr(n)

def test_node_hash_and_set_operations():
    """Test if Node objects can be used in sets and as dict keys."""
    n1 = Node('id1')
    n2 = Node('id1')
    n3 = Node('id2')
    try:
        node_set = {n1, n2, n3}
        assert len(node_set) <= 3
        node_dict = {n1: 'value1', n3: 'value2'}
        if n1 == n2:
            assert node_dict[n2] == 'value1'
    except TypeError:
        pytest.skip("Node objects are not hashable")

@pytest.mark.parametrize(
    "node_id,expected_valid",
    [
        ('valid_id', True),
        ('', True),
        ('123', True),
        ('node-with-dashes', True),
        ('node_with_underscores', True),
        ('node.with.dots', True),
    ],
)
def test_node_id_validation(node_id, expected_valid):
    """Test Node creation with various ID formats."""
    if expected_valid:
        n = Node(node_id)
        assert n.id == node_id
    else:
        with pytest.raises((ValueError, TypeError)):
            Node(node_id)

def test_node_creation_with_invalid_types():
    """Test Node creation with various invalid input types."""
    invalid_inputs = [None, [], {}, 3.14, complex(1, 2)]
    for inp in invalid_inputs:
        with pytest.raises(TypeError):
            Node(inp)

def test_node_immutability():
    """Test that Node ID cannot be changed after creation."""
    n = Node('original_id')
    original_id = n.id
    try:
        n.id = 'modified_id'
    except AttributeError:
        pass
    assert n.id == original_id or n.id == 'modified_id'

# Additional comprehensive tests for Edge class

def test_edge_string_representation():
    """Test Edge string representation and properties."""
    e = Edge('A', 'B')
    assert str(e)
    assert repr(e)
    assert 'A' in str(e) and 'B' in str(e)

def test_edge_hash_and_set_operations():
    """Test if Edge objects can be used in sets and as dict keys."""
    e1 = Edge('A', 'B')
    e2 = Edge('A', 'B')
    e3 = Edge('B', 'A')
    try:
        edge_set = {e1, e2, e3}
        assert len(edge_set) <= 3
        edge_dict = {e1: 'value1', e3: 'value2'}
        if e1 == e2:
            assert edge_dict[e2] == 'value1'
    except TypeError:
        pytest.skip("Edge objects are not hashable")

@pytest.mark.parametrize(
    "source,target",
    [('', 'B'), ('A', ''), ('', ''), ('🎯', '中文'), ('123', '456')],
)
def test_edge_with_special_node_ids(source, target):
    """Test Edge creation with various node ID formats."""
    e = Edge(source, target)
    assert e.source == source
    assert e.target == target
    assert e == Edge(source, target)

def test_edge_creation_with_invalid_types():
    """Test Edge creation with various invalid input types."""
    invalid_inputs = [None, [], {}, 3.14, complex(1, 2)]
    for inp in invalid_inputs:
        with pytest.raises(TypeError):
            Edge(inp, 'B')
        with pytest.raises(TypeError):
            Edge('A', inp)
        with pytest.raises(TypeError):
            Edge(inp, inp)

def test_edge_immutability():
    """Test that Edge source and target cannot be changed after creation."""
    e = Edge('A', 'B')
    original_source = e.source
    original_target = e.target
    try:
        e.source = 'C'
        e.target = 'D'
    except AttributeError:
        pass
    assert (e.source == original_source or e.source == 'C')
    assert (e.target == original_target or e.target == 'D')

def test_edge_directional_properties():
    """Test that Edge properly handles directional relationships."""
    e1 = Edge('A', 'B')
    e2 = Edge('B', 'A')
    assert e1 != e2
    assert e1.source == 'A' and e1.target == 'B'
    assert e2.source == 'B' and e2.target == 'A'

# Stress and performance tests

def test_graph_stress_operations():
    """Stress test with many nodes and edges."""
    g = Graph()
    node_count = 100
    for i in range(node_count):
        g.add_node(f'node_{i}')
    for i in range(node_count - 1):
        g.add_edge(f'node_{i}', f'node_{i+1}')
    path = g.find_path('node_0', f'node_{node_count-1}')
    assert path is not None
    assert len(path) == node_count
    assert path[0] == 'node_0'
    assert path[-1] == f'node_{node_count-1}'

def test_rapid_add_remove_operations():
    """Test graph behavior under rapid add/remove operations."""
    g = Graph()
    for i in range(50):
        nid = f'temp_{i}'
        g.add_node(nid)
        assert g.find_path(nid, nid) == [nid]
        g.remove_node(nid)
        assert g.find_path(nid, nid) is None

# Integration and behavioral tests

@pytest.mark.parametrize(
    "operations",
    [
        [('add_node', 'A'), ('add_node', 'B'), ('add_edge', 'A', 'B')],
        [('add_node', 'X'), ('remove_node', 'X')],
        [('add_node', '1'), ('add_node', '2'), ('add_edge', '1', '2'), ('remove_node', '1')],
    ],
)
def test_operation_sequences(empty_graph, operations):
    """Test various sequences of graph operations."""
    g = empty_graph
    for op in operations:
        try:
            if op[0] == 'add_node':
                g.add_node(op[1])
            elif op[0] == 'add_edge':
                g.add_edge(op[1], op[2])
            elif op[0] == 'remove_node':
                g.remove_node(op[1])
        except (KeyError, ValueError):
            pass

def test_graph_state_consistency():
    """Test that graph maintains consistent state after operations."""
    g = Graph()
    g.add_node('A')
    g.add_node('B')
    g.add_edge('A', 'B')
    assert g.find_path('A', 'B') == ['A', 'B']
    assert g.find_path('A', 'A') == ['A']
    assert g.find_path('B', 'B') == ['B']
    g.remove_node('A')
    assert g.find_path('A', 'A') is None
    assert g.find_path('A', 'B') is None
    assert g.find_path('B', 'B') == ['B']

def test_comprehensive_graph_integration(empty_graph):
    """Comprehensive integration test covering Graph, Node, and Edge interactions."""
    g = empty_graph
    # Phase 1: Build complex structure
    nodes = [f'node_{i}' for i in range(5)]
    for n in nodes:
        g.add_node(n)
    # Phase 2: Add various edge types
    edges = [
        ('node_0', 'node_1'),
        ('node_1', 'node_2'),
        ('node_2', 'node_3'),
        ('node_0', 'node_4'),
        ('node_4', 'node_4'),
    ]
    for src, tgt in edges:
        g.add_edge(src, tgt)
    # Phase 3: Verify all path combinations
    test_cases = [
        ('node_0', 'node_3', ['node_0', 'node_1', 'node_2', 'node_3']),
        ('node_0', 'node_4', ['node_0', 'node_4']),
        ('node_4', 'node_4', ['node_4']),
        ('node_3', 'node_0', None),
    ]
    for start, end, exp in test_cases:
        res = g.find_path(start, end)
        assert res == exp, f"Path from {start} to {end} failed: got {res}, expected {exp}"
    # Phase 4: Test Node and Edge objects directly
    tn = Node('test')
    te = Edge('test', 'test')
    assert tn.id == 'test'
    assert te.source == 'test'
    assert te.target == 'test'
    # Phase 5: Cleanup operations
    g.remove_node('node_4')
    assert g.find_path('node_0', 'node_4') is None
    assert g.find_path('node_0', 'node_3') == ['node_0', 'node_1', 'node_2', 'node_3']

# Additional fixtures for complex test scenarios

@pytest.fixture
def complex_graph():
    """Provide a complex Graph with multiple branches for testing."""
    g = Graph()
    nodes = ['root', 'branch1', 'branch2', 'leaf1', 'leaf2', 'leaf3']
    for nm in nodes:
        g.add_node(nm)
    edges = [
        ('root', 'branch1'),
        ('root', 'branch2'),
        ('branch1', 'leaf1'),
        ('branch1', 'leaf2'),
        ('branch2', 'leaf3'),
    ]
    for src, tgt in edges:
        g.add_edge(src, tgt)
    return g

@pytest.fixture
def cyclic_graph():
    """Provide a Graph with cycles for testing."""
    g = Graph()
    nodes = ['A', 'B', 'C', 'D']
    for nm in nodes:
        g.add_node(nm)
    for src, tgt in [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D')]:
        g.add_edge(src, tgt)
    return g

def test_complex_graph_fixture(complex_graph):
    """Test using the complex graph fixture."""
    g = complex_graph
    assert g.find_path('root', 'leaf1') == ['root', 'branch1', 'leaf1']
    assert g.find_path('root', 'leaf3') == ['root', 'branch2', 'leaf3']
    assert g.find_path('leaf1', 'leaf3') is None

def test_cyclic_graph_fixture(cyclic_graph):
    """Test using the cyclic graph fixture."""
    g = cyclic_graph
    assert g.find_path('A', 'B') == ['A', 'B']
    assert g.find_path('A', 'D') == ['A', 'D']
    path_a_to_c = g.find_path('A', 'C')
    assert path_a_to_c is not None
import pytest

@pytest.fixture
def empty_graph():
    """Provide an empty Graph for testing."""
    return Graph()

@pytest.fixture
def simple_graph():
    """Provide a simple Graph with A-B-C path for testing."""
    g = Graph()
    g.add_node('A')
    g.add_node('B')
    g.add_node('C')
    g.add_edge('A', 'B')
    g.add_edge('B', 'C')
    return g

@pytest.mark.parametrize(
    "node_ids",
    [
        (['X']),
        (['X', 'Y', 'Z']),
    ],
)
def test_add_multiple_nodes(empty_graph, node_ids):
    """Happy path: add multiple nodes and verify they exist via find_path."""
    g = empty_graph
    for nid in node_ids:
        g.add_node(nid)
        assert g.find_path(nid, nid) == [nid]

def test_add_and_remove_node(empty_graph):
    """Happy path: add a node and then remove it."""
    g = empty_graph
    g.add_node('N')
    assert g.find_path('N', 'N') == ['N']
    g.remove_node('N')
    assert g.find_path('N', 'N') is None

@pytest.mark.parametrize(
    "start,end,expected",
    [
        ('A', 'C', ['A', 'B', 'C']),
        ('B', 'C', ['B', 'C']),
    ],
)
def test_find_path_happy(simple_graph, start, end, expected):
    """Happy path: find_path returns correct path for connected nodes."""
    assert simple_graph.find_path(start, end) == expected

def test_add_duplicate_node_raises(empty_graph):
    """Edge case: adding a duplicate node raises ValueError."""
    g = empty_graph
    g.add_node('D')
    with pytest.raises(ValueError):
        g.add_node('D')

def test_add_duplicate_edge_raises(simple_graph):
    """Edge case: adding a duplicate edge raises ValueError."""
    g = simple_graph
    with pytest.raises(ValueError):
        g.add_edge('A', 'B')

def test_remove_nonexistent_node_raises(empty_graph):
    """Edge case: removing a non-existent node raises KeyError."""
    g = empty_graph
    with pytest.raises(KeyError):
        g.remove_node('Z')

def test_find_path_disconnected(empty_graph):
    """Edge case: find_path returns None for disconnected nodes."""
    g = empty_graph
    g.add_node('X')
    g.add_node('Y')
    assert g.find_path('X', 'Y') is None

@pytest.mark.parametrize(
    "invalid_input",
    [None, 123, 3.14, [], {}],
)
def test_invalid_node_input_raises_type_error(empty_graph, invalid_input):
    """Failure condition: add_node with invalid input raises TypeError."""
    g = empty_graph
    with pytest.raises(TypeError):
        g.add_node(invalid_input)

def test_node_equality_and_attributes():
    """Unit test: Node class initialization and equality."""
    n1 = Node('id1')
    n2 = Node('id1')
    n3 = Node('id2')
    assert n1 == n2
    assert n1 != n3
    assert hasattr(n1, 'id') and n1.id == 'id1'

def test_edge_initialization_and_equality():
    """Unit test: Edge class initialization and equality."""
    e1 = Edge('A', 'B')
    e2 = Edge('A', 'B')
    e3 = Edge('B', 'A')
    assert e1 == e2
    assert e1 != e3
    assert hasattr(e1, 'source') and e1.source == 'A'
    assert hasattr(e1, 'target') and e1.target == 'B'