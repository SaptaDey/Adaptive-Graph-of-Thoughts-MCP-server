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