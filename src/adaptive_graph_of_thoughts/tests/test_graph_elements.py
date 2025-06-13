@pytest.fixture
def empty_graph():
    """
    Provides an empty Graph instance for use in tests.
    """
    return Graph()

@pytest.fixture
def simple_graph():
    """
    Provides a Graph instance with three nodes ('A', 'B', 'C') connected in a linear path (A-B-C) for use in tests.
    """
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
    """
    Adds multiple nodes to an empty graph and verifies each node can be found via a path from itself to itself.
    
    Args:
        node_ids: A list of node identifiers to add to the graph.
    """
    g = empty_graph
    for nid in node_ids:
        g.add_node(nid)
        assert g.find_path(nid, nid) == [nid]

def test_add_and_remove_node(empty_graph):
    """
    Tests adding a node to the graph and then removing it.
    
    Verifies that the node can be found after addition and is no longer found after removal.
    """
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
    """
    Tests that find_path returns the correct path between connected nodes in the simple graph.
    
    Args:
        start: The starting node ID.
        end: The ending node ID.
        expected: The expected list of node IDs representing the path.
    """
    assert simple_graph.find_path(start, end) == expected

def test_add_duplicate_node_raises(empty_graph):
    """
    Tests that adding a duplicate node to the graph raises a ValueError.
    """
    g = empty_graph
    g.add_node('D')
    with pytest.raises(ValueError):
        g.add_node('D')

def test_add_duplicate_edge_raises(simple_graph):
    """
    Tests that adding a duplicate edge to the graph raises a ValueError.
    """
    g = simple_graph
    with pytest.raises(ValueError):
        g.add_edge('A', 'B')

def test_remove_nonexistent_node_raises(empty_graph):
    """
    Tests that removing a non-existent node from the graph raises a KeyError.
    """
    g = empty_graph
    with pytest.raises(KeyError):
        g.remove_node('Z')

def test_find_path_disconnected(empty_graph):
    """
    Tests that find_path returns None when there is no path between disconnected nodes.
    """
    g = empty_graph
    g.add_node('X')
    g.add_node('Y')
    assert g.find_path('X', 'Y') is None

@pytest.mark.parametrize(
    "invalid_input",
    [None, 123, 3.14, [], {}],
)
def test_invalid_node_input_raises_type_error(empty_graph, invalid_input):
    """
    Tests that adding a node with invalid input to the graph raises a TypeError.
    
    Verifies that the graph's add_node method enforces input type validation by raising
    a TypeError when provided with unsupported types.
    """
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
    """
    Tests Edge class initialization, attribute assignment, and equality semantics.
    
    Verifies that Edge instances with the same source and target are equal, while those with reversed endpoints are not. Also checks that the 'source' and 'target' attributes are correctly set.
    """
    e1 = Edge('A', 'B')
    e2 = Edge('A', 'B')
    e3 = Edge('B', 'A')
    assert e1 == e2
    assert e1 != e3
    assert hasattr(e1, 'source') and e1.source == 'A'
    assert hasattr(e1, 'target') and e1.target == 'B'