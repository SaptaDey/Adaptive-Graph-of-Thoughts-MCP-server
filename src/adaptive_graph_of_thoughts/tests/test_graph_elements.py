@pytest.fixture
def empty_graph():
    """
    Provides an empty Graph instance for use in tests.
    """
    return Graph()

@pytest.fixture
def simple_graph():
    """
    Provides a Graph instance with nodes 'A', 'B', and 'C' connected in a linear path A-B-C for use in tests.
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
    Adds multiple nodes to an empty graph and verifies each node exists by checking that a path from the node to itself is found.
    
    Args:
        node_ids: Iterable of node identifiers to add to the graph.
    """
    g = empty_graph
    for nid in node_ids:
        g.add_node(nid)
        assert g.find_path(nid, nid) == [nid]

def test_add_and_remove_node(empty_graph):
    """
    Tests adding a node to an empty graph and then removing it.
    
    Verifies that the node can be found after addition and is no longer present after removal.
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
    Tests that find_path returns the expected path between connected nodes in the simple graph.
    
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
    
    Verifies that attempting to add an edge that already exists between two nodes results in a ValueError being raised.
    """
    g = simple_graph
    with pytest.raises(ValueError):
        g.add_edge('A', 'B')

def test_remove_nonexistent_node_raises(empty_graph):
    """
    Tests that attempting to remove a node that does not exist in the graph raises a KeyError.
    """
    g = empty_graph
    with pytest.raises(KeyError):
        g.remove_node('Z')

def test_find_path_disconnected(empty_graph):
    """
    Tests that find_path returns None when searching between disconnected nodes.
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
    
    Verifies that the graph's add_node method rejects invalid node types such as None, numbers, lists, or dictionaries by raising a TypeError.
    """
    g = empty_graph
    with pytest.raises(TypeError):
        g.add_node(invalid_input)

def test_node_equality_and_attributes():
    """
    Tests Node class initialization, attribute assignment, and equality behavior.
    
    Verifies that nodes with the same ID are equal, nodes with different IDs are not equal, and that the 'id' attribute is correctly set.
    """
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