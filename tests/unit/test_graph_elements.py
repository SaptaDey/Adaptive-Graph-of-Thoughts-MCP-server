import pytest

from graph_elements import Node, Edge, Graph


@pytest.fixture
def small_graph():
    g = Graph()
    g.add_node(Node(1, label='A'))
    g.add_node(Node(2, label='B'))
    g.add_edge(Edge(1, 2))
    return g


@pytest.fixture
def cyclic_graph():
    g = Graph()
    g.add_node(Node(1))
    g.add_node(Node(2))
    g.add_edge(Edge(1, 2))
    g.add_edge(Edge(2, 1))
    return g


@pytest.fixture
def large_graph():
    g = Graph()
    for i in range(5):
        g.add_node(Node(i))
    for i in range(4):
        g.add_edge(Edge(i, i + 1))
    return g


def test_node_creation_valid():
    node = Node('node1', label='Test')
    assert node.id == 'node1'
    assert node.label == 'Test'
    assert repr(node) == "Node(id='node1', label='Test')"


def test_node_invalid_id():
    with pytest.raises(ValueError):
        Node(None)
    with pytest.raises(ValueError):
        Node('')


@pytest.mark.parametrize('id1,id2,expected', [
    (1, 1, True),
    (1, 2, False),
    ('a', 'a', True),
    ('a', 'b', False),
])
def test_node_eq_hash_behaviour(id1, id2, expected):
    n1 = Node(id1)
    n2 = Node(id2)
    assert (n1 == n2) is expected
    if expected:
        assert hash(n1) == hash(n2)
    else:
        assert hash(n1) != hash(n2)


def test_edge_connects_existing_nodes():
    g = Graph()
    g.add_node(Node(1))
    g.add_node(Node(2))
    e = Edge(1, 2)
    g.add_edge(e)
    assert e in g.edges


def test_edge_self_loop_error():
    with pytest.raises(ValueError):
        Edge(1, 1)


@pytest.mark.parametrize('src,dst,src2,dst2,expected', [
    (1, 2, 1, 2, True),
    (1, 2, 2, 1, False),
])
def test_edge_eq_hash_behaviour(src, dst, src2, dst2, expected):
    e1 = Edge(src, dst)
    e2 = Edge(src2, dst2)
    assert (e1 == e2) is expected
    if expected:
        assert hash(e1) == hash(e2)
    else:
        assert hash(e1) != hash(e2)


def test_edge_repr():
    edge = Edge(1, 2)
    assert repr(edge) == "Edge(src=1, dst=2)"


def test_graph_add_remove_nodes_edges(small_graph):
    g = small_graph
    # add a new node and edge
    extra = Node(3)
    g.add_node(extra)
    assert extra in g.nodes
    e = Edge(2, 3)
    g.add_edge(e)
    assert e in g.edges
    # remove them
    g.remove_edge(e)
    assert e not in g.edges
    g.remove_node(extra)
    assert extra not in g.nodes


def test_graph_duplicate_node():
    g = Graph()
    g.add_node(Node(1))
    with pytest.raises(ValueError):
        g.add_node(Node(1))


def test_graph_duplicate_edge_error():
    g = Graph()
    g.add_node(Node(1))
    g.add_node(Node(2))
    e = Edge(1, 2)
    g.add_edge(e)
    with pytest.raises(ValueError):
        g.add_edge(Edge(1, 2))


@pytest.mark.skipif(not hasattr(Graph, 'has_cycle'), reason="cycle detection not implemented")
def test_graph_cycle_detection_if_applicable(cyclic_graph):
    assert cyclic_graph.has_cycle() is True


@pytest.mark.skipif(
    not (hasattr(Graph, 'serialize') and hasattr(Graph, 'deserialize')),
    reason="serialization not implemented"
)
def test_graph_serialization_roundtrip(tmp_path, small_graph):
    file_path = tmp_path / "graph.json"
    small_graph.serialize(str(file_path))
    assert file_path.exists()
    loaded = Graph.deserialize(str(file_path))
    assert loaded.nodes == small_graph.nodes
    assert loaded.edges == small_graph.edges