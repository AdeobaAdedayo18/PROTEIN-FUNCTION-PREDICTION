import networkx as nx
from ldpfp.go_hierarchy import get_ancestors, propagate_labels, build_parent_child_pairs

def toy_graph():
    g = nx.MultiDiGraph()
    g.add_edge("GO:child", "GO:parent", key=0)
    g.add_edge("GO:parent", "GO:grandparent", key=0)
    return g

def test_get_ancestors():
    g = toy_graph()
    assert get_ancestors(g, "GO:child") == {"GO:parent", "GO:grandparent"}

def test_propagate_labels_adds_ancestors():
    g = toy_graph()
    out = propagate_labels({"GO:child"}, g)
    assert out == {"GO:child", "GO:parent", "GO:grandparent"}

def test_build_parent_child_pairs():
    g = toy_graph()
    terms = ["GO:child", "GO:parent", "GO:grandparent"]
    pairs = build_parent_child_pairs(g, terms)
    assert (0, 1) in pairs  # child -> parent
    assert (1, 2) in pairs  # parent -> grandparent
