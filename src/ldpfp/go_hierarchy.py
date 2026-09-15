"""Phase 6a: GO DAG utilities — parent lookup, ancestor propagation, consistency check."""
from __future__ import annotations
import obonet
import networkx as nx

def load_go_graph(obo_path: str) -> nx.MultiDiGraph:
    """obonet builds a graph where edges point child -> parent (is_a)."""
    return obonet.read_obo(obo_path)

def get_ancestors(graph: nx.MultiDiGraph, go_id: str) -> set[str]:
    """All ancestor GO terms of go_id (i.e., every parent transitively required by Eq. 17)."""
    if go_id not in graph:
        return set()
    return nx.descendants(graph, go_id)  # obonet edges child->parent, so descendants() = ancestors

def propagate_labels(labels: set[str], graph: nx.MultiDiGraph) -> set[str]:
    """True-path rule: if a child term is annotated, all its ancestors are implicitly
    annotated too. Used both to enrich training labels and to build the parent index
    used by the hierarchical loss."""
    expanded = set(labels)
    for go_id in list(labels):
        expanded |= get_ancestors(graph, go_id)
    return expanded

def build_parent_child_pairs(graph: nx.MultiDiGraph, go_terms: list[str]) -> list[tuple[int, int]]:
    """Index pairs (child_idx, parent_idx) within a fixed label vocabulary, used to
    enforce ŷ(g_c) <= ŷ(g_p) (Eq. 18) during training."""
    term_to_idx = {t: i for i, t in enumerate(go_terms)}
    pairs = []
    for child in go_terms:
        if child not in graph:
            continue
        for parent in graph.successors(child):  # child -> parent (is_a)
            if parent in term_to_idx:
                pairs.append((term_to_idx[child], term_to_idx[parent]))
    return pairs
