"""Phase 6d runner: construct protein-level training data and train the
hierarchical GO-aware classifier.

Pipeline:
    consolidated PMID embeddings
        +
    protein -> PMID mapping
        +
    protein -> GO annotations
        +
    GO hierarchy
        ↓
    ProteinLiteratureDataset
        ↓
    HierarchicalGOClassifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ldpfp.go_hierarchy import (
    load_go_graph,
    propagate_labels,
    build_parent_child_pairs,
)
from ldpfp.training_data import ProteinLiteratureDataset
from ldpfp.train import train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        default="data/processed/embeddings/pmid_embeddings.npz",
    )

    parser.add_argument(
        "--pmid-map",
        default="data/interim/protein_pmid_map.parquet",
    )

    parser.add_argument(
        "--annotations",
        default="data/interim/annotations.parquet",
    )

    parser.add_argument(
        "--go-obo",
        default="data/external/go-basic.obo",
    )

    parser.add_argument(
        "--model-output",
        default="data/models/hierarchical_go_classifier.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lambda-hierarchy",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Embeddings
# ============================================================


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    print("\nLoading PMID embeddings...")

    data = np.load(path)

    pmids = data["pmids"]
    vectors = data["vectors"]

    if len(pmids) != len(vectors):
        raise ValueError(
            "PMID count does not match embedding vector count."
        )

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2-D embedding matrix, got {vectors.shape}"
        )

    pmid_to_vec = {
        str(pmid): vector.astype(np.float32)
        for pmid, vector in zip(pmids, vectors)
    }

    print(f"PMID embeddings: {len(pmid_to_vec):,}")
    print(f"Embedding dimension: {vectors.shape[1]}")

    return pmid_to_vec


# ============================================================
# Protein -> PMID mapping
# ============================================================


def load_protein_pmid_map(path: str) -> dict[str, list[str]]:
    print("\nLoading protein → PMID mapping...")

    df = pl.read_parquet(path)

    print("Columns:", df.columns)

    required = {"Protein_ID", "PMIDs"}

    if not required.issubset(df.columns):
        raise ValueError(
            "protein_pmid_map.parquet must contain "
            f"{required}. Found: {df.columns}"
        )

    protein_to_pmids = {}

    for row in df.iter_rows(named=True):

        protein = str(row["Protein_ID"])

        raw_pmids = row["PMIDs"] or []

        pmids = [
            str(pmid)
            for pmid in raw_pmids
            if pmid is not None
        ]

        protein_to_pmids[protein] = pmids

    print(
        f"Proteins with PMID mapping: "
        f"{len(protein_to_pmids):,}"
    )

    return protein_to_pmids


# ============================================================
# Protein -> GO annotations
# ============================================================


def load_protein_go_annotations(
    path: str,
) -> dict[str, set[str]]:

    print("\nLoading protein → GO annotations...")

    df = pl.read_parquet(path)

    required = {"Protein_ID", "GO_Label"}

    if not required.issubset(df.columns):
        raise ValueError(
            "annotations.parquet must contain "
            f"{required}. Found: {df.columns}"
        )

    protein_to_go: dict[str, set[str]] = {}

    for protein, go_id in df.select(
        ["Protein_ID", "GO_Label"]
    ).iter_rows():

        if protein is None or go_id is None:
            continue

        protein = str(protein)
        go_id = str(go_id)

        protein_to_go.setdefault(
            protein,
            set(),
        ).add(go_id)

    print(
        f"Proteins with GO annotations: "
        f"{len(protein_to_go):,}"
    )

    return protein_to_go
# ============================================================
# GO propagation
# ============================================================


def build_label_vocabulary(
    protein_to_go: dict[str, set[str]],
    graph,
):
    print("\nPropagating GO labels...")

    propagated_by_protein = {}

    all_terms = set()

    for protein, labels in protein_to_go.items():

        propagated = propagate_labels(
            labels,
            graph,
        )

        propagated_by_protein[protein] = propagated

        all_terms.update(propagated)

    # Deterministic ordering is critical.
    go_terms = sorted(all_terms)

    term_to_idx = {
        term: i
        for i, term in enumerate(go_terms)
    }

    print(
        f"Propagated GO vocabulary: "
        f"{len(go_terms):,}"
    )

    return (
        propagated_by_protein,
        go_terms,
        term_to_idx,
    )


# ============================================================
# Multi-hot labels
# ============================================================


def build_multihot_labels(
    propagated_by_protein,
    term_to_idx,
):
    print("\nBuilding protein multi-hot labels...")

    n_labels = len(term_to_idx)

    protein_to_labels = {}

    for protein, labels in propagated_by_protein.items():

        vector = np.zeros(
            n_labels,
            dtype=np.float32,
        )

        for go_id in labels:

            idx = term_to_idx.get(go_id)

            if idx is not None:
                vector[idx] = 1.0

        protein_to_labels[protein] = vector

    print(
        f"Label vectors constructed: "
        f"{len(protein_to_labels):,}"
    )

    return protein_to_labels


# ============================================================
# Alignment checks
# ============================================================


def alignment_report(
    protein_to_pmids,
    protein_to_labels,
    pmid_to_vec,
):
    print("\nChecking dataset alignment...")

    pmid_proteins = set(protein_to_pmids)
    label_proteins = set(protein_to_labels)

    common = pmid_proteins & label_proteins

    proteins_with_embeddings = []

    missing_pmids = set()

    for protein in common:

        available = False

        for pmid in protein_to_pmids[protein]:

            if pmid in pmid_to_vec:
                available = True
            else:
                missing_pmids.add(pmid)

        if available:
            proteins_with_embeddings.append(
                protein
            )

    print(
        f"Proteins in PMID map:       "
        f"{len(pmid_proteins):,}"
    )

    print(
        f"Proteins with GO labels:    "
        f"{len(label_proteins):,}"
    )

    print(
        f"Protein intersection:       "
        f"{len(common):,}"
    )

    print(
        f"Trainable proteins:         "
        f"{len(proteins_with_embeddings):,}"
    )

    print(
        f"Referenced PMIDs missing "
        f"embeddings:                 "
        f"{len(missing_pmids):,}"
    )

    if not proteins_with_embeddings:
        raise ValueError(
            "No proteins survived alignment. "
            "Check protein identifier formats."
        )

    return sorted(proteins_with_embeddings)


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    # --------------------------------------------------------
    # Verify inputs
    # --------------------------------------------------------

    for path in [
        args.embeddings,
        args.pmid_map,
        args.annotations,
        args.go_obo,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    # --------------------------------------------------------
    # Load consolidated embeddings
    # --------------------------------------------------------

    pmid_to_vec = load_embeddings(
        args.embeddings
    )

    embed_dim = len(
        next(iter(pmid_to_vec.values()))
    )

    # --------------------------------------------------------
    # Load mappings
    # --------------------------------------------------------

    protein_to_pmids = load_protein_pmid_map(
        args.pmid_map
    )

    protein_to_go = load_protein_go_annotations(
        args.annotations
    )

    # --------------------------------------------------------
    # GO DAG
    # --------------------------------------------------------

    print("\nLoading GO DAG...")

    graph = load_go_graph(
        args.go_obo
    )

    print(
        f"GO DAG terms: "
        f"{len(graph):,}"
    )

    # --------------------------------------------------------
    # Propagate labels
    # --------------------------------------------------------

    (
        propagated_by_protein,
        go_terms,
        term_to_idx,
    ) = build_label_vocabulary(
        protein_to_go,
        graph,
    )

    n_labels = len(go_terms)

    # --------------------------------------------------------
    # Hierarchy relationships
    # --------------------------------------------------------

    print(
        "\nBuilding parent-child pairs..."
    )

    parent_child_pairs = (
        build_parent_child_pairs(
            graph,
            go_terms,
        )
    )

    print(
        f"Parent-child pairs: "
        f"{len(parent_child_pairs):,}"
    )

    # --------------------------------------------------------
    # Multi-hot targets
    # --------------------------------------------------------

    protein_to_labels = (
        build_multihot_labels(
            propagated_by_protein,
            term_to_idx,
        )
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    protein_ids = alignment_report(
        protein_to_pmids,
        protein_to_labels,
        pmid_to_vec,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nConstructing dataset...")

    dataset = ProteinLiteratureDataset(
        protein_ids=protein_ids,
        protein_to_pmids=protein_to_pmids,
        protein_to_labels=protein_to_labels,
        pmid_to_vec=pmid_to_vec,
    )

    print(
        f"Final dataset proteins: "
        f"{len(dataset):,}"
    )

    if len(dataset) != len(protein_ids):
        print(
            "WARNING: Some proteins were removed "
            "during dataset construction."
        )

    # --------------------------------------------------------
    # Final summary BEFORE expensive training
    # --------------------------------------------------------

    print("\n================================")
    print("TRAINING DATA SUMMARY")
    print("================================")

    print(
        f"Proteins:          "
        f"{len(dataset):,}"
    )

    print(
        f"PMID embeddings:   "
        f"{len(pmid_to_vec):,}"
    )

    print(
        f"Embedding dim:      "
        f"{embed_dim}"
    )

    print(
        f"GO labels:          "
        f"{n_labels:,}"
    )

    print(
        f"Hierarchy pairs:    "
        f"{len(parent_child_pairs):,}"
    )

    print("================================\n")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, history = train(
        dataset=dataset,
        parent_child_pairs=parent_child_pairs,
        n_labels=n_labels,
        embed_dim=embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lam=args.lambda_hierarchy,
        device=args.device,
        seed=args.seed,
        model_output_path=args.model_output,
        split_output_path="data/splits/protein_splits.json",
    )

    print("\nTraining complete.")

    print(
        f"Final train loss: "
        f"{history['train_loss'][-1]:.6f}"
    )

    print(
        f"Final validation loss: "
        f"{history['val_loss'][-1]:.6f}"
    )


if __name__ == "__main__":
    main()