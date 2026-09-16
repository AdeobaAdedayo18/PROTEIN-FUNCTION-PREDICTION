"""Validation sanity check for the hierarchical GO classifier.

Evaluates a trained checkpoint ONLY on the validation split created
during Phase 6 training. The held-out test split is never accessed.

Reports:
    - Fmax
    - best Fmax threshold
    - micro AUPR
    - macro AUPR
    - hierarchy violation rate before consistency correction
    - hierarchy violation rate after consistency correction
    - average true labels per protein
    - average predicted labels per protein
    - several qualitative protein examples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from ldpfp.go_hierarchy import (
    load_go_graph,
    propagate_labels,
    build_parent_child_pairs,
)
from ldpfp.models.hierarchical_classifier import (
    HierarchicalGOClassifier,
    enforce_consistency,
)
from ldpfp.training_data import (
    ProteinLiteratureDataset,
    collate_protein_batch,
)
from ldpfp.evaluate import (
    fmax_score,
    aupr_scores,
    hierarchy_violation_rate,
)


# ============================================================
# Arguments
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        required=True,
    )

    parser.add_argument(
        "--pmid-map",
        required=True,
    )

    parser.add_argument(
        "--annotations",
        required=True,
    )

    parser.add_argument(
        "--go-obo",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--splits",
        default="data/splits/protein_splits.json",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=5,
    )

    return parser.parse_args()


# ============================================================
# Load embeddings
# ============================================================


def load_embeddings(path):
    print("\nLoading PMID embeddings...")

    data = np.load(path)

    pmids = data["pmids"]
    vectors = data["vectors"]

    pmid_to_vec = {
        str(pmid): vec.astype(np.float32)
        for pmid, vec in zip(pmids, vectors)
    }

    print(f"PMID embeddings: {len(pmid_to_vec):,}")
    print(f"Embedding dimension: {vectors.shape[1]}")

    return pmid_to_vec


# ============================================================
# Load protein -> PMID
# ============================================================


def load_protein_pmid_map(path):
    print("\nLoading protein → PMID mapping...")

    df = pl.read_parquet(path)

    protein_to_pmids = {}

    for row in df.iter_rows(named=True):
        protein = str(row["Protein_ID"])

        protein_to_pmids[protein] = [
            str(pmid)
            for pmid in (row["PMIDs"] or [])
            if pmid is not None
        ]

    print(
        f"Proteins with PMID mapping: "
        f"{len(protein_to_pmids):,}"
    )

    return protein_to_pmids


# ============================================================
# Load annotations
# ============================================================


def load_protein_go_annotations(path):
    print("\nLoading protein → GO annotations...")

    df = pl.read_parquet(path)

    protein_to_go = {}

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
# Reconstruct EXACT training label vocabulary
# ============================================================


def build_labels(protein_to_go, graph):
    print("\nReconstructing propagated GO vocabulary...")

    propagated_by_protein = {}
    all_terms = set()

    for protein, labels in protein_to_go.items():

        propagated = propagate_labels(
            labels,
            graph,
        )

        propagated_by_protein[protein] = propagated
        all_terms.update(propagated)

    # CRITICAL:
    # Must match Phase 6 training exactly.
    go_terms = sorted(all_terms)

    term_to_idx = {
        term: i
        for i, term in enumerate(go_terms)
    }

    n_labels = len(go_terms)

    protein_to_labels = {}

    for protein, labels in propagated_by_protein.items():

        vector = np.zeros(
            n_labels,
            dtype=np.float32,
        )

        for go_id in labels:
            vector[term_to_idx[go_id]] = 1.0

        protein_to_labels[protein] = vector

    print(f"GO vocabulary: {n_labels:,}")

    return (
        go_terms,
        term_to_idx,
        protein_to_labels,
    )


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    for path in [
        args.embeddings,
        args.pmid_map,
        args.annotations,
        args.go_obo,
        args.checkpoint,
        args.splits,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    pmid_to_vec = load_embeddings(
        args.embeddings
    )

    embed_dim = len(
        next(iter(pmid_to_vec.values()))
    )

    protein_to_pmids = load_protein_pmid_map(
        args.pmid_map
    )

    protein_to_go = load_protein_go_annotations(
        args.annotations
    )

    print("\nLoading GO DAG...")

    graph = load_go_graph(
        args.go_obo
    )

    print(f"GO DAG terms: {len(graph):,}")

    (
        go_terms,
        term_to_idx,
        protein_to_labels,
    ) = build_labels(
        protein_to_go,
        graph,
    )

    n_labels = len(go_terms)

    print("\nBuilding hierarchy relationships...")

    parent_child_pairs = build_parent_child_pairs(
        graph,
        go_terms,
    )

    print(
        f"Hierarchy pairs: "
        f"{len(parent_child_pairs):,}"
    )

    # --------------------------------------------------------
    # Load EXACT saved split
    # --------------------------------------------------------

    print("\nLoading saved dataset split...")

    with open(args.splits, "r") as f:
        splits = json.load(f)

    # Support the expected naming from train.py.
    if "val" in splits:
        val_protein_ids = splits["val"]
    elif "validation" in splits:
        val_protein_ids = splits["validation"]
    elif "val_protein_ids" in splits:
        val_protein_ids = splits["val_protein_ids"]
    else:
        raise KeyError(
            "Could not find validation protein IDs in "
            f"{args.splits}. Keys found: {list(splits)}"
        )

    val_protein_ids = [
        str(p)
        for p in val_protein_ids
    ]

    print(
        f"Validation proteins from saved split: "
        f"{len(val_protein_ids):,}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    usable_val_ids = []

    for protein in val_protein_ids:

        if protein not in protein_to_pmids:
            continue

        if protein not in protein_to_labels:
            continue

        if not any(
            pmid in pmid_to_vec
            for pmid in protein_to_pmids[protein]
        ):
            continue

        usable_val_ids.append(protein)

    if len(usable_val_ids) != len(val_protein_ids):
        raise RuntimeError(
            "Validation dataset no longer exactly matches "
            "the saved Phase 6 split. "
            f"Expected {len(val_protein_ids):,}, "
            f"usable {len(usable_val_ids):,}."
        )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = ProteinLiteratureDataset(
        protein_ids=usable_val_ids,
        protein_to_pmids=protein_to_pmids,
        protein_to_labels=protein_to_labels,
        pmid_to_vec=pmid_to_vec,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_protein_batch,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("\nLoading trained checkpoint...")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    checkpoint_n_labels = checkpoint["n_labels"]
    checkpoint_embed_dim = checkpoint["embed_dim"]

    if checkpoint_n_labels != n_labels:
        raise ValueError(
            "GO vocabulary mismatch: "
            f"checkpoint expects {checkpoint_n_labels:,}, "
            f"current data produced {n_labels:,}."
        )

    if checkpoint_embed_dim != embed_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"checkpoint expects {checkpoint_embed_dim}, "
            f"data contains {embed_dim}."
        )

    model = HierarchicalGOClassifier(
        embed_dim=checkpoint_embed_dim,
        n_labels=checkpoint_n_labels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning validation inference...")

    all_targets = []
    all_probs = []

    protein_records = []

    with torch.no_grad():

        for (
            protein_ids,
            embeddings,
            mask,
            targets,
        ) in val_loader:

            embeddings = embeddings.to(
                device,
                non_blocking=True,
            )

            mask = mask.to(
                device,
                non_blocking=True,
            )

            logits, alpha = model(
                embeddings,
                mask,
            )

            probs = torch.sigmoid(logits)

            all_targets.append(
                targets.cpu().numpy()
            )

            all_probs.append(
                probs.cpu().numpy()
            )

            alpha_cpu = alpha.cpu().numpy()

            for i, protein in enumerate(protein_ids):

                real_doc_count = int(
                    mask[i].sum().item()
                )

                protein_records.append(
                    {
                        "protein_id": protein,
                        "alpha":
                            alpha_cpu[
                                i,
                                :real_doc_count
                            ],
                    }
                )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_prob = np.concatenate(
        all_probs,
        axis=0,
    )

    print(
        f"Prediction matrix: {y_prob.shape}"
    )

    # --------------------------------------------------------
    # Fmax
    # --------------------------------------------------------

    print("\nCalculating Fmax...")

    fmax, best_threshold = fmax_score(
        y_true,
        y_prob,
    )

    # --------------------------------------------------------
    # AUPR
    # --------------------------------------------------------

    print("Calculating AUPR...")

    aupr = aupr_scores(
        y_true,
        y_prob,
    )

    # --------------------------------------------------------
    # Hierarchy violation
    # --------------------------------------------------------

    print("Calculating hierarchy violations...")

    hvr_before = hierarchy_violation_rate(
        y_prob,
        parent_child_pairs,
    )

    probs_tensor = torch.from_numpy(
        y_prob
    )

    corrected = enforce_consistency(
        probs_tensor,
        parent_child_pairs,
    ).numpy()

    hvr_after = hierarchy_violation_rate(
        corrected,
        parent_child_pairs,
    )

    # --------------------------------------------------------
    # Prediction density
    # --------------------------------------------------------

    true_counts = y_true.sum(axis=1)

    predicted_counts = (
        y_prob >= best_threshold
    ).sum(axis=1)

    avg_true = float(
        true_counts.mean()
    )

    avg_predicted = float(
        predicted_counts.mean()
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("VALIDATION SANITY CHECK")
    print("=" * 60)

    print(
        f"Validation proteins:               "
        f"{len(val_dataset):,}"
    )

    print(
        f"GO labels:                         "
        f"{n_labels:,}"
    )

    print(
        f"Fmax:                              "
        f"{fmax:.6f}"
    )

    print(
        f"Best threshold:                    "
        f"{best_threshold:.2f}"
    )

    print(
        f"AUPR (micro):                      "
        f"{aupr['aupr_micro']:.6f}"
    )

    print(
        f"AUPR (macro):                      "
        f"{aupr['aupr_macro']:.6f}"
    )

    print(
        f"Hierarchy violation rate BEFORE:   "
        f"{hvr_before:.6f}"
    )

    print(
        f"Hierarchy violation rate AFTER:    "
        f"{hvr_after:.6f}"
    )

    print(
        f"Average true labels / protein:      "
        f"{avg_true:.2f}"
    )

    print(
        f"Average predicted labels / protein: "
        f"{avg_predicted:.2f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Qualitative examples
    # --------------------------------------------------------

    print("\nQUALITATIVE EXAMPLES")
    print("=" * 60)

    n_examples = min(
        args.examples,
        len(val_dataset),
    )

    for i in range(n_examples):

        protein = protein_records[i]["protein_id"]
        alpha = protein_records[i]["alpha"]

        true_indices = np.where(
            y_true[i] == 1
        )[0]

        top_pred_indices = np.argsort(
            y_prob[i]
        )[::-1][:10]

        pmids = [
            pmid
            for pmid in protein_to_pmids[protein]
            if pmid in pmid_to_vec
        ]

        attention_order = np.argsort(
            alpha
        )[::-1]

        print(f"\nProtein: {protein}")

        print(
            f"True GO labels: "
            f"{len(true_indices)}"
        )

        print("\nTop predicted GO terms:")

        for idx in top_pred_indices[:10]:
            print(
                f"  {go_terms[idx]}  "
                f"{y_prob[i, idx]:.4f}"
            )

        print("\nTop-attended PMIDs:")

        for j in attention_order[:5]:

            if j < len(pmids):
                print(
                    f"  PMID {pmids[j]}  "
                    f"alpha={alpha[j]:.4f}"
                )

        print("-" * 60)


if __name__ == "__main__":
    main()