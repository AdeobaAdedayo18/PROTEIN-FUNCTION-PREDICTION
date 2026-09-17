"""Phase 7: Final held-out evaluation of the hierarchical GO classifier.

Evaluates the frozen best checkpoint on the test split that was reserved
during Phase 6 training.

Reports:
    - CAFA-style protein-centric Fmax
    - precision / recall / F1 at the validation-selected threshold
    - micro-AUPR
    - macro-AUPR over GO terms represented in the test split
    - hierarchy violation rate before correction
    - hierarchy violation rate after correction
    - BP / MF / CC-specific metrics
    - qualitative attention examples

The test split is NEVER used for model selection or retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from ldpfp.evaluate import (
    fmax_score,
    aupr_scores,
    hierarchy_violation_rate,
    precision_recall_at_threshold,
)
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


# ============================================================
# Arguments
# ============================================================


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
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--splits",
        default="data/splits/protein_splits.json",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
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
        "--validation-threshold",
        type=float,
        default=0.27,
        help=(
            "Operating threshold selected on validation data. "
            "Default corresponds to the best threshold from the "
            "30-epoch training run."
        ),
    )

    parser.add_argument(
        "--qualitative-examples",
        type=int,
        default=5,
    )

    return parser.parse_args()


# ============================================================
# Loading utilities
# ============================================================


def load_embeddings(path):
    print("\nLoading PMID embeddings...")

    data = np.load(path)

    pmids = data["pmids"]
    vectors = data["vectors"]

    pmid_to_vec = {
        str(pmid): vector.astype(np.float32)
        for pmid, vector in zip(pmids, vectors)
    }

    print(f"PMID embeddings: {len(pmid_to_vec):,}")
    print(f"Embedding dimension: {vectors.shape[1]}")

    return pmid_to_vec


def load_protein_pmid_map(path):
    print("\nLoading protein → PMID mapping...")

    df = pl.read_parquet(path)

    protein_to_pmids = {}

    for row in df.iter_rows(named=True):
        protein = str(row["Protein_ID"])

        raw_pmids = row["PMIDs"] or []

        protein_to_pmids[protein] = [
            str(pmid)
            for pmid in raw_pmids
            if pmid is not None
        ]

    print(
        f"Proteins with PMID mapping: "
        f"{len(protein_to_pmids):,}"
    )

    return protein_to_pmids


def load_annotations(path):
    print("\nLoading protein → GO annotations...")

    df = pl.read_parquet(path)

    protein_to_go = {}
    go_category = {}

    for row in df.iter_rows(named=True):

        protein = row["Protein_ID"]
        go_id = row["GO_Label"]

        if protein is None or go_id is None:
            continue

        protein = str(protein)
        go_id = str(go_id)

        protein_to_go.setdefault(
            protein,
            set(),
        ).add(go_id)

        if "GO_Category" in df.columns:
            category = row.get("GO_Category")

            if category is not None:
                go_category[go_id] = str(category)

    print(
        f"Proteins with GO annotations: "
        f"{len(protein_to_go):,}"
    )

    return protein_to_go, go_category


# ============================================================
# Reconstruct training vocabulary
# ============================================================


def build_label_vocabulary(
    protein_to_go,
    graph,
):
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

    # MUST match Phase 6 training exactly.
    go_terms = sorted(all_terms)

    term_to_idx = {
        term: i
        for i, term in enumerate(go_terms)
    }

    print(
        f"GO vocabulary: {len(go_terms):,}"
    )

    return (
        propagated_by_protein,
        go_terms,
        term_to_idx,
    )


def build_multihot_labels(
    propagated_by_protein,
    term_to_idx,
):
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

    return protein_to_labels


# ============================================================
# Category handling
# ============================================================


def normalize_category(category):
    if category is None:
        return None

    category = category.lower().strip()

    if category in {
        "biological process",
        "biological_process",
        "bp",
    }:
        return "BP"

    if category in {
        "molecular function",
        "molecular_function",
        "mf",
    }:
        return "MF"

    if category in {
        "cellular component",
        "cellular_component",
        "cc",
    }:
        return "CC"

    return None


def infer_go_categories(
    graph,
    go_terms,
    annotation_categories,
):
    """Determine BP/MF/CC for every propagated GO term.

    Directly annotated terms use GO_Category from annotations.parquet.

    Ancestor terms added during propagation may not occur directly in
    annotations.parquet, so their namespace is recovered from the GO DAG.
    """

    categories = {}

    namespace_map = {
        "biological_process": "BP",
        "molecular_function": "MF",
        "cellular_component": "CC",
    }

    for go_id in go_terms:

        direct_category = normalize_category(
            annotation_categories.get(go_id)
        )

        if direct_category is not None:
            categories[go_id] = direct_category
            continue

        if go_id in graph:

            namespace = graph.nodes[go_id].get(
                "namespace"
            )

            categories[go_id] = namespace_map.get(
                namespace
            )

    return categories


# ============================================================
# Metric helpers
# ============================================================


def f1_at_threshold(
    y_true,
    y_prob,
    threshold,
):
    precision, recall = precision_recall_at_threshold(
        y_true,
        y_prob,
        threshold,
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_matrix(
    y_true,
    y_prob,
    parent_child_pairs,
    validation_threshold,
):
    fmax, test_best_threshold = fmax_score(
        y_true,
        y_prob,
    )

    aupr = aupr_scores(
        y_true,
        y_prob,
    )

    operating = f1_at_threshold(
        y_true,
        y_prob,
        validation_threshold,
    )

    hvr = hierarchy_violation_rate(
        y_prob,
        parent_child_pairs,
    )

    return {
    "fmax": float(fmax),
    "test_fmax_threshold": float(test_best_threshold),
    "validation_threshold": float(validation_threshold),

    "precision_at_validation_threshold":
        operating["precision"],
    "recall_at_validation_threshold":
        operating["recall"],
    "f1_at_validation_threshold":
        operating["f1"],

    "aupr_micro": float(
        aupr["aupr_micro"]
    ),
    "aupr_macro": float(
        aupr["aupr_macro"]
    ),

    "macro_evaluable_terms": int(
        aupr["macro_evaluable_terms"]
    ),
    "macro_total_terms": int(
        aupr["macro_total_terms"]
    ),

    "hierarchy_violation_rate": float(hvr),
    }


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    required_paths = [
        args.embeddings,
        args.pmid_map,
        args.annotations,
        args.go_obo,
        args.checkpoint,
        args.splits,
    ]

    for path in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    device = torch.device(device)

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Load source data
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

    (
        protein_to_go,
        annotation_categories,
    ) = load_annotations(
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
        f"GO DAG terms: {len(graph):,}"
    )

    # --------------------------------------------------------
    # Reconstruct exact vocabulary
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

    parent_child_pairs = (
        build_parent_child_pairs(
            graph,
            go_terms,
        )
    )

    print(
        f"Hierarchy pairs: "
        f"{len(parent_child_pairs):,}"
    )

    protein_to_labels = build_multihot_labels(
        propagated_by_protein,
        term_to_idx,
    )

    # --------------------------------------------------------
    # GO categories
    # --------------------------------------------------------

    go_categories = infer_go_categories(
        graph,
        go_terms,
        annotation_categories,
    )

    category_indices = {
        category: [
            i
            for i, go_id in enumerate(go_terms)
            if go_categories.get(go_id)
            == category
        ]
        for category in ["BP", "MF", "CC"]
    }

    print("\nGO category sizes:")

    for category, indices in category_indices.items():
        print(
            f"{category}: {len(indices):,}"
        )

    # --------------------------------------------------------
    # Saved split
    # --------------------------------------------------------

    print("\nLoading saved dataset split...")

    with open(
        args.splits,
        "r",
        encoding="utf-8",
    ) as f:
        split_data = json.load(f)

    # Support the likely naming variants while remaining
    # explicit about what is being used.
    test_ids = (
        split_data.get("test")
        or split_data.get("test_ids")
        or split_data.get("test_protein_ids")
    )

    if test_ids is None:
        raise KeyError(
            "Could not find test protein IDs in "
            f"{args.splits}. Keys found: "
            f"{list(split_data.keys())}"
        )

    test_ids = [
        str(protein)
        for protein in test_ids
    ]

    print(
        f"Test proteins from saved split: "
        f"{len(test_ids):,}"
    )

    # --------------------------------------------------------
    # Check test proteins
    # --------------------------------------------------------

    missing_labels = [
        p
        for p in test_ids
        if p not in protein_to_labels
    ]

    missing_pmids = [
        p
        for p in test_ids
        if p not in protein_to_pmids
    ]

    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} test proteins "
            "are missing reconstructed GO labels."
        )

    if missing_pmids:
        raise ValueError(
            f"{len(missing_pmids)} test proteins "
            "are missing PMID mappings."
        )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = ProteinLiteratureDataset(
        protein_ids=test_ids,
        protein_to_pmids=protein_to_pmids,
        protein_to_labels=protein_to_labels,
        pmid_to_vec=pmid_to_vec,
    )

    if len(test_dataset) != len(test_ids):
        raise ValueError(
            "Test dataset size changed during "
            "ProteinLiteratureDataset construction. "
            f"Expected {len(test_ids)}, "
            f"got {len(test_dataset)}."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_protein_batch,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    print("\nLoading frozen checkpoint...")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    checkpoint_n_labels = checkpoint.get(
        "n_labels"
    )

    checkpoint_embed_dim = checkpoint.get(
        "embed_dim"
    )

    if checkpoint_n_labels != n_labels:
        raise ValueError(
            "GO vocabulary size does not match checkpoint. "
            f"Checkpoint={checkpoint_n_labels}, "
            f"reconstructed={n_labels}"
        )

    if checkpoint_embed_dim != embed_dim:
        raise ValueError(
            "Embedding dimension does not match checkpoint. "
            f"Checkpoint={checkpoint_embed_dim}, "
            f"loaded={embed_dim}"
        )

    print(
        f"Checkpoint GO labels: "
        f"{checkpoint_n_labels:,}"
    )

    print(
        f"Checkpoint embedding dimension: "
        f"{checkpoint_embed_dim}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HierarchicalGOClassifier(
        embed_dim=embed_dim,
        n_labels=n_labels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning FINAL held-out test inference...")

    all_probs = []
    all_targets = []
    all_protein_ids = []

    attention_records = []

    with torch.no_grad():

        for (
            protein_ids,
            embeddings,
            mask,
            targets,
        ) in test_loader:

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

            probs = torch.sigmoid(
                logits
            )

            all_probs.append(
                probs.cpu().numpy()
            )

            all_targets.append(
                targets.numpy()
            )

            all_protein_ids.extend(
                protein_ids
            )

            alpha = alpha.cpu().numpy()
            batch_mask = mask.cpu().numpy()

            for i, protein in enumerate(
                protein_ids
            ):
                valid_count = int(
                    batch_mask[i].sum()
                )

                attention_records.append(
                    {
                        "protein":
                            protein,
                        "weights":
                            alpha[
                                i,
                                :valid_count,
                            ].copy(),
                    }
                )

    y_prob = np.concatenate(
        all_probs,
        axis=0,
    )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    print(
        f"Prediction matrix: "
        f"{y_prob.shape}"
    )

    print(
        f"Target matrix:     "
        f"{y_true.shape}"
    )

    if y_prob.shape != (
        len(test_ids),
        n_labels,
    ):
        raise ValueError(
            "Unexpected prediction matrix shape."
        )

    # --------------------------------------------------------
    # Raw evaluation
    # --------------------------------------------------------

    print("\nCalculating raw test metrics...")

    raw_report = evaluate_matrix(
        y_true,
        y_prob,
        parent_child_pairs,
        args.validation_threshold,
    )

    # --------------------------------------------------------
    # Hierarchical correction
    # --------------------------------------------------------

    print(
        "Applying post-hoc hierarchy consistency..."
    )

    corrected_tensor = enforce_consistency(
        torch.from_numpy(y_prob),
        parent_child_pairs,
    )

    y_prob_corrected = (
        corrected_tensor.numpy()
    )
    corrected_report = evaluate_matrix(
        y_true,
        y_prob_corrected,
        parent_child_pairs,
        args.validation_threshold,
    )

    corrected_report = evaluate_matrix(
        y_true,
        y_prob_corrected,
        parent_child_pairs,
        args.validation_threshold,
    )

    # --------------------------------------------------------
    # Category metrics
    # --------------------------------------------------------

    category_reports = {}

    for category in ["BP", "MF", "CC"]:

        indices = category_indices[
            category
        ]

        if not indices:
            continue

        # Restrict hierarchy relationships to this namespace
        # and remap global indices to local matrix indices.
        global_to_local = {
            global_idx: local_idx
            for local_idx, global_idx
            in enumerate(indices)
        }

        category_pairs = []

        for child, parent in parent_child_pairs:

            if (
                child in global_to_local
                and parent in global_to_local
            ):
                category_pairs.append(
                    (
                        global_to_local[child],
                        global_to_local[parent],
                    )
                )

        category_true = y_true[
            :,
            indices,
        ]

        category_prob = y_prob[
            :,
            indices,
        ]

        category_prob_corrected = (
            y_prob_corrected[:, indices]
        )

        raw_category_report = evaluate_matrix(
            category_true,
            category_prob,
            category_pairs,
            args.validation_threshold,
        )

        corrected_category_report = evaluate_matrix(
            category_true,
            category_prob_corrected,
            category_pairs,
            args.validation_threshold,
        )

        category_reports[category] = {
            "raw": raw_category_report,
            "corrected": corrected_category_report,
        }

    # --------------------------------------------------------
    # Prediction statistics
    # --------------------------------------------------------

    operating_threshold = (
        args.validation_threshold
    )

    predicted_binary = (
        y_prob >= operating_threshold
    )

    average_true_labels = float(
        y_true.sum(axis=1).mean()
    )

    average_predicted_labels = float(
        predicted_binary.sum(axis=1).mean()
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 68)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 68)

    print(
        f"Test proteins:                         "
        f"{len(test_ids):,}"
    )

    print(
        f"GO labels:                             "
        f"{n_labels:,}"
    )

    print(
        f"Fmax:                                  "
        f"{raw_report['fmax']:.6f}"
    )

    print(
        f"Threshold maximizing test Fmax:        "
        f"{raw_report['test_fmax_threshold']:.2f}"
    )

    print(
        f"Validation-selected threshold:         "
        f"{operating_threshold:.2f}"
    )

    print(
        f"Precision @ validation threshold:      "
        f"{raw_report['precision_at_validation_threshold']:.6f}"
    )

    print(
        f"Recall @ validation threshold:         "
        f"{raw_report['recall_at_validation_threshold']:.6f}"
    )

    print(
        f"F1 @ validation threshold:             "
        f"{raw_report['f1_at_validation_threshold']:.6f}"
    )

    print(
        f"AUPR (micro):                          "
        f"{raw_report['aupr_micro']:.6f}"
    )

    print(
        f"AUPR (macro):                          "
        f"{raw_report['aupr_macro']:.6f}"
    )

    print(
        f"GO terms used for macro-AUPR:          "
        f"{raw_report['macro_evaluable_terms']:,}"
    )

    print(
        f"Hierarchy violation BEFORE correction: "
        f"{raw_report['hierarchy_violation_rate']:.6f}"
    )

    print(
        f"Hierarchy violation AFTER correction:  "
        f"{corrected_hvr:.6f}"
    )

    print(
        f"Average true labels / protein:          "
        f"{average_true_labels:.2f}"
    )

    print(
        f"Average predicted labels / protein:     "
        f"{average_predicted_labels:.2f}"
    )

    print("=" * 68)

    # --------------------------------------------------------
    # Category results
    # --------------------------------------------------------

    print("\nPER-GO-CATEGORY RESULTS")
    print("=" * 68)

    for category in ["BP", "MF", "CC"]:

        if category not in category_reports:
            continue

        report = category_reports[
            category
        ]

        print(
            f"\n{category}"
        )

        print(
            f"  Labels:       "
            f"{len(category_indices[category]):,}"
        )

        print(
            f"  Fmax:         "
            f"{report['fmax']:.6f}"
        )

        print(
            f"  micro-AUPR:   "
            f"{report['aupr_micro']:.6f}"
        )

        print(
            f"  macro-AUPR:   "
            f"{report['aupr_macro']:.6f}"
        )

        print(
            f"  macro labels: "
            f"{report['macro_evaluable_terms']:,}"
        )

        print(
            f"  HVR:          "
            f"{report['hierarchy_violation_rate']:.6f}"
        )

    # --------------------------------------------------------
    # Qualitative attention examples
    # --------------------------------------------------------

    print("\n")
    print("=" * 68)
    print("QUALITATIVE ATTENTION EXAMPLES")
    print("=" * 68)

    n_examples = min(
        args.qualitative_examples,
        len(all_protein_ids),
    )

    for example_idx in range(
        n_examples
    ):

        protein = all_protein_ids[
            example_idx
        ]

        probs = y_prob[
            example_idx
        ]

        true_count = int(
            y_true[
                example_idx
            ].sum()
        )

        top_prediction_indices = (
            np.argsort(probs)[-10:][::-1]
        )

        print(
            f"\nProtein: {protein}"
        )

        print(
            f"True GO labels: {true_count}"
        )

        print("\nTop predicted GO terms:")

        for idx in top_prediction_indices:

            print(
                f"  {go_terms[idx]}  "
                f"{probs[idx]:.4f}"
            )

        # Dataset retains the same PMID ordering supplied
        # by protein_to_pmids after unavailable embeddings
        # are removed.
        valid_pmids = [
            pmid
            for pmid in protein_to_pmids[
                protein
            ]
            if pmid in pmid_to_vec
        ]

        weights = attention_records[
            example_idx
        ]["weights"]

        top_attention = np.argsort(
            weights
        )[::-1][:10]

        print("\nTop-attended PMIDs:")

        for idx in top_attention:

            if idx >= len(valid_pmids):
                continue

            print(
                f"  PMID {valid_pmids[idx]}  "
                f"alpha={weights[idx]:.4f}"
            )

        print("-" * 68)

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------

    final_results = {
        "model":
            "Hierarchical Attention Classifier",
        "checkpoint":
            str(args.checkpoint),
        "split":
            "held-out test",
        "n_test_proteins":
            len(test_ids),
        "n_go_labels":
            n_labels,
        "overall":
            raw_report,
        "hierarchy_violation_rate_after":
            float(corrected_hvr),
        "average_true_labels_per_protein":
            average_true_labels,
        "average_predicted_labels_per_protein":
            average_predicted_labels,
        "categories":
            category_reports,
    }

    json_path = (
        output_dir
        / "hierarchical_attention_test_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save CSV suitable for Chapter 4
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / "metrics.csv"
    )

    rows = []

    rows.append(
        {
            "model":
                "Hierarchical Attention Classifier",
            "category":
                "Overall",
            "fmax":
                raw_report["fmax"],
            "aupr_micro":
                raw_report["aupr_micro"],
            "aupr_macro":
                raw_report["aupr_macro"],
            "hierarchy_violation_before":
                raw_report[
                    "hierarchy_violation_rate"
                ],
            "hierarchy_violation_after":
                corrected_hvr,
        }
    )

    for category, report in (
        category_reports.items()
    ):

        rows.append(
            {
                "model":
                    "Hierarchical Attention Classifier",
                "category":
                    category,
                "fmax":
                    report["fmax"],
                "aupr_micro":
                    report["aupr_micro"],
                "aupr_macro":
                    report["aupr_macro"],
                "hierarchy_violation_before":
                    report[
                        "hierarchy_violation_rate"
                    ],
                "hierarchy_violation_after":
                    "",
            }
        )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    print("\nPhase 7A evaluation complete.")


if __name__ == "__main__":
    main()