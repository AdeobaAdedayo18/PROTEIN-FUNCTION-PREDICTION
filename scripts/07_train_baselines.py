"""Phase 7B: Train and evaluate conventional tree baselines.

Baselines:
    - XGBoost
    - LightGBM
    - CatBoost
    - Random Forest

All baselines use the same underlying PMID literature embeddings as the
proposed model. Variable-length PMID sets are converted to fixed-length
protein representations using simple mean pooling.

The exact saved Phase 6 protein split and propagated GO vocabulary are
reconstructed. Every output prediction matrix retains the complete GO
vocabulary, including GO terms with no positive training examples.

Training is performed one GO term at a time. Progress is periodically
checkpointed so interrupted Colab runs can resume without restarting.

The held-out test split is used only for final prediction/evaluation,
never for fitting or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ldpfp.evaluate import (
    aupr_scores,
    fmax_score,
    hierarchy_violation_rate,
)
from ldpfp.go_hierarchy import (
    build_parent_child_pairs,
    load_go_graph,
    propagate_labels,
)
from ldpfp.models.baselines import (
    BASELINE_NAMES,
    build_baseline,
)
from ldpfp.models.hierarchical_classifier import (
    enforce_consistency,
)

import torch


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
        "--splits",
        default="data/splits/protein_splits.json",
    )

    parser.add_argument(
        "--output-dir",
        default="results/baselines",
    )

    parser.add_argument(
        "--model",
        choices=BASELINE_NAMES,
        required=True,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help=(
            "Save progress after this many newly processed GO terms."
        ),
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Train only a representative subset of GO terms and "
            "estimate full-run time."
        ),
    )

    parser.add_argument(
        "--benchmark-per-bin",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Data loading
# ============================================================


def load_embeddings(path):
    print("\nLoading PMID embeddings...")

    data = np.load(path)

    pmids = data["pmids"]
    vectors = data["vectors"].astype(
        np.float32,
        copy=False,
    )

    pmid_to_vec = {
        str(pmid): vector
        for pmid, vector in zip(
            pmids,
            vectors,
        )
    }

    print(
        f"PMID embeddings: {len(pmid_to_vec):,}"
    )
    print(
        f"Embedding dimension: {vectors.shape[1]}"
    )

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
    annotation_categories = {}

    has_category = "GO_Category" in df.columns

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

        if has_category:
            category = row.get("GO_Category")

            if category is not None:
                annotation_categories[go_id] = str(
                    category
                )

    print(
        f"Proteins with GO annotations: "
        f"{len(protein_to_go):,}"
    )

    return (
        protein_to_go,
        annotation_categories,
    )


# ============================================================
# GO labels
# ============================================================


def build_label_vocabulary(
    protein_to_go,
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

        propagated_by_protein[protein] = (
            propagated
        )

        all_terms.update(propagated)

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


def build_label_matrix(
    protein_ids,
    propagated_by_protein,
    term_to_idx,
):
    matrix = np.zeros(
        (
            len(protein_ids),
            len(term_to_idx),
        ),
        dtype=np.uint8,
    )

    for row_idx, protein in enumerate(
        protein_ids
    ):
        labels = propagated_by_protein.get(
            protein,
            (),
        )

        for go_id in labels:
            label_idx = term_to_idx.get(go_id)

            if label_idx is not None:
                matrix[
                    row_idx,
                    label_idx,
                ] = 1

    return matrix


# ============================================================
# Mean-pooled protein representations
# ============================================================


def mean_pool_protein(
    protein,
    protein_to_pmids,
    pmid_to_vec,
    embed_dim,
):
    vectors = []

    for pmid in protein_to_pmids.get(
        protein,
        (),
    ):
        vector = pmid_to_vec.get(
            str(pmid)
        )

        if vector is not None:
            vectors.append(vector)

    if not vectors:
        raise ValueError(
            f"Protein {protein} has no available "
            "PMID embeddings."
        )

    stacked = np.stack(
        vectors,
        axis=0,
    )

    return stacked.mean(
        axis=0,
        dtype=np.float32,
    )


def build_feature_matrix(
    protein_ids,
    protein_to_pmids,
    pmid_to_vec,
    embed_dim,
):
    print(
        f"\nMean-pooling {len(protein_ids):,} "
        "proteins..."
    )

    X = np.empty(
        (
            len(protein_ids),
            embed_dim,
        ),
        dtype=np.float32,
    )

    for i, protein in enumerate(
        protein_ids
    ):
        X[i] = mean_pool_protein(
            protein,
            protein_to_pmids,
            pmid_to_vec,
            embed_dim,
        )

        if (
            (i + 1) % 1000 == 0
            or i + 1 == len(protein_ids)
        ):
            print(
                f"  {i + 1:,}/"
                f"{len(protein_ids):,}"
            )

    return X


# ============================================================
# Checkpointing
# ============================================================


def checkpoint_paths(
    output_dir,
    model_name,
):
    base = Path(output_dir)

    return (
        base / f"{model_name}_progress.npz",
        base / f"{model_name}_progress.json",
    )


def save_progress(
    output_dir,
    model_name,
    predictions,
    completed,
    go_terms,
    test_ids,
):
    npz_path, json_path = checkpoint_paths(
        output_dir,
        model_name,
    )

    np.savez_compressed(
        npz_path,
        predictions=predictions,
        completed=completed,
        go_terms=np.asarray(
            go_terms,
            dtype=str,
        ),
        protein_ids=np.asarray(
            test_ids,
            dtype=str,
        ),
    )

    metadata = {
        "model": model_name,
        "n_labels": len(go_terms),
        "n_test_proteins": len(test_ids),
        "completed_labels": int(
            completed.sum()
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"\nCheckpoint saved: "
        f"{int(completed.sum()):,}/"
        f"{len(go_terms):,} labels"
    )


def load_progress(
    output_dir,
    model_name,
    go_terms,
    test_ids,
):
    npz_path, _ = checkpoint_paths(
        output_dir,
        model_name,
    )

    if not npz_path.exists():
        return None

    print(
        f"\nLoading baseline checkpoint → "
        f"{npz_path}"
    )

    data = np.load(
        npz_path,
        allow_pickle=False,
    )

    predictions = data["predictions"]
    completed = data["completed"].astype(bool)

    cached_go_terms = [
        str(x)
        for x in data["go_terms"].tolist()
    ]

    cached_ids = [
        str(x)
        for x in data["protein_ids"].tolist()
    ]

    if cached_go_terms != list(go_terms):
        raise ValueError(
            "Baseline checkpoint GO vocabulary "
            "does not match current vocabulary."
        )

    if cached_ids != list(test_ids):
        raise ValueError(
            "Baseline checkpoint test protein "
            "ordering does not match current split."
        )

    expected_shape = (
        len(test_ids),
        len(go_terms),
    )

    if predictions.shape != expected_shape:
        raise ValueError(
            "Checkpoint prediction shape mismatch. "
            f"Expected {expected_shape}, "
            f"got {predictions.shape}."
        )

    if completed.shape != (
        len(go_terms),
    ):
        raise ValueError(
            "Checkpoint completion mask has "
            "unexpected shape."
        )

    print(
        f"Resuming from "
        f"{int(completed.sum()):,}/"
        f"{len(go_terms):,} completed labels."
    )

    return predictions, completed


# ============================================================
# Benchmark selection
# ============================================================


def select_benchmark_labels(
    positive_counts,
    per_bin,
    seed,
):
    rng = np.random.default_rng(seed)

    bins = [
        ("1", positive_counts == 1),
        (
            "2-5",
            (positive_counts >= 2)
            & (positive_counts <= 5),
        ),
        (
            "6-10",
            (positive_counts >= 6)
            & (positive_counts <= 10),
        ),
        (
            "11-50",
            (positive_counts >= 11)
            & (positive_counts <= 50),
        ),
        (
            "51-100",
            (positive_counts >= 51)
            & (positive_counts <= 100),
        ),
        (
            ">100",
            positive_counts > 100,
        ),
    ]

    selected = []

    print("\nBenchmark label selection:")

    for name, mask in bins:
        candidates = np.flatnonzero(mask)

        count = min(
            per_bin,
            len(candidates),
        )

        if count == 0:
            continue

        chosen = rng.choice(
            candidates,
            size=count,
            replace=False,
        )

        selected.extend(
            int(x)
            for x in chosen
        )

        print(
            f"  {name:<8}: {count}"
        )

    return sorted(set(selected))


# ============================================================
# Binary model helper
# ============================================================


def positive_probability(
    model,
    X,
):
    proba = model.predict_proba(X)

    classes = np.asarray(
        model.classes_
    )

    positive_positions = np.flatnonzero(
        classes == 1
    )

    if len(positive_positions) == 0:
        return np.zeros(
            len(X),
            dtype=np.float32,
        )

    positive_index = int(
        positive_positions[0]
    )

    return np.asarray(
        proba[:, positive_index],
        dtype=np.float32,
    )


# ============================================================
# Category helpers
# ============================================================


def normalize_category(category):
    if category is None:
        return None

    category = str(
        category
    ).lower().strip()

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
    categories = {}

    namespace_map = {
        "biological_process": "BP",
        "molecular_function": "MF",
        "cellular_component": "CC",
    }

    for go_id in go_terms:
        direct = normalize_category(
            annotation_categories.get(
                go_id
            )
        )

        if direct is not None:
            categories[go_id] = direct
            continue

        if go_id in graph:
            namespace = graph.nodes[
                go_id
            ].get("namespace")

            categories[go_id] = (
                namespace_map.get(namespace)
            )

    return categories


# ============================================================
# Evaluation
# ============================================================


def evaluate_predictions(
    y_true,
    y_prob,
    parent_child_pairs,
):
    fmax, threshold = fmax_score(
        y_true,
        y_prob,
    )

    aupr = aupr_scores(
        y_true,
        y_prob,
    )

    hvr = hierarchy_violation_rate(
        y_prob,
        parent_child_pairs,
    )

    return {
        "fmax": float(fmax),
        "best_threshold": float(threshold),
        "aupr_micro": float(
            aupr["aupr_micro"]
        ),
        "aupr_macro": float(
            aupr["aupr_macro"]
        ),
        "macro_evaluable_terms": int(
            aupr.get(
                "macro_evaluable_terms",
                aupr.get(
                    "macro_labels_evaluated",
                    0,
                ),
            )
        ),
        "macro_total_terms": int(
            aupr.get(
                "macro_total_terms",
                y_true.shape[1],
            )
        ),
        "hierarchy_violation_rate": float(
            hvr
        ),
    }


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    for path in (
        args.embeddings,
        args.pmid_map,
        args.annotations,
        args.go_obo,
        args.splits,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 68)
    print("PHASE 7B — BASELINE TRAINING")
    print("=" * 68)

    print(
        f"Model: {args.model}"
    )

    # --------------------------------------------------------
    # Source data
    # --------------------------------------------------------

    pmid_to_vec = load_embeddings(
        args.embeddings
    )

    embed_dim = len(
        next(iter(pmid_to_vec.values()))
    )

    protein_to_pmids = (
        load_protein_pmid_map(
            args.pmid_map
        )
    )

    (
        protein_to_go,
        annotation_categories,
    ) = load_annotations(
        args.annotations
    )

    print("\nLoading GO DAG...")

    graph = load_go_graph(
        args.go_obo
    )

    print(
        f"GO DAG terms: {len(graph):,}"
    )

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

    # --------------------------------------------------------
    # Exact saved split
    # --------------------------------------------------------

    print("\nLoading saved dataset splits...")

    with open(
        args.splits,
        "r",
        encoding="utf-8",
    ) as f:
        split_data = json.load(f)

    train_ids = (
        split_data.get("train")
        or split_data.get("train_ids")
        or split_data.get(
            "train_protein_ids"
        )
    )

    val_ids = (
        split_data.get("validation")
        or split_data.get("val")
        or split_data.get("val_ids")
        or split_data.get(
            "validation_ids"
        )
    )

    test_ids = (
        split_data.get("test")
        or split_data.get("test_ids")
        or split_data.get(
            "test_protein_ids"
        )
    )

    if (
        train_ids is None
        or val_ids is None
        or test_ids is None
    ):
        raise KeyError(
            "Could not locate train/validation/"
            "test IDs in split JSON. "
            f"Keys: {list(split_data.keys())}"
        )

    train_ids = [
        str(x)
        for x in train_ids
    ]

    val_ids = [
        str(x)
        for x in val_ids
    ]

    test_ids = [
        str(x)
        for x in test_ids
    ]

    print(
        f"Train:      {len(train_ids):,}"
    )
    print(
        f"Validation: {len(val_ids):,}"
    )
    print(
        f"Test:       {len(test_ids):,}"
    )

    if len(set(train_ids)) != len(
        train_ids
    ):
        raise ValueError(
            "Duplicate proteins in training split."
        )

    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) & set(test_ids)
        or set(val_ids) & set(test_ids)
    ):
        raise ValueError(
            "Protein split leakage detected."
        )

    # --------------------------------------------------------
    # Feature matrices
    # --------------------------------------------------------

    X_train = build_feature_matrix(
        train_ids,
        protein_to_pmids,
        pmid_to_vec,
        embed_dim,
    )

    X_test = build_feature_matrix(
        test_ids,
        protein_to_pmids,
        pmid_to_vec,
        embed_dim,
    )

    print(
        f"\nX_train: {X_train.shape}"
    )
    print(
        f"X_test:  {X_test.shape}"
    )

    # --------------------------------------------------------
    # Label matrices
    # --------------------------------------------------------

    print("\nBuilding label matrices...")

    Y_train = build_label_matrix(
        train_ids,
        propagated_by_protein,
        term_to_idx,
    )

    Y_test = build_label_matrix(
        test_ids,
        propagated_by_protein,
        term_to_idx,
    )

    print(
        f"Y_train: {Y_train.shape}"
    )
    print(
        f"Y_test:  {Y_test.shape}"
    )

    positive_counts = (
        Y_train.sum(axis=0)
    ).astype(np.int64)

    zero_positive = int(
        (positive_counts == 0).sum()
    )

    trainable = int(
        (positive_counts > 0).sum()
    )

    print("\nTraining label coverage:")
    print(
        f"  Trainable GO terms:     "
        f"{trainable:,}"
    )
    print(
        f"  Zero-positive GO terms: "
        f"{zero_positive:,}"
    )

    # --------------------------------------------------------
    # Benchmark mode
    # --------------------------------------------------------

    if args.benchmark:
        label_indices = (
            select_benchmark_labels(
                positive_counts,
                args.benchmark_per_bin,
                args.seed,
            )
        )

        print(
            f"\nBenchmarking "
            f"{len(label_indices):,} labels..."
        )

        times = []

        for number, label_idx in enumerate(
            label_indices,
            start=1,
        ):
            y_train = Y_train[
                :,
                label_idx,
            ]

            model = build_baseline(
                args.model,
                n_jobs=args.n_jobs,
                random_state=args.seed,
            )

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            _ = positive_probability(
                model,
                X_test,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            times.append(elapsed)

            print(
                f"[{number:>3}/"
                f"{len(label_indices)}] "
                f"{go_terms[label_idx]} | "
                f"positives="
                f"{int(positive_counts[label_idx]):,} | "
                f"{elapsed:.2f}s"
            )

            del model

        mean_seconds = float(
            np.mean(times)
        )

        median_seconds = float(
            np.median(times)
        )

        estimated_seconds = (
            mean_seconds
            * trainable
        )

        print("\n" + "=" * 68)
        print("BENCHMARK SUMMARY")
        print("=" * 68)

        print(
            f"Model:                  "
            f"{args.model}"
        )

        print(
            f"Labels benchmarked:     "
            f"{len(times):,}"
        )

        print(
            f"Mean seconds / label:   "
            f"{mean_seconds:.2f}"
        )

        print(
            f"Median seconds / label: "
            f"{median_seconds:.2f}"
        )

        print(
            f"Trainable labels:       "
            f"{trainable:,}"
        )

        print(
            f"Naive full-run estimate:"
            f" {estimated_seconds / 3600:.2f} hours"
        )

        print("=" * 68)

        return

    # --------------------------------------------------------
    # Initialize / resume prediction matrix
    # --------------------------------------------------------

    progress = load_progress(
        output_dir,
        args.model,
        go_terms,
        test_ids,
    )

    if progress is None:
        predictions = np.zeros(
            (
                len(test_ids),
                n_labels,
            ),
            dtype=np.float32,
        )

        completed = np.zeros(
            n_labels,
            dtype=bool,
        )

    else:
        predictions, completed = progress

    # --------------------------------------------------------
    # Train one GO classifier at a time
    # --------------------------------------------------------

    print("\nTraining baseline classifiers...")
    print(
        "Zero-positive training labels are "
        "assigned probability 0."
    )

    newly_processed = 0
    start_time = time.perf_counter()

    for label_idx in range(n_labels):

        if completed[label_idx]:
            continue

        positive_count = int(
            positive_counts[label_idx]
        )

        # No positive evidence in training.
        if positive_count == 0:
            predictions[
                :,
                label_idx,
            ] = 0.0

            completed[label_idx] = True
            newly_processed += 1

        else:
            y_train = Y_train[
                :,
                label_idx,
            ]

            # Defensive check: although impossible here
            # unless every training protein has the term.
            unique_classes = np.unique(
                y_train
            )

            if len(unique_classes) == 1:
                constant_value = float(
                    unique_classes[0]
                )

                predictions[
                    :,
                    label_idx,
                ] = constant_value

                completed[
                    label_idx
                ] = True

                newly_processed += 1

            else:
                model = build_baseline(
                    args.model,
                    n_jobs=args.n_jobs,
                    random_state=args.seed,
                )

                model.fit(
                    X_train,
                    y_train,
                )

                predictions[
                    :,
                    label_idx,
                ] = positive_probability(
                    model,
                    X_test,
                )

                completed[
                    label_idx
                ] = True

                newly_processed += 1

                del model

        completed_count = int(
            completed.sum()
        )

        if (
            newly_processed
            % args.checkpoint_every
            == 0
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"\nProgress: "
                f"{completed_count:,}/"
                f"{n_labels:,} "
                f"({100 * completed_count / n_labels:.2f}%)"
            )

            print(
                f"Elapsed this run: "
                f"{elapsed / 60:.1f} min"
            )

            save_progress(
                output_dir,
                args.model,
                predictions,
                completed,
                go_terms,
                test_ids,
            )

    # Always save final state.
    save_progress(
        output_dir,
        args.model,
        predictions,
        completed,
        go_terms,
        test_ids,
    )

    if not completed.all():
        raise RuntimeError(
            "Baseline training ended before all "
            "GO labels were processed."
        )

    # --------------------------------------------------------
    # Save final predictions
    # --------------------------------------------------------

    final_prediction_path = (
        output_dir
        / f"{args.model}_test_predictions.npz"
    )

    np.savez_compressed(
        final_prediction_path,
        y_true=Y_test.astype(
            np.float32
        ),
        y_prob=predictions.astype(
            np.float32
        ),
        protein_ids=np.asarray(
            test_ids,
            dtype=str,
        ),
        go_terms=np.asarray(
            go_terms,
            dtype=str,
        ),
        representation=np.asarray(
            "mean_pooled_pmid_embeddings"
        ),
    )

    print(
        f"\nFinal predictions saved → "
        f"{final_prediction_path}"
    )

    # --------------------------------------------------------
    # Raw evaluation
    # --------------------------------------------------------

    print("\nEvaluating raw predictions...")

    raw_report = evaluate_predictions(
        Y_test,
        predictions,
        parent_child_pairs,
    )

    # --------------------------------------------------------
    # Hierarchical correction
    # --------------------------------------------------------

    print(
        "Applying post-hoc GO consistency..."
    )

    corrected = enforce_consistency(
        torch.from_numpy(
            predictions
        ),
        parent_child_pairs,
    ).numpy()

    corrected_report = (
        evaluate_predictions(
            Y_test,
            corrected,
            parent_child_pairs,
        )
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
            for i, go_id in enumerate(
                go_terms
            )
            if go_categories.get(go_id)
            == category
        ]
        for category in (
            "BP",
            "MF",
            "CC",
        )
    }

    category_reports = {}

    for category, indices in (
        category_indices.items()
    ):
        if not indices:
            continue

        global_to_local = {
            global_idx: local_idx
            for local_idx, global_idx
            in enumerate(indices)
        }

        local_pairs = []

        for child, parent in (
            parent_child_pairs
        ):
            if (
                child in global_to_local
                and parent in global_to_local
            ):
                local_pairs.append(
                    (
                        global_to_local[
                            child
                        ],
                        global_to_local[
                            parent
                        ],
                    )
                )

        category_true = Y_test[
            :,
            indices,
        ]

        category_raw = predictions[
            :,
            indices,
        ]

        category_corrected = corrected[
            :,
            indices,
        ]

        category_reports[
            category
        ] = {
            "raw": evaluate_predictions(
                category_true,
                category_raw,
                local_pairs,
            ),
            "corrected":
                evaluate_predictions(
                    category_true,
                    category_corrected,
                    local_pairs,
                ),
            "n_labels": len(indices),
        }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 68)
    print(
        f"{args.model.upper()} HELD-OUT TEST RESULTS"
    )
    print("=" * 68)

    print(
        f"Fmax:                    "
        f"{raw_report['fmax']:.6f}"
    )

    print(
        f"Best threshold:           "
        f"{raw_report['best_threshold']:.2f}"
    )

    print(
        f"AUPR (micro):            "
        f"{raw_report['aupr_micro']:.6f}"
    )

    print(
        f"AUPR (macro):            "
        f"{raw_report['aupr_macro']:.6f}"
    )

    print(
        f"Macro evaluable terms:   "
        f"{raw_report['macro_evaluable_terms']:,}"
    )

    print(
        f"HVR before correction:   "
        f"{raw_report['hierarchy_violation_rate']:.6f}"
    )

    print(
        f"HVR after correction:    "
        f"{corrected_report['hierarchy_violation_rate']:.6f}"
    )

    print("=" * 68)

    print("\nPER-GO-CATEGORY RESULTS")

    for category in (
        "BP",
        "MF",
        "CC",
    ):
        if category not in category_reports:
            continue

        reports = category_reports[
            category
        ]

        raw = reports["raw"]
        post = reports["corrected"]

        print(f"\n{category}")
        print(
            f"  Labels:       "
            f"{reports['n_labels']:,}"
        )
        print(
            f"  Raw Fmax:     "
            f"{raw['fmax']:.6f}"
        )
        print(
            f"  Raw micro:    "
            f"{raw['aupr_micro']:.6f}"
        )
        print(
            f"  Raw macro:    "
            f"{raw['aupr_macro']:.6f}"
        )
        print(
            f"  HVR before:   "
            f"{raw['hierarchy_violation_rate']:.6f}"
        )
        print(
            f"  HVR after:    "
            f"{post['hierarchy_violation_rate']:.6f}"
        )

    # --------------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------------

    results = {
        "model": args.model,
        "representation":
            "mean_pooled_pmid_embeddings",
        "n_train_proteins":
            len(train_ids),
        "n_test_proteins":
            len(test_ids),
        "n_go_labels":
            n_labels,
        "n_trainable_labels":
            trainable,
        "n_zero_positive_train_labels":
            zero_positive,
        "overall": {
            "raw": raw_report,
            "corrected": corrected_report,
        },
        "categories":
            category_reports,
    }

    json_path = (
        output_dir
        / f"{args.model}_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save compact CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / f"{args.model}_metrics.csv"
    )

    rows = [
        {
            "model": args.model,
            "category": "Overall",
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
                corrected_report[
                    "hierarchy_violation_rate"
                ],
        }
    ]

    for category, reports in (
        category_reports.items()
    ):
        rows.append(
            {
                "model": args.model,
                "category": category,
                "fmax":
                    reports["raw"]["fmax"],
                "aupr_micro":
                    reports[
                        "raw"
                    ]["aupr_micro"],
                "aupr_macro":
                    reports[
                        "raw"
                    ]["aupr_macro"],
                "hierarchy_violation_before":
                    reports[
                        "raw"
                    ][
                        "hierarchy_violation_rate"
                    ],
                "hierarchy_violation_after":
                    reports[
                        "corrected"
                    ][
                        "hierarchy_violation_rate"
                    ],
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
    print(
        f"  {final_prediction_path}"
    )
    print(
        f"  {json_path}"
    )
    print(
        f"  {csv_path}"
    )

    print(
        f"\nPhase 7B {args.model} complete."
    )


if __name__ == "__main__":
    main()