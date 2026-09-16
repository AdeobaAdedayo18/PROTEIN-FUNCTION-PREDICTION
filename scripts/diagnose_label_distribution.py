"""Diagnose GO-label distribution across the saved train/validation/test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import polars as pl

from ldpfp.go_hierarchy import load_go_graph, propagate_labels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations.parquet",
    )
    parser.add_argument(
        "--go-obo",
        required=True,
        help="Path to go-basic.obo",
    )
    parser.add_argument(
        "--splits",
        default="data/splits/protein_splits.json",
        help="Path to saved protein split JSON",
    )

    return parser.parse_args()


def build_protein_labels(annotations_path, graph):
    df = pl.read_parquet(annotations_path)

    protein_to_direct = {}

    for row in df.select(["Protein_ID", "GO_Label"]).iter_rows():
        protein_id, go_id = row

        protein_to_direct.setdefault(
            str(protein_id), set()
        ).add(str(go_id))

    protein_to_propagated = {}

    for protein_id, labels in protein_to_direct.items():
        protein_to_propagated[protein_id] = propagate_labels(
            labels,
            graph,
        )

    return protein_to_direct, protein_to_propagated


def get_frequencies(protein_ids, protein_to_labels):
    counter = Counter()

    proteins_found = 0

    for protein_id in protein_ids:
        labels = protein_to_labels.get(protein_id)

        if labels is None:
            continue

        proteins_found += 1
        counter.update(labels)

    return counter, proteins_found


def print_frequency_buckets(counter):
    values = list(counter.values())

    buckets = {
        "1 positive": 0,
        "2-5 positives": 0,
        "6-10 positives": 0,
        "11-50 positives": 0,
        "51-100 positives": 0,
        ">100 positives": 0,
    }

    for count in values:
        if count == 1:
            buckets["1 positive"] += 1
        elif count <= 5:
            buckets["2-5 positives"] += 1
        elif count <= 10:
            buckets["6-10 positives"] += 1
        elif count <= 50:
            buckets["11-50 positives"] += 1
        elif count <= 100:
            buckets["51-100 positives"] += 1
        else:
            buckets[">100 positives"] += 1

    for name, count in buckets.items():
        print(f"{name:<20} {count:>7,}")


def main():
    args = parse_args()

    print("Loading saved dataset splits...")

    with open(args.splits) as f:
        splits = json.load(f)

    # Support either common naming convention.
    train_ids = set(
        map(
            str,
            splits.get(
                "train",
                splits.get("train_protein_ids", []),
            ),
        )
    )

    val_ids = set(
        map(
            str,
            splits.get(
                "validation",
                splits.get(
                    "val",
                    splits.get("val_protein_ids", []),
                ),
            ),
        )
    )

    test_ids = set(
        map(
            str,
            splits.get(
                "test",
                splits.get("test_protein_ids", []),
            ),
        )
    )

    if not train_ids or not val_ids or not test_ids:
        raise ValueError(
            "Could not find train/validation/test protein IDs "
            f"in {args.splits}. Keys found: {list(splits.keys())}"
        )

    print(f"Train proteins:      {len(train_ids):,}")
    print(f"Validation proteins: {len(val_ids):,}")
    print(f"Test proteins:       {len(test_ids):,}")

    print("\nLoading GO DAG...")
    graph = load_go_graph(args.go_obo)

    print(f"GO DAG terms: {len(graph):,}")

    print("\nBuilding direct and propagated labels...")

    direct_labels, propagated_labels = build_protein_labels(
        args.annotations,
        graph,
    )

    train_freq, train_found = get_frequencies(
        train_ids,
        propagated_labels,
    )

    val_freq, val_found = get_frequencies(
        val_ids,
        propagated_labels,
    )

    test_freq, test_found = get_frequencies(
        test_ids,
        propagated_labels,
    )

    train_terms = set(train_freq)
    val_terms = set(val_freq)
    test_terms = set(test_freq)

    val_unseen = val_terms - train_terms
    test_unseen = test_terms - train_terms

    print("\n" + "=" * 60)
    print("SPLIT LABEL COVERAGE")
    print("=" * 60)

    print(f"Training proteins found:       {train_found:,}")
    print(f"Validation proteins found:     {val_found:,}")
    print(f"Test proteins found:           {test_found:,}")

    print()

    print(
        f"Training GO terms with positives:   "
        f"{len(train_terms):,}"
    )

    print(
        f"Validation GO terms with positives: "
        f"{len(val_terms):,}"
    )

    print(
        f"Test GO terms with positives:       "
        f"{len(test_terms):,}"
    )

    print()

    print(
        f"Validation terms unseen in train:   "
        f"{len(val_unseen):,}"
    )

    print(
        f"Test terms unseen in train:         "
        f"{len(test_unseen):,}"
    )

    if val_terms:
        print(
            f"Validation label coverage:          "
            f"{100 * len(val_terms & train_terms) / len(val_terms):.2f}%"
        )

    if test_terms:
        print(
            f"Test label coverage:                "
            f"{100 * len(test_terms & train_terms) / len(test_terms):.2f}%"
        )

    print("\n" + "=" * 60)
    print("TRAINING LABEL FREQUENCY")
    print("=" * 60)

    print_frequency_buckets(train_freq)

    # ---------------------------------------------------------
    # Most common propagated terms
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("MOST COMMON TRAINING GO TERMS")
    print("=" * 60)

    for go_id, count in train_freq.most_common(20):
        percentage = (
            100 * count / train_found
            if train_found
            else 0
        )

        print(
            f"{go_id:<15} "
            f"{count:>7,} proteins "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Direct vs propagated labels
    # ---------------------------------------------------------

    train_direct_freq, _ = get_frequencies(
        train_ids,
        direct_labels,
    )

    print("\n" + "=" * 60)
    print("DIRECT VS PROPAGATED LABELS")
    print("=" * 60)

    direct_total = sum(train_direct_freq.values())
    propagated_total = sum(train_freq.values())

    print(
        f"Direct positive assignments:      "
        f"{direct_total:,}"
    )

    print(
        f"Propagated positive assignments:  "
        f"{propagated_total:,}"
    )

    if direct_total:
        print(
            f"Propagation expansion factor:     "
            f"{propagated_total / direct_total:.2f}x"
        )

    # ---------------------------------------------------------
    # Unseen labels
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("UNSEEN LABEL EXAMPLES")
    print("=" * 60)

    print("\nValidation-only GO terms:")
    print(sorted(val_unseen)[:20])

    print("\nTest-only GO terms:")
    print(sorted(test_unseen)[:20])

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()