"""Generate PubMedBERT embeddings with resumable checkpoints."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ldpfp.embeddings import BiomedicalEmbedder
from ldpfp.text_preprocessing import assemble_document


def get_pmids(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)

    rows = conn.execute(
        """
        SELECT pmid
        FROM literature
        WHERE status = 'ok'
        ORDER BY pmid
        """
    ).fetchall()

    conn.close()

    return [row[0] for row in rows]


def load_document(
    conn: sqlite3.Connection,
    pmid: str,
) -> str | None:

    row = conn.execute(
        """
        SELECT title, abstract, full_text
        FROM literature
        WHERE pmid = ?
          AND status = 'ok'
        """,
        (pmid,),
    ).fetchone()

    if row is None:
        return None

    title, abstract, full_text = row

    document = assemble_document(
        title,
        abstract,
        full_text,
    )

    return document or None


def load_completed_pmids(
    checkpoint_dir: Path,
) -> set[str]:

    completed = set()

    for path in checkpoint_dir.glob(
        "checkpoint_*.npz"
    ):
        try:
            data = np.load(path)

            completed.update(
                str(pmid)
                for pmid in data["pmids"]
            )

        except Exception as exc:
            print(
                f"Warning: could not read "
                f"{path}: {exc}"
            )

    return completed


def save_checkpoint(
    checkpoint_dir: Path,
    checkpoint_number: int,
    pmids: list[str],
    vectors: list[np.ndarray],
) -> None:

    if not pmids:
        return

    path = checkpoint_dir / (
        f"checkpoint_{checkpoint_number:05d}.npz"
    )

    np.savez_compressed(
        path,
        pmids=np.array(pmids),
        vectors=np.stack(vectors),
    )

    print(
        f"\nSaved {len(pmids):,} embeddings "
        f"→ {path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default="data/literature_cache.sqlite",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/embeddings",
    )

    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.output_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_pmids = get_pmids(args.db)

    completed = load_completed_pmids(
        checkpoint_dir
    )

    remaining = [
        pmid
        for pmid in all_pmids
        if pmid not in completed
    ]

    if args.limit is not None:
        remaining = remaining[:args.limit]

    print(
        f"Total PMIDs: {len(all_pmids):,}"
    )

    print(
        f"Already embedded: {len(completed):,}"
    )

    print(
        f"Remaining this run: {len(remaining):,}"
    )

    if not remaining:
        print("Nothing to do.")
        return

    embedder = BiomedicalEmbedder(
        batch_size=args.batch_size
    )

    conn = sqlite3.connect(args.db)

    checkpoint_number = (
        len(
            list(
                checkpoint_dir.glob(
                    "checkpoint_*.npz"
                )
            )
        )
        + 1
    )

    checkpoint_pmids = []
    checkpoint_vectors = []

    try:
        for pmid in tqdm(
            remaining,
            desc="Embedding documents",
        ):
            document = load_document(
                conn,
                pmid,
            )

            if not document:
                continue

            try:
                vector = embedder.embed_text(
                    document
                )

            except RuntimeError as exc:
                # Helpful when GPU memory is exhausted.
                if "out of memory" in str(exc).lower():
                    print(
                        "\nGPU out of memory. "
                        "Try a smaller --batch-size."
                    )

                raise

            checkpoint_pmids.append(
                pmid
            )

            checkpoint_vectors.append(
                vector
            )

            if (
                len(checkpoint_pmids)
                >= args.checkpoint_size
            ):
                save_checkpoint(
                    checkpoint_dir,
                    checkpoint_number,
                    checkpoint_pmids,
                    checkpoint_vectors,
                )

                checkpoint_number += 1

                checkpoint_pmids = []
                checkpoint_vectors = []

    finally:
        # Even Ctrl+C should preserve the current partial batch.
        if checkpoint_pmids:
            save_checkpoint(
                checkpoint_dir,
                checkpoint_number,
                checkpoint_pmids,
                checkpoint_vectors,
            )

        conn.close()


if __name__ == "__main__":
    main()