"""Utilities for constructing variable-length protein training samples."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinLiteratureDataset(Dataset):
    """
    One sample = one protein.

    Each protein can have a different number of PMID embeddings, so samples
    remain variable-length here. Padding is performed dynamically by
    collate_protein_batch().
    """

    def __init__(
        self,
        protein_ids: list[str],
        protein_to_pmids: dict[str, list[str]],
        protein_to_labels: dict[str, np.ndarray],
        pmid_to_vec: dict[str, np.ndarray],
    ):
        self.samples = []

        for protein_id in protein_ids:
            pmids = protein_to_pmids.get(protein_id, [])

            vectors = [
                pmid_to_vec[pmid]
                for pmid in pmids
                if pmid in pmid_to_vec
            ]

            # The literature model requires at least one document.
            if not vectors:
                continue

            if protein_id not in protein_to_labels:
                continue

            embeddings = np.stack(vectors).astype(np.float32)
            labels = np.asarray(
                protein_to_labels[protein_id],
                dtype=np.float32,
            )

            self.samples.append(
                (
                    protein_id,
                    embeddings,
                    labels,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_id, embeddings, labels = self.samples[idx]

        return (
            protein_id,
            torch.from_numpy(embeddings),
            torch.from_numpy(labels),
        )


def collate_protein_batch(batch):
    """
    Dynamically pad publications only to the largest k in this batch.

    Returns:
        protein_ids: list[str]
        padded: (batch, k_max, embed_dim)
        mask: (batch, k_max)
        labels: (batch, n_labels)
    """

    protein_ids, embedding_lists, labels = zip(*batch)

    batch_size = len(embedding_lists)
    k_max = max(x.shape[0] for x in embedding_lists)
    embed_dim = embedding_lists[0].shape[1]

    padded = torch.zeros(
        batch_size,
        k_max,
        embed_dim,
        dtype=torch.float32,
    )

    mask = torch.zeros(
        batch_size,
        k_max,
        dtype=torch.bool,
    )

    for i, embeddings in enumerate(embedding_lists):
        k = embeddings.shape[0]

        padded[i, :k] = embeddings
        mask[i, :k] = True

    labels = torch.stack(labels)

    return list(protein_ids), padded, mask, labels