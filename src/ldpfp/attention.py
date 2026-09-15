"""Phase 5: Attention-based PMID weighting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PMIDAttention(nn.Module):
    """
    Learned additive attention over publication embeddings belonging
    to a protein.

    The attention mechanism assigns a normalized importance weight
    to each publication and produces a weighted protein-level
    representation.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        doc_embeddings:
            (batch_size, k_max, embed_dim)

        mask:
            (batch_size, k_max)
            True for real publications, False for padding.

        Returns:
            weighted:
                (batch_size, embed_dim)

            alpha:
                (batch_size, k_max)
        """

        # Compute one relevance score per publication
        scores = self.score_fn(doc_embeddings).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # Normalize scores across publications belonging to the protein
        alpha = torch.softmax(scores, dim=-1)

        # Safety guard for a protein containing no valid documents
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Weighted sum of publication embeddings
        weighted = torch.bmm(
            alpha.unsqueeze(1),
            doc_embeddings,
        ).squeeze(1)

        return weighted, alpha


def build_protein_batch(
    pmid_lists: list[list[str]],
    pmid_to_vec: dict[str, np.ndarray],
    embed_dim: int = 768,
):
    """
    Convert variable-length PMID lists into a padded tensor.

    Returns:
        batch:
            (batch_size, k_max, embed_dim)

        mask:
            (batch_size, k_max)
    """

    k_max = max((len(pmids) for pmids in pmid_lists), default=1)
    k_max = max(k_max, 1)

    batch = np.zeros(
        (len(pmid_lists), k_max, embed_dim),
        dtype=np.float32,
    )

    mask = np.zeros(
        (len(pmid_lists), k_max),
        dtype=bool,
    )

    for i, pmids in enumerate(pmid_lists):
        for j, pmid in enumerate(pmids):

            vec = pmid_to_vec.get(str(pmid))

            if vec is not None:
                batch[i, j] = vec
                mask[i, j] = True

    return (
        torch.from_numpy(batch),
        torch.from_numpy(mask),
    )