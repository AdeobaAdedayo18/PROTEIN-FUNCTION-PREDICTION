"""Phase 6b: Attention pooling -> GO classifier -> hierarchical loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldpfp.attention import PMIDAttention


class HierarchicalGOClassifier(nn.Module):
    """
    Multi-label GO classifier operating on publication embeddings.

    Supported protein-level pooling strategies:
        - attention: learned PMID attention (proposed model)
        - mean: masked mean pooling
        - max: masked max pooling

    All pooling strategies produce the same (batch, embed_dim)
    protein representation so that the downstream classifier remains
    identical across ablation experiments.
    """

    VALID_POOLING = {"attention", "mean", "max"}

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        n_labels: int = 17707,
        pooling: str = "attention",
    ):
        super().__init__()

        pooling = pooling.lower().strip()

        if pooling not in self.VALID_POOLING:
            raise ValueError(
                f"Unknown pooling strategy {pooling!r}. "
                f"Expected one of {sorted(self.VALID_POOLING)}."
            )

        self.pooling = pooling

        # Only the proposed model needs learned attention parameters.
        if pooling == "attention":
            self.attention = PMIDAttention(
                embed_dim=embed_dim,
                hidden_dim=256,
            )
        else:
            self.attention = None

        # IMPORTANT:
        # Keep this classifier identical across all ablations.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_labels),
        )

    def _mean_pool(
        self,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        mask_f = mask.unsqueeze(-1).to(
            dtype=doc_embeddings.dtype
        )

        summed = (
            doc_embeddings * mask_f
        ).sum(dim=1)

        counts = mask_f.sum(dim=1).clamp_min(1.0)

        return summed / counts

    def _max_pool(
        self,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        masked_embeddings = doc_embeddings.masked_fill(
            ~mask.unsqueeze(-1),
            float("-inf"),
        )

        pooled = masked_embeddings.max(dim=1).values

        # Safety guard in case a sample somehow contains no
        # valid publication embeddings.
        pooled = torch.where(
            torch.isfinite(pooled),
            pooled,
            torch.zeros_like(pooled),
        )

        return pooled

    def forward(
        self,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ):

        if self.pooling == "attention":

            R, alpha = self.attention(
                doc_embeddings,
                mask,
            )

        elif self.pooling == "mean":

            R = self._mean_pool(
                doc_embeddings,
                mask,
            )

            # No learned attention weights exist for this ablation.
            alpha = None

        elif self.pooling == "max":

            R = self._max_pool(
                doc_embeddings,
                mask,
            )

            alpha = None

        else:
            # Defensive guard. __init__ already validates this.
            raise RuntimeError(
                f"Unsupported pooling strategy: {self.pooling}"
            )

        logits = self.classifier(R)

        return logits, alpha

def hierarchical_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    parent_child_pairs: list[tuple[int, int]],
    lam: float = 0.5,
) -> torch.Tensor:
    """
    L_total = L_BCE + lambda * L_hierarchy

    The hierarchy term penalizes predictions where:

        P(child) > P(parent)

    which violates the GO true-path constraint.
    """

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    if not parent_child_pairs:
        return bce

    probs = torch.sigmoid(logits)

    child_idx = torch.tensor(
        [child for child, _ in parent_child_pairs],
        dtype=torch.long,
        device=logits.device,
    )

    parent_idx = torch.tensor(
        [parent for _, parent in parent_child_pairs],
        dtype=torch.long,
        device=logits.device,
    )

    child_probs = probs[:, child_idx]
    parent_probs = probs[:, parent_idx]

    violations = F.relu(
        child_probs - parent_probs
    )

    hierarchy_penalty = violations.mean()

    return bce + lam * hierarchy_penalty


def enforce_consistency(
    probs: torch.Tensor,
    parent_child_pairs: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Post-processing step for inference.

    Ensures that no child's probability exceeds the probability
    assigned to its parent.
    """

    probs = probs.clone()

    for _ in range(20):
        changed = False

        for child, parent in parent_child_pairs:
            violation_mask = (
                probs[:, child] > probs[:, parent]
            )

            if violation_mask.any():
                probs[violation_mask, child] = (
                    probs[violation_mask, parent]
                )
                changed = True

        if not changed:
            break

    return probs