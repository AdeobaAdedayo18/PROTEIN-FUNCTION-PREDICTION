"""Phase 6b: Attention pooling -> GO classifier -> hierarchical loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldpfp.attention import PMIDAttention


class HierarchicalGOClassifier(nn.Module):
    """
    Multi-label GO classifier operating on publication embeddings.

    PMIDAttention first combines the variable number of publication
    embeddings belonging to each protein into a single protein
    representation R(p_i). The classifier then predicts one logit
    for every GO term in the vocabulary.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        n_labels: int = 17707,
    ):
        super().__init__()

        self.attention = PMIDAttention(
            embed_dim=embed_dim,
            hidden_dim=256,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(
        self,
        doc_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ):
        # R: (batch, embed_dim)
        # alpha: (batch, k_max)
        R, alpha = self.attention(doc_embeddings, mask)

        # logits: (batch, n_labels)
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