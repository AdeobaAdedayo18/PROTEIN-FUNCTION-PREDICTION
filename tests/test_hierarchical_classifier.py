import torch

from ldpfp.models.hierarchical_classifier import (
    HierarchicalGOClassifier,
    hierarchical_loss,
    enforce_consistency,
)


def test_classifier_output_shapes():
    model = HierarchicalGOClassifier(
        embed_dim=16,
        hidden_dim=8,
        n_labels=5,
    )

    docs = torch.randn(2, 3, 16)

    mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])

    logits, alpha = model(docs, mask)

    assert logits.shape == (2, 5)
    assert alpha.shape == (2, 3)


def test_hierarchical_loss_penalizes_violation():
    # Label index 0 = child
    # Label index 1 = parent
    pairs = [(0, 1)]

    targets = torch.zeros((1, 2))

    # Child probability high, parent low
    violating_logits = torch.tensor([
        [5.0, -5.0]
    ])

    # Child probability low, parent high
    consistent_logits = torch.tensor([
        [-5.0, 5.0]
    ])

    violating_loss = hierarchical_loss(
        violating_logits,
        targets,
        pairs,
        lam=1.0,
    )

    consistent_loss = hierarchical_loss(
        consistent_logits,
        targets,
        pairs,
        lam=1.0,
    )

    # Isolate the hierarchy contribution rather than comparing
    # total BCE + hierarchy losses.
    violating_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        violating_logits,
        targets,
    )

    consistent_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        consistent_logits,
        targets,
    )

    violating_penalty = violating_loss - violating_bce
    consistent_penalty = consistent_loss - consistent_bce

    assert violating_penalty > consistent_penalty


def test_enforce_consistency():
    pairs = [(0, 1)]

    # child=0.9, parent=0.2 -> violation
    probs = torch.tensor([
        [0.9, 0.2]
    ])

    corrected = enforce_consistency(
        probs,
        pairs,
    )

    assert corrected[0, 0] <= corrected[0, 1]


def test_hierarchical_loss_without_pairs():
    logits = torch.randn(2, 4)
    targets = torch.zeros(2, 4)

    loss = hierarchical_loss(
        logits,
        targets,
        [],
    )

    assert torch.isfinite(loss)