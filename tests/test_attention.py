import pytest
import torch

from ldpfp.attention import PMIDAttention


def test_attention_weights_sum_to_one():
    att = PMIDAttention(embed_dim=16, hidden_dim=8)

    docs = torch.randn(2, 3, 16)

    mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])

    weighted, alpha = att(docs, mask)

    assert weighted.shape == (2, 16)
    assert alpha.shape == (2, 3)

    assert torch.allclose(
        alpha.sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )

    # Padding must receive zero attention
    assert alpha[0, 2].item() == pytest.approx(0.0, abs=1e-7)


def test_attention_no_mask():
    att = PMIDAttention(embed_dim=8, hidden_dim=4)

    docs = torch.randn(1, 2, 8)

    weighted, alpha = att(docs)

    assert weighted.shape == (1, 8)
    assert alpha.shape == (1, 2)

    assert alpha.sum().item() == pytest.approx(
        1.0,
        abs=1e-5,
    )