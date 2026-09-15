import numpy as np

from ldpfp.training_data import (
    ProteinLiteratureDataset,
    collate_protein_batch,
)


def test_variable_length_dataset_and_collation():
    pmid_to_vec = {
        "1": np.ones(4, dtype=np.float32),
        "2": np.ones(4, dtype=np.float32) * 2,
        "3": np.ones(4, dtype=np.float32) * 3,
        "4": np.ones(4, dtype=np.float32) * 4,
    }

    protein_to_pmids = {
        "P1": ["1"],
        "P2": ["2", "3", "4"],
    }

    protein_to_labels = {
        "P1": np.array([1, 0, 1], dtype=np.float32),
        "P2": np.array([0, 1, 0], dtype=np.float32),
    }

    dataset = ProteinLiteratureDataset(
        protein_ids=["P1", "P2"],
        protein_to_pmids=protein_to_pmids,
        protein_to_labels=protein_to_labels,
        pmid_to_vec=pmid_to_vec,
    )

    batch = [
        dataset[0],
        dataset[1],
    ]

    protein_ids, embeddings, mask, labels = collate_protein_batch(batch)

    assert protein_ids == ["P1", "P2"]

    # Largest protein has 3 publications.
    assert embeddings.shape == (2, 3, 4)

    assert mask.shape == (2, 3)

    assert mask.tolist() == [
        [True, False, False],
        [True, True, True],
    ]

    assert labels.shape == (2, 3)

    # Padding must actually be zero.
    assert embeddings[0, 1:].sum().item() == 0