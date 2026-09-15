from ldpfp.embeddings import BiomedicalEmbedder


def test_embed_text_shape():
    embedder = BiomedicalEmbedder(
        model_name="google/bert_uncased_L-2_H-128_A-2"
    )

    vec = embedder.embed_text(
        "Kinase activity was measured in vitro."
    )

    assert vec.ndim == 1
    assert vec.shape[0] == 128