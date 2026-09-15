import numpy as np
from ldpfp.text_preprocessing import load_and_assemble
from ldpfp.embeddings import BiomedicalEmbedder

pmid_texts = load_and_assemble("data/literature_cache.sqlite")
embedder = BiomedicalEmbedder()
vectors = embedder.embed_batch(pmid_texts)

np.savez_compressed(
    "data/processed/embeddings/pmid_embeddings.npz",
    pmids=np.array(list(vectors.keys())),
    vectors=np.stack(list(vectors.values())),
)
print(f"Embedded {len(vectors):,} documents, dim={next(iter(vectors.values())).shape[0]}")
