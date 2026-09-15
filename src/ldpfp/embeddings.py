"""Phase 4: Generate document-level embeddings with a biomedical transformer."""
from __future__ import annotations
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512  # BERT-family hard limit; long full-text docs are chunked+averaged
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BiomedicalEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Mean-pooled [CLS]-free embedding, chunked for texts longer than 512 tokens."""
        tokens = self.tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        if len(tokens) <= MAX_LENGTH:
            return self._embed_chunk(text)
        # chunk long full-text into overlapping windows, average the chunk embeddings
        stride = MAX_LENGTH - 50
        chunk_vecs = []
        for start in range(0, len(tokens), stride):
            chunk_ids = tokens[start:start + MAX_LENGTH]
            if not chunk_ids:
                break
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunk_vecs.append(self._embed_chunk(chunk_text))
            if start + MAX_LENGTH >= len(tokens):
                break
        return np.mean(chunk_vecs, axis=0)

    @torch.no_grad()
    def _embed_chunk(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        # mean pooling over token embeddings, masked by attention_mask
        last_hidden = outputs.last_hidden_state  # (1, seq_len, d)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.squeeze(0).cpu().numpy()

    def embed_batch(self, pmid_to_text: dict[str, str]) -> dict[str, np.ndarray]:
        out = {}
        for pmid, text in tqdm(pmid_to_text.items(), desc="Embedding documents"):
            out[pmid] = self.embed_text(text)
        return out
