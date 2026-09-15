"""Phase 4: Generate document-level biomedical embeddings."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

MAX_LENGTH = 512
CHUNK_OVERLAP = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BiomedicalEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        batch_size: int = 16,
    ):
        self.device = device
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        print(f"Device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = (
            AutoModel
            .from_pretrained(model_name)
            .to(device)
            .eval()
        )

        self.hidden_size = self.model.config.hidden_size

    def _chunk_text(self, text: str) -> list[list[int]]:
        """
        Tokenize a document and split it into overlapping token windows.

        Special tokens are added later by the tokenizer.
        """

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        # Reserve room for [CLS] and [SEP].
        content_length = MAX_LENGTH - 2
        stride = content_length - CHUNK_OVERLAP

        chunks = []

        for start in range(0, len(token_ids), stride):
            chunk = token_ids[start:start + content_length]

            if not chunk:
                break

            chunks.append(chunk)

            if start + content_length >= len(token_ids):
                break

        return chunks

    @torch.inference_mode()
    def _embed_token_chunks(
    self,
    chunks: list[list[int]],
    ) -> np.ndarray:
        """
        Embed multiple token chunks using GPU batching.

        Returns one vector per chunk.
        """

        chunk_vectors = []

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        if cls_id is None or sep_id is None:
            raise ValueError(
                "Tokenizer must provide CLS and SEP token IDs."
            )

        for start in range(
            0,
            len(chunks),
            self.batch_size,
        ):
            batch_chunks = chunks[
                start:start + self.batch_size
            ]

            # Manually construct BERT input:
            # [CLS] content tokens [SEP]
            input_ids = [
                [cls_id] + chunk + [sep_id]
                for chunk in batch_chunks
            ]

            encoded = self.tokenizer.pad(
                {
                    "input_ids": input_ids,
                },
                padding=True,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            outputs = self.model(**encoded)

            hidden = outputs.last_hidden_state

            attention_mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .float()
            )

            pooled = (
                (hidden * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp(min=1e-9)
            )

            chunk_vectors.append(
                pooled.cpu().numpy()
            )

        return np.concatenate(
            chunk_vectors,
            axis=0,
        )
    def embed_text(self, text: str) -> np.ndarray:
        """
        Produce one embedding for a publication.

        Long documents are split into overlapping chunks.
        Chunk embeddings are averaged to produce the final
        document-level vector.
        """

        chunks = self._chunk_text(text)

        if not chunks:
            return np.zeros(
                self.hidden_size,
                dtype=np.float32,
            )

        chunk_vectors = self._embed_token_chunks(chunks)

        return chunk_vectors.mean(axis=0).astype(
            np.float32
        )