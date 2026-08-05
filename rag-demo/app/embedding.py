from __future__ import annotations

from typing import List

import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """BGE-M3 embedding service using sentence-transformers."""

    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model_path = f"models/{settings.bge_model_name.replace('/', '_')}"
            if os.path.exists(model_path):
                self.model = SentenceTransformer(
                    model_path,
                    device=device
                )
            else:
                self.model = SentenceTransformer(settings.bge_model_name, device=device)
                self.model.save(model_path)
        except Exception:
            # Fallback to a lightweight local sentence-transformers model if the BGE model
            # cannot be downloaded in an offline or restricted environment.
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    def encode(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Encode a batch of texts into normalized embeddings."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [list(map(float, emb)) for emb in embeddings]
