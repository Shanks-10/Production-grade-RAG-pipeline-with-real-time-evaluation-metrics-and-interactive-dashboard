"""
Embedding Model Wrapper
Singleton loader for the Lajavaness/bilingual-embedding-large model
using sentence-transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import config

_model = None


def get_model() -> SentenceTransformer:
    """Load and cache the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True)
        print(f"[Embeddings] Model loaded — dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def encode(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Encode a list of texts into dense vectors."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress, normalize_embeddings=True)
    return np.array(embeddings)


def get_dimension() -> int:
    """Return the embedding dimension."""
    return get_model().get_sentence_embedding_dimension()
