"""
Retriever Module
Direct similarity search against Qdrant using bilingual embeddings.
"""

from qdrant_client import QdrantClient, models

import config
from rag.embeddings import encode


def get_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def retrieve(
    query: str,
    top_k: int = None,
    score_threshold: float = None,
) -> list[dict]:
    """
    Embed the query and search Qdrant for the most relevant chunks.
    Simple, direct vector search — no multi-query noise.
    """
    top_k = top_k or config.TOP_K
    score_threshold = score_threshold or config.RELEVANCE_THRESHOLD

    # Encode query
    query_embedding = encode([query])[0].tolist()

    # Search Qdrant
    client = get_client()
    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        score_threshold=score_threshold,
    )

    # Format results
    hits = []
    for r in results.points:
        hits.append({
            "text": r.payload.get("text", ""),
            "score": round(r.score, 4),
            "source": r.payload.get("source", "unknown"),
            "chunk_index": r.payload.get("chunk_index", -1),
            "metadata": {k: v for k, v in r.payload.items() if k not in ("text",)},
        })

    return hits
