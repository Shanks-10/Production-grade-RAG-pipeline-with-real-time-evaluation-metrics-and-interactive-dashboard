"""
RAG Pipeline — End-to-End Orchestration
Combines retrieval and generation with latency tracking.
"""

import time

from rag.retriever import retrieve
from rag.generator import generate_answer


def query_rag(
    query: str,
    top_k: int = None,
    provider: str = None,
) -> dict:
    """
    Execute the full RAG pipeline:
      1. Retrieve relevant chunks from Qdrant
      2. Generate an answer using the LLM
      3. Track latency at each stage
    """
    t_start = time.time()

    # ── Step 1: Retrieve ──
    t_retrieve_start = time.time()
    sources = retrieve(query, top_k=top_k)
    latency_retrieval = time.time() - t_retrieve_start

    if not sources:
        return {
            "answer": "No relevant documents found. Please upload some documents first.",
            "sources": [],
            "compressed_sources": [],
            "latency_retrieval": latency_retrieval,
            "latency_compression": 0.0,
            "latency_generation": 0.0,
            "total_latency": time.time() - t_start,
        }

    # ── Step 2: Generate ──
    t_generate_start = time.time()
    answer = generate_answer(query, sources, provider=provider)
    latency_generation = time.time() - t_generate_start

    total_latency = time.time() - t_start

    return {
        "answer": answer,
        "sources": sources,
        "compressed_sources": sources,  # Same as sources (no compression)
        "latency_retrieval": round(latency_retrieval, 3),
        "latency_compression": 0.0,
        "latency_generation": round(latency_generation, 3),
        "total_latency": round(total_latency, 3),
    }
