"""
RAG Evaluation Metrics Engine
Computes 6 key metrics for evaluating RAG system quality.
"""

import time
import numpy as np

from rag.embeddings import encode
from rag.pipeline import query_rag
from rag.generator import judge_faithfulness


# No predefined QA pairs — all metrics are computed live from user questions.
EVAL_QA_PAIRS = []



class RAGEvaluator:
    """Evaluates a RAG pipeline with 6 comprehensive metrics."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k

    def evaluate_single(
        self,
        question: str,
        reference_answer: str = "",
        progress_callback=None,
    ) -> dict:
        """Evaluate a single question through the RAG pipeline."""
        # Run RAG pipeline
        result = query_rag(question, top_k=self.top_k)
        answer = result["answer"]
        sources = result["sources"]

        metrics = {
            "question": question,
            "answer": answer,
            "reference_answer": reference_answer,
            "num_sources": len(sources),
        }

        # 1. Retrieval Precision@K
        if sources:
            above_threshold = sum(1 for s in sources if s["score"] >= 0.3)
            metrics["precision_at_k"] = round(above_threshold / len(sources), 4)
        else:
            metrics["precision_at_k"] = 0.0

        # 2. Context Relevance (avg cosine similarity of query vs sources)
        if sources:
            query_emb = encode([question])[0]
            source_texts = [s["text"] for s in sources]
            source_embs = encode(source_texts)
            similarities = [
                float(np.dot(query_emb, se) / (np.linalg.norm(query_emb) * np.linalg.norm(se) + 1e-8))
                for se in source_embs
            ]
            metrics["context_relevance"] = round(np.mean(similarities), 4)
        else:
            metrics["context_relevance"] = 0.0

        # 3. Answer Relevance (cosine similarity of query vs answer)
        if answer:
            query_emb = encode([question])[0]
            answer_emb = encode([answer])[0]
            sim = float(np.dot(query_emb, answer_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(answer_emb) + 1e-8))
            metrics["answer_relevance"] = round(max(0, sim), 4)
        else:
            metrics["answer_relevance"] = 0.0

        # 4. Faithfulness (LLM-as-judge)
        if sources and answer:
            faith_result = judge_faithfulness(question, sources, answer)
            metrics["faithfulness"] = faith_result["score"]
            metrics["faithfulness_reasoning"] = faith_result["reasoning"]
        else:
            metrics["faithfulness"] = 0.0
            metrics["faithfulness_reasoning"] = "No sources or answer to evaluate"

        # 5. Chunk Utilization
        if sources and answer:
            source_words = set()
            for s in sources:
                source_words.update(s["text"].lower().split())
            answer_words = set(answer.lower().split())
            # Remove stopwords for a fairer comparison
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                         "to", "for", "of", "and", "or", "but", "it", "this", "that",
                         "with", "from", "by", "as", "not", "be", "have", "has", "had",
                         "do", "does", "did", "will", "would", "can", "could", "should",
                         "i", "you", "he", "she", "we", "they", "my", "your", "his", "her"}
            source_words -= stopwords
            answer_words -= stopwords
            if source_words:
                overlap = len(answer_words & source_words) / len(source_words)
                metrics["chunk_utilization"] = round(min(overlap * 3, 1.0), 4)  # Scale up
            else:
                metrics["chunk_utilization"] = 0.0
        else:
            metrics["chunk_utilization"] = 0.0

        # 6. Latency
        metrics["latency_retrieval"] = result["latency_retrieval"]
        metrics["latency_generation"] = result["latency_generation"]
        metrics["total_latency"] = result["total_latency"]

        # Source details
        metrics["sources"] = sources

        return metrics

    def evaluate_batch(
        self,
        qa_pairs: list[dict] = None,
        progress_callback=None,
    ) -> dict:
        """Evaluate on a batch of Q&A pairs. Returns aggregated metrics + per-query details."""
        if qa_pairs is None:
            qa_pairs = EVAL_QA_PAIRS

        results = []
        for i, qa in enumerate(qa_pairs):
            result = self.evaluate_single(
                qa["question"],
                qa.get("reference", ""),
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(qa_pairs))

        # Aggregate
        aggregated = self._aggregate(results)
        aggregated["per_query"] = results
        return aggregated

    def _aggregate(self, results: list[dict]) -> dict:
        """Compute aggregate metrics from individual results."""
        def _safe_mean(values):
            valid = [v for v in values if v is not None]
            return round(np.mean(valid), 4) if valid else None

        return {
            "num_queries": len(results),
            "avg_precision_at_k": _safe_mean([r["precision_at_k"] for r in results]),
            "avg_context_relevance": _safe_mean([r["context_relevance"] for r in results]),
            "avg_answer_relevance": _safe_mean([r["answer_relevance"] for r in results]),
            "avg_faithfulness": _safe_mean([r["faithfulness"] for r in results]),
            "avg_chunk_utilization": _safe_mean([r["chunk_utilization"] for r in results]),
            "avg_retrieval_latency": _safe_mean([r["latency_retrieval"] for r in results]),
            "avg_generation_latency": _safe_mean([r["latency_generation"] for r in results]),
            "avg_latency": _safe_mean([r["total_latency"] for r in results]),

        }
