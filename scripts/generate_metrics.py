#!/usr/bin/env python3
"""
CLI Metrics Report Generator
Runs a full RAG evaluation and prints/exports results.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from metrics.evaluator import RAGEvaluator, EVAL_QA_PAIRS
from metrics.report import export_json, export_csv


def print_header(text: str, char: str = "═"):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_metric(name: str, value, fmt: str = ".2%"):
    if value is None:
        print(f"  {name:<30} {'N/A':>10}")
    else:
        print(f"  {name:<30} {value:{fmt}}")


def main():
    print_header("RAG EVALUATION REPORT", "█")
    print(f"  Evaluating {len(EVAL_QA_PAIRS)} predefined Q&A pairs...")
    print(f"  This may take a few minutes...\n")

    evaluator = RAGEvaluator()

    def progress(current, total):
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  Progress: [{bar}] {current}/{total}", end="", flush=True)

    results = evaluator.evaluate_batch(progress_callback=progress)
    print()  # newline after progress bar

    # ─── Aggregate Metrics ────────────────────────────────────────────────
    print_header("AGGREGATE METRICS")
    print_metric("Retrieval Precision@K", results["avg_precision_at_k"])
    print_metric("Context Relevance", results["avg_context_relevance"])
    print_metric("Answer Relevance", results["avg_answer_relevance"])
    print_metric("Faithfulness", results["avg_faithfulness"])
    print_metric("ROUGE-L", results["avg_rouge_l"])
    print_metric("BLEU", results["avg_bleu"])
    print_metric("Chunk Utilization", results["avg_chunk_utilization"])
    print()
    print_metric("Avg Retrieval Latency", results["avg_retrieval_latency"], ".3f")
    print(f"{'':>32} seconds")
    print_metric("Avg Generation Latency", results["avg_generation_latency"], ".3f")
    print(f"{'':>32} seconds")
    print_metric("Avg Total Latency", results["avg_latency"], ".3f")
    print(f"{'':>32} seconds")

    # ─── Per-Query Details ────────────────────────────────────────────────
    print_header("PER-QUERY RESULTS")
    for i, q in enumerate(results.get("per_query", []), 1):
        print(f"\n  Q{i}: {q['question']}")
        print(f"  {'─' * 60}")
        print(f"  Answer: {q['answer'][:120]}...")
        print(f"  Precision@K: {q['precision_at_k']:.2%}  "
              f"Context: {q['context_relevance']:.2%}  "
              f"Answer: {q['answer_relevance']:.2%}  "
              f"Faithful: {q['faithfulness']:.2%}")
        if q.get("rouge_l") is not None:
            print(f"  ROUGE-L: {q['rouge_l']:.2%}  BLEU: {q['bleu']:.2%}")
        print(f"  Latency: {q['total_latency']:.2f}s "
              f"(retrieval: {q['latency_retrieval']:.3f}s, "
              f"generation: {q['latency_generation']:.3f}s)")

    # ─── Export ───────────────────────────────────────────────────────────
    data_dir = PROJECT_DIR / "data"
    json_path = export_json(results, data_dir / "metrics_report.json")
    csv_path = export_csv(results, data_dir / "metrics_report.csv")

    print_header("EXPORT")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print()
    print("  ✅ Evaluation complete!")
    print()


if __name__ == "__main__":
    main()
