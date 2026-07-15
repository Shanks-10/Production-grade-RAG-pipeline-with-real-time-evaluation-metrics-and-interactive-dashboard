"""
Context Compressor Module
Extracts only the most relevant sentences from retrieved chunks,
improving Chunk Utilization and reducing noise for the generator.
"""

import requests
import config


COMPRESS_PROMPT = """Extract ONLY the sentences from the following context that are relevant to answering the question. 
Return the relevant sentences exactly as they appear — do NOT paraphrase, summarize, or add any new information.
If no sentences are relevant, return "NO_RELEVANT_CONTENT".

Question: {question}

Context:
{chunk_text}

Relevant sentences:"""


def compress_chunk(question: str, chunk_text: str) -> str:
    """Use the LLM to extract only relevant sentences from a chunk."""
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": COMPRESS_PROMPT.format(question=question, chunk_text=chunk_text),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        if "NO_RELEVANT_CONTENT" in result or not result:
            return chunk_text  # Fall back to original if compression fails
        return result
    except Exception:
        return chunk_text  # Fall back to original on error


def compress_sources(question: str, sources: list[dict]) -> list[dict]:
    """
    Compress all retrieved chunks by extracting only relevant sentences.
    Returns new source dicts with compressed text + original text preserved.
    """
    compressed = []
    for source in sources:
        original_text = source["text"]
        compressed_text = compress_chunk(question, original_text)
        new_source = source.copy()
        new_source["text"] = compressed_text
        new_source["original_text"] = original_text
        new_source["compression_ratio"] = round(
            len(compressed_text) / max(len(original_text), 1), 3
        )
        compressed.append(new_source)
    return compressed
