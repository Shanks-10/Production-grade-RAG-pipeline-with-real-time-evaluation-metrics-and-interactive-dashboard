"""
LLM Generator Module
Handles answer generation via Ollama and Groq, including LLM-as-judge evaluation.
"""

import json
import re
import requests
from groq import Groq

import config

# ─── System Prompt ────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a document assistant. You ONLY answer from the provided context. You never use your own knowledge. If the context does not contain the answer, say so."""

RAG_USER_PROMPT = """Here is the context from the documents:

{context}

---

Question: {question}

Answer the question using ONLY the context above. Include all relevant details and complete sentences from the context. Do NOT use your own knowledge. Do NOT say "I cannot" — just answer with what is in the context.
"""

FAITHFULNESS_PROMPT = """You are an evaluation judge. Your task is to assess whether the given answer is faithfully grounded in the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Evaluate the answer on a scale of 0.0 to 1.0:
- 1.0 = Completely faithful, every claim is supported by the context
- 0.5 = Partially faithful, some claims are supported but others are not
- 0.0 = Not faithful, the answer contains information not in the context

Respond with ONLY a JSON object in this exact format:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string with relevance scores."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("score", 0)
        parts.append(f"[Source {i} | Relevance: {score:.2f} | {source}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _clean_answer(text: str) -> str:
    """Post-process LLM output to fix common formatting issues."""
    # Replace bullet character with proper markdown dash
    text = text.replace('•', '-')
    # Fix bullets jammed onto one line
    text = re.sub(r'\s+- ', '\n- ', text)
    # Ensure headers have blank line before them
    text = re.sub(r'([^\n])\n(## )', r'\1\n\n\2', text)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def generate_ollama(query: str, context_chunks: list[dict], temperature: float = 0.3) -> str:
    """Generate answer using Ollama."""
    context = _build_context(context_chunks)
    user_prompt = RAG_USER_PROMPT.format(context=context, question=query)

    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": user_prompt,
            "system": RAG_SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 2048},
        },
        timeout=120,
    )
    response.raise_for_status()
    return _clean_answer(response.json().get("response", "").strip())


def generate_groq(query: str, context_chunks: list[dict], temperature: float = 0.3) -> str:
    """Generate answer using Groq."""
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in .env")

    context = _build_context(context_chunks)
    user_prompt = RAG_USER_PROMPT.format(context=context, question=query)

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    return _clean_answer(response.choices[0].message.content.strip())


def generate_answer(query: str, context_chunks: list[dict], provider: str = None) -> str:
    """Generate answer using the configured LLM provider."""
    provider = provider or config.LLM_PROVIDER
    if provider == "groq":
        return generate_groq(query, context_chunks)
    return generate_ollama(query, context_chunks)


def judge_faithfulness(query: str, context_chunks: list[dict], answer: str) -> dict:
    """
    Use LLM-as-judge to evaluate faithfulness of the answer.
    Returns: {"score": float, "reasoning": str}
    """
    context = _build_context(context_chunks)
    prompt = FAITHFULNESS_PROMPT.format(
        context=context, question=query, answer=answer
    )

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # Try to parse JSON from response
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        return {
            "score": float(result.get("score", 0.5)),
            "reasoning": result.get("reasoning", "No reasoning provided"),
        }
    except Exception as e:
        return {"score": 0.5, "reasoning": f"Evaluation failed: {str(e)}"}
