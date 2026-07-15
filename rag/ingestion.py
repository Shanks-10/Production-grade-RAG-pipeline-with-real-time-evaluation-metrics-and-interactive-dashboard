"""
Document Ingestion Module
Handles PDF/text loading, recursive chunking, and Qdrant upsert.
"""

import uuid
import time
from pathlib import Path

from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct
)

import config
from rag.embeddings import encode, get_dimension


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client."""
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def ensure_collection(client: QdrantClient | None = None) -> QdrantClient:
    """Ensure the Qdrant collection exists. Create if missing."""
    if client is None:
        client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in collections:
        dim = get_dimension()
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[Ingestion] Created collection '{config.QDRANT_COLLECTION}' (dim={dim})")
    return client


def load_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def load_pdf_bytes(file_bytes, filename: str = "upload.pdf") -> str:
    """Extract text from PDF bytes (Streamlit upload)."""
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    """
    Semantic chunking: split by section headers first, then by paragraphs if needed.
    Keeps coherent document sections together regardless of length (up to MAX_CHUNK).
    Only splits further if a section exceeds the max limit.
    """
    import re
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    MAX_CHUNK = 3000  # Hard cap — split sections longer than this

    if not text.strip():
        return []

    # Step 1: Split text into sections by detecting headers
    # Headers: lines that are short (<80 chars), often title-cased or ALL CAPS,
    # followed by content. Also detect numbered section headers like "1.", "1.1", "Chapter"
    lines = text.split('\n')
    sections = []
    current_section = []

    def _is_header(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 100:
            return False
        # Numbered headers: "1.", "1.1", "2.3.1" etc
        if re.match(r'^\d+(\.\d+)*\.?\s+\w', stripped):
            return True
        # Title-like: short line, mostly capitalized words
        words = stripped.split()
        if len(words) <= 8 and sum(1 for w in words if w[0].isupper()) >= len(words) * 0.5:
            # Check it's not a regular sentence (no period at end unless it's short)
            if not stripped.endswith('.') or len(words) <= 3:
                return True
        # ALL CAPS headers
        if stripped.isupper() and len(words) <= 10:
            return True
        return False

    for i, line in enumerate(lines):
        if _is_header(line) and current_section:
            # Save previous section
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append(section_text)
            current_section = [line]
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            sections.append(section_text)

    # If no sections detected, fall back to paragraph splitting
    if len(sections) <= 1:
        sections = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Step 2: Merge tiny sections with the next one, split huge sections
    final_chunks = []
    buffer = ""

    for section in sections:
        if buffer:
            candidate = buffer + "\n\n" + section
        else:
            candidate = section

        if len(candidate) <= MAX_CHUNK:
            # If it's still small, keep buffering
            if len(candidate) < chunk_size // 2:
                buffer = candidate
            else:
                final_chunks.append(candidate.strip())
                buffer = ""
        else:
            # Flush buffer first
            if buffer and buffer != candidate:
                final_chunks.append(buffer.strip())
                buffer = ""

            # Split the large section by paragraphs
            if len(section) > MAX_CHUNK:
                paragraphs = section.split('\n\n')
                para_buffer = ""
                for para in paragraphs:
                    if para_buffer:
                        test = para_buffer + "\n\n" + para
                    else:
                        test = para
                    if len(test) <= MAX_CHUNK:
                        para_buffer = test
                    else:
                        if para_buffer:
                            final_chunks.append(para_buffer.strip())
                        para_buffer = para
                if para_buffer:
                    final_chunks.append(para_buffer.strip())
            else:
                final_chunks.append(section.strip())

    if buffer:
        final_chunks.append(buffer.strip())

    # Step 3: Apply overlap between chunks
    if chunk_overlap > 0 and len(final_chunks) > 1:
        overlapped = [final_chunks[0]]
        for i in range(1, len(final_chunks)):
            prev = final_chunks[i - 1]
            overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            overlapped.append(overlap_text + "\n" + final_chunks[i])
        final_chunks = overlapped

    return [c for c in final_chunks if c.strip()]


def ingest_chunks(
    chunks: list[str],
    metadata: dict = None,
    client: QdrantClient | None = None,
    progress_callback=None,
) -> int:
    """Embed chunks and upsert into Qdrant."""
    if not chunks:
        return 0

    client = ensure_collection(client)
    metadata = metadata or {}

    # Embed in batches
    batch_size = 32
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = encode(batch)

        points = []
        for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "chunk_index": i + j,
                "ingested_at": time.time(),
                **metadata,
            }
            points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        total_upserted += len(points)

        if progress_callback:
            progress_callback(total_upserted, len(chunks))

    return total_upserted


def ingest_text(text: str, metadata: dict = None, progress_callback=None) -> int:
    """Convenience: chunk text and ingest into Qdrant."""
    chunks = chunk_text(text)
    return ingest_chunks(chunks, metadata, progress_callback=progress_callback)


def ingest_pdf(file_path: str | Path, progress_callback=None) -> int:
    """Convenience: load PDF, chunk, and ingest."""
    text = load_pdf(file_path)
    metadata = {"source": str(file_path), "type": "pdf"}
    return ingest_text(text, metadata, progress_callback=progress_callback)


def get_collection_info() -> dict:
    """Get info about the current Qdrant collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(config.QDRANT_COLLECTION)
        return {
            "name": config.QDRANT_COLLECTION,
            "vectors_count": info.points_count or 0,
            "points_count": info.points_count or 0,
            "status": str(info.status),
        }
    except Exception:
        return {"name": config.QDRANT_COLLECTION, "vectors_count": 0, "points_count": 0, "status": "not_found"}


def clear_collection() -> bool:
    """Delete and recreate the collection."""
    try:
        client = get_qdrant_client()
        client.delete_collection(config.QDRANT_COLLECTION)
        ensure_collection(client)
        return True
    except Exception:
        return False
