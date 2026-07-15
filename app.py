"""
RAG Demo — Streamlit Application
A beautiful 4-tab interface for demonstrating Retrieval-Augmented Generation.
"""

import sys
import time
import json
from pathlib import Path

import streamlit as st
import requests

# ─── Ensure project root is in path ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

import config
from rag.ingestion import (
    ingest_text, ingest_chunks, chunk_text,
    load_pdf_bytes, get_collection_info, clear_collection, ensure_collection
)
from rag.pipeline import query_rag
from rag.embeddings import encode
from rag.generator import judge_faithfulness
from metrics.evaluator import RAGEvaluator, EVAL_QA_PAIRS
from metrics.report import (
    create_radar_chart, create_metrics_bar_chart,
    create_latency_chart, create_retrieval_heatmap,
    create_per_query_comparison, export_json, export_csv
)
import numpy as np

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load CSS ─────────────────────────────────────────────────────────────────
css_path = PROJECT_DIR / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "metrics_results" not in st.session_state:
    st.session_state.metrics_results = None
if "live_metrics" not in st.session_state:
    st.session_state.live_metrics = []
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False


def compute_live_metrics(query: str, result: dict) -> dict:
    """Compute metrics for a single live query using compressed sources."""
    answer = result["answer"]
    sources = result["sources"]
    compressed = result.get("compressed_sources", sources)
    metrics = {
        "question": query,
        "answer": answer,
        "num_sources": len(sources),
        "timestamp": time.strftime("%H:%M:%S"),
    }
    # 1. Precision@K — how many chunks scored above a meaningful threshold
    if sources:
        above_threshold = sum(1 for s in sources if s["score"] >= 0.3)
        metrics["precision_at_k"] = round(above_threshold / len(sources), 4)
    else:
        metrics["precision_at_k"] = 0.0
    # 2. Context Relevance — cosine similarity (query ↔ compressed chunks), normalized
    if compressed:
        query_emb = encode([query])[0]
        source_texts = [s["text"] for s in compressed]
        source_embs = encode(source_texts)
        sims = [float(np.dot(query_emb, se) / (np.linalg.norm(query_emb) * np.linalg.norm(se) + 1e-8)) for se in source_embs]
        # Raw cosine sim between short query and text passage is typically 0.3-0.7
        # Normalize to 0-1 range: score = (raw - 0.2) / 0.6, clamped
        raw = np.mean(sims)
        normalized = min(max((raw - 0.2) / 0.6, 0.0), 1.0)
        metrics["context_relevance"] = round(normalized, 4)
    else:
        metrics["context_relevance"] = 0.0
    # 3. Answer Relevance — cosine similarity (query ↔ answer), normalized
    if answer:
        query_emb = encode([query])[0]
        answer_emb = encode([answer])[0]
        sim = float(np.dot(query_emb, answer_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(answer_emb) + 1e-8))
        normalized = min(max((sim - 0.2) / 0.6, 0.0), 1.0)
        metrics["answer_relevance"] = round(normalized, 4)
    else:
        metrics["answer_relevance"] = 0.0
    # 4. Faithfulness (LLM-as-judge)
    if compressed and answer:
        faith = judge_faithfulness(query, compressed, answer)
        metrics["faithfulness"] = faith["score"]
        metrics["faithfulness_reasoning"] = faith["reasoning"]
    else:
        metrics["faithfulness"] = 0.0
        metrics["faithfulness_reasoning"] = "No sources"
    # 5. Chunk Utilization — per-chunk overlap with answer (higher with focused chunks)
    if compressed and answer:
        stopwords = {"the","a","an","is","are","was","were","in","on","at","to","for","of","and","or","but","it","this","that","with","from","by","as","not","be","has","have","had","been","its","they","them","their","can","will","would","could","should","more","also","than","into","about","over","such","these","those","each","which","do","does","did"}
        answer_words = set(w for w in answer.lower().split() if w not in stopwords and len(w) > 2)
        per_chunk_scores = []
        for s in compressed:
            chunk_words = set(w for w in s["text"].lower().split() if w not in stopwords and len(w) > 2)
            if chunk_words:
                overlap = len(answer_words & chunk_words) / len(chunk_words)
                per_chunk_scores.append(overlap)
        if per_chunk_scores:
            # Average per-chunk utilization, scaled to be more representative
            raw_score = np.mean(per_chunk_scores)
            metrics["chunk_utilization"] = round(min(raw_score * 3.0, 1.0), 4)
        else:
            metrics["chunk_utilization"] = 0.0
    else:
        metrics["chunk_utilization"] = 0.0
    # 6. Compression ratio
    if compressed and compressed[0].get("compression_ratio") is not None:
        ratios = [s.get("compression_ratio", 1.0) for s in compressed]
        metrics["avg_compression_ratio"] = round(np.mean(ratios), 3)
    else:
        metrics["avg_compression_ratio"] = 1.0
    # 7. Latency
    metrics["latency_retrieval"] = result["latency_retrieval"]
    metrics["latency_compression"] = result.get("latency_compression", 0.0)
    metrics["latency_generation"] = result["latency_generation"]
    metrics["total_latency"] = result["total_latency"]
    metrics["sources"] = sources
    metrics["compressed_sources"] = compressed
    return metrics


# ─── Helper Functions ─────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_qdrant() -> bool:
    try:
        r = requests.get(f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/collections", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def load_sample_data():
    """Load sample documents into Qdrant (supports .txt and .pdf)."""
    from rag.ingestion import load_pdf, ingest_pdf
    sample_dir = config.SAMPLE_DIR
    total = 0
    details = []
    for f in sorted(sample_dir.iterdir()):
        if f.suffix == '.txt':
            text = f.read_text()
            count = ingest_text(text, {"source": f.name, "type": "sample"})
            total += count
            details.append((f.name, count))
        elif f.suffix == '.pdf':
            count = ingest_pdf(f)
            total += count
            details.append((f.name, count))
    st.session_state.sample_loaded = True
    return total, details


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🧠 RAG System")
    st.divider()

    # Connection Status
    st.markdown("### 🔌 Status")
    col1, col2 = st.columns(2)
    with col1:
        qdrant_ok = check_qdrant()
        st.markdown(
            f"**Qdrant:** {'🟢 Online' if qdrant_ok else '🔴 Offline'}"
        )
    with col2:
        ollama_ok = check_ollama()
        st.markdown(
            f"**Ollama:** {'🟢 Online' if ollama_ok else '🔴 Offline'}"
        )

    st.divider()

    # Collection Info
    st.markdown("### 📦 Collection")
    info = get_collection_info()
    st.markdown(f"**Name:** `{info['name']}`")
    st.markdown(f"**Vectors:** `{info['vectors_count']}`")
    st.markdown(f"**Status:** `{info['status']}`")

    st.divider()

    # Sample Data
    st.markdown("### 📥 Quick Start")
    if st.button("📥 Load Sample Data", use_container_width=True, key="load_sample"):
        with st.spinner("Ingesting sample documents..."):
            try:
                total, details = load_sample_data()
                st.success(f"✅ Loaded {total} chunks from {len(details)} file(s)!")
                for fname, cnt in details:
                    st.write(f"  📄 **{fname}** → {cnt} chunks")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.button("🗑️ Clear Collection", use_container_width=True, key="clear_collection"):
        if clear_collection():
            st.session_state.sample_loaded = False
            st.success("✅ Collection cleared!")
            time.sleep(1)
            st.rerun()

    st.divider()

    # Model Info
    st.markdown("### 🤖 Configuration")
    active_model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.OLLAMA_MODEL
    st.markdown(f"**LLM:** `{active_model}`")
    st.markdown(f"**Provider:** `{config.LLM_PROVIDER.upper()}`")
    st.markdown(f"**Embedding:** `{config.EMBEDDING_MODEL.split('/')[-1]}`")
    st.markdown(f"**Chunk Size:** `{config.CHUNK_SIZE}`")
    st.markdown(f"**Top-K:** `{config.TOP_K}`")


# ─── Main Content — Tabs ─────────────────────────────────────────────────────

st.markdown("# 🧠 RAG System  —  Retrieval-Augmented Generation")
st.markdown("*Powered by GPT-OSS, Qdrant & Bilingual Embeddings*")

tab_upload, tab_chat, tab_metrics, tab_settings = st.tabs([
    "📄 Document Upload",
    "💬 Ask Questions",
    "📊 Metrics Dashboard",
    "⚙️ Settings",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Document Upload
# ═══════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.markdown("## 📄 Upload Documents")
    st.markdown("Upload PDFs or paste text to add to the knowledge base.")

    col_pdf, col_text = st.columns(2)

    with col_pdf:
        st.markdown("""
        <div class="glass-card">
            <h3>📁 PDF Upload</h3>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_upload",
        )

        if uploaded_files:
            if st.button("🚀 Ingest PDFs", key="ingest_pdfs"):
                total_chunks = 0
                file_details = []
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    with st.spinner(f"Processing {file.name}..."):
                        try:
                            file_bytes = file.getvalue()
                            text = load_pdf_bytes(file_bytes, file.name)
                            if not text.strip():
                                st.warning(f"⚠️ No text extracted from {file.name}")
                                continue
                            chunks = chunk_text(text)
                            count = ingest_chunks(chunks, {"source": file.name, "type": "pdf"})
                            total_chunks += count
                            file_details.append({"name": file.name, "pages": len(text.split('\n\n')), "chunks": count})
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {e}")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                progress_bar.empty()
                if total_chunks > 0:
                    st.success(f"✅ Successfully ingested {total_chunks} chunks from {len(file_details)} PDF(s)!")
                    for fd in file_details:
                        st.markdown(f"  📄 **{fd['name']}** → {fd['chunks']} chunks")
                    updated_info = get_collection_info()
                    st.info(f"📦 Collection now has **{updated_info['vectors_count']} total vectors** | Status: **{updated_info['status']}**")

    with col_text:
        st.markdown("""
        <div class="glass-card">
            <h3>📝 Paste Text</h3>
        </div>
        """, unsafe_allow_html=True)

        text_input = st.text_area(
            "Paste document text here",
            height=200,
            key="text_input",
            placeholder="Paste your document content here...",
        )

        text_source = st.text_input(
            "Source name (optional)",
            placeholder="e.g., Research Paper, Chapter 1",
            key="text_source",
        )

        if text_input and st.button("🚀 Ingest Text", key="ingest_text"):
            with st.spinner("Processing text..."):
                metadata = {"source": text_source or "pasted_text", "type": "text"}
                count = ingest_text(text_input, metadata)
                st.success(f"✅ Ingested {count} chunks!")
                st.rerun()

    # Show collection stats after upload
    st.divider()
    info = get_collection_info()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['vectors_count']}</div>
            <div class="metric-label">Total Vectors</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['points_count']}</div>
            <div class="metric-label">Total Points</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        status_ok = 'green' in str(info['status']).lower() or info['status'] not in ('not_found',)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{'🟢 Active' if status_ok and info['vectors_count'] > 0 else '🔴 Empty' if status_ok else '🔴 Not Found'}</div>
            <div class="metric-label">Collection Status</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Ask Questions
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.markdown("## 💬 Ask Questions")
    st.markdown("Ask questions about your uploaded documents.")

    # Query input
    query = st.text_input(
        "🔍 Enter your question",
        placeholder="e.g., What are the types of machine learning?",
        key="query_input",
    )

    col_ask, col_clear = st.columns([1, 4])
    with col_ask:
        ask_clicked = st.button("🚀 Ask", key="ask_button", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_clicked and query:
        with st.spinner("🔍 Retrieving and generating answer..."):
            try:
                result = query_rag(query)
                st.session_state.chat_history.insert(0, {
                    "query": query,
                    "result": result,
                    "timestamp": time.strftime("%H:%M:%S"),
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")
                result = None

        # Compute live metrics in background
        if result and result.get("sources"):
            with st.spinner("📊 Computing metrics..."):
                try:
                    m = compute_live_metrics(query, result)
                    st.session_state.live_metrics.append(m)
                except Exception as e:
                    st.warning(f"⚠️ Metrics computation skipped: {e}")

    # Display chat history
    for i, entry in enumerate(st.session_state.chat_history):
        result = entry["result"]

        st.markdown(f"""
        <div class="glass-card fade-in">
            <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.5rem;">
                🕐 {entry['timestamp']}
            </div>
            <div style="color: #c7d2fe; font-weight: 600; margin-bottom: 0.8rem;">
                ❓ {entry['query']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(result["answer"])

        # Latency info
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            st.metric("⏱️ Retrieval", f"{result['latency_retrieval']:.3f}s")
        with col_l2:
            st.metric("🗜️ Compression", f"{result.get('latency_compression', 0):.3f}s")
        with col_l3:
            st.metric("⏱️ Generation", f"{result['latency_generation']:.3f}s")
        with col_l4:
            st.metric("⏱️ Total", f"{result['total_latency']:.3f}s")

        # Sources expander
        with st.expander(f"📎 Sources ({len(result['sources'])} chunks)", expanded=False):
            for j, src in enumerate(result["sources"]):
                score_color = "#10b981" if src["score"] > 0.7 else "#f59e0b" if src["score"] > 0.4 else "#ef4444"
                st.markdown(f"""
                <div style="background: rgba(30,30,70,0.4); border-left: 3px solid {score_color};
                            padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px;">
                    <div style="color: {score_color}; font-weight: 600; margin-bottom: 0.3rem;">
                        Chunk {j+1} — Score: {src['score']:.4f}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        {src['text'][:300]}{'...' if len(src['text']) > 300 else ''}
                    </div>
                    <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.3rem;">
                        Source: {src.get('source', 'unknown')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()




# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Metrics Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab_metrics:
    st.markdown("## 📊 Live Metrics Dashboard")
    st.markdown("Metrics are computed automatically for every question you ask in the **💬 Ask Questions** tab.")

    live = st.session_state.live_metrics

    if live:
        # ── Build aggregate results in the format report.py expects ──
        def _safe_mean(vals):
            valid = [v for v in vals if v is not None]
            return round(np.mean(valid), 4) if valid else 0

        results = {
            "num_queries": len(live),
            "avg_precision_at_k": _safe_mean([m["precision_at_k"] for m in live]),
            "avg_context_relevance": _safe_mean([m["context_relevance"] for m in live]),
            "avg_answer_relevance": _safe_mean([m["answer_relevance"] for m in live]),
            "avg_faithfulness": _safe_mean([m["faithfulness"] for m in live]),
            "avg_chunk_utilization": _safe_mean([m["chunk_utilization"] for m in live]),
            "avg_latency": _safe_mean([m["total_latency"] for m in live]),
            "avg_retrieval_latency": _safe_mean([m["latency_retrieval"] for m in live]),
            "avg_generation_latency": _safe_mean([m["latency_generation"] for m in live]),
            "per_query": live,
        }

        # ── Aggregate Metric Cards ──
        st.markdown(f"### 📈 Aggregate Scores ({len(live)} questions evaluated)")
        cols = st.columns(3)
        card_items = [
            ("Precision@K", results["avg_precision_at_k"]),
            ("Context Relevance", results["avg_context_relevance"]),
            ("Answer Relevance", results["avg_answer_relevance"]),
        ]
        for i, (name, val) in enumerate(card_items):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val:.1%}</div>
                    <div class="metric-label">{name}</div>
                </div>
                """, unsafe_allow_html=True)

        cols2 = st.columns(3)
        card_items2 = [
            ("Faithfulness", results["avg_faithfulness"], True),
            ("Chunk Utilization", results["avg_chunk_utilization"], True),
            ("Avg Latency", results["avg_latency"], False),
        ]
        for i, (name, val, is_pct) in enumerate(card_items2):
            with cols2[i]:
                if is_pct:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val:.1%}</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val:.2f}s</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # ── Charts ──
        st.markdown("### 📊 Visualizations")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(create_radar_chart(results), use_container_width=True, key="radar")
        with chart_col2:
            st.plotly_chart(create_metrics_bar_chart(results), use_container_width=True, key="bar")

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.plotly_chart(create_latency_chart(results), use_container_width=True, key="latency")
        with chart_col4:
            st.plotly_chart(create_retrieval_heatmap(results), use_container_width=True, key="heatmap")

        st.plotly_chart(create_per_query_comparison(results), use_container_width=True, key="comparison")

        # ── Per-Query Details ──
        st.divider()
        st.markdown("### 📝 Per-Query Details")

        for i, q in enumerate(live):
            with st.expander(f"Q{i+1}: {q['question']}  ({q.get('timestamp', '')})", expanded=False):
                st.markdown(f"**Answer:** {q['answer'][:500]}{'...' if len(q['answer']) > 500 else ''}")

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Precision@K", f"{q['precision_at_k']:.2%}")
                mc2.metric("Context Relevance", f"{q['context_relevance']:.2%}")
                mc3.metric("Answer Relevance", f"{q['answer_relevance']:.2%}")

                mc4, mc5, mc6 = st.columns(3)
                mc4.metric("Faithfulness", f"{q['faithfulness']:.2%}")
                mc5.metric("Chunk Utilization", f"{q['chunk_utilization']:.2%}")
                mc6.metric("Latency", f"{q['total_latency']:.2f}s")

                if q.get("faithfulness_reasoning"):
                    st.info(f"🧑‍⚖️ **Faithfulness Reasoning:** {q['faithfulness_reasoning']}")

        # ── Export ──
        st.divider()
        st.markdown("### 📥 Export Results")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            export_data = {k: v for k, v in results.items() if k != "per_query"}
            export_data["per_query"] = [{k: v for k, v in q.items() if k != "sources"} for q in live]
            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                json_data,
                "rag_metrics_report.json",
                "application/json",
                use_container_width=True,
            )

        with exp_col2:
            import io
            import csv
            output = io.StringIO()
            fieldnames = [
                "question", "precision_at_k", "context_relevance",
                "answer_relevance", "faithfulness", "chunk_utilization",
                "total_latency", "latency_retrieval", "latency_generation",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for q in live:
                writer.writerow(q)
            st.download_button(
                "📥 Download CSV",
                output.getvalue(),
                "rag_metrics_report.csv",
                "text/csv",
                use_container_width=True,
            )

        # Clear metrics button
        if st.button("🗑️ Clear Metrics", key="clear_metrics"):
            st.session_state.live_metrics = []
            st.rerun()
    else:
        st.info("💬 Ask questions in the **Ask Questions** tab — metrics will appear here automatically!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Settings
# ═══════════════════════════════════════════════════════════════════════════════

with tab_settings:
    st.markdown("## ⚙️ Settings & System Info")

    # Connection Status
    st.markdown("### 🔌 Service Connections")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("""
        <div class="glass-card">
            <h3>Qdrant Vector Database</h3>
        </div>
        """, unsafe_allow_html=True)
        qdrant_ok = check_qdrant()
        if qdrant_ok:
            st.success(f"🟢 Connected to `{config.QDRANT_HOST}:{config.QDRANT_PORT}`")
            info = get_collection_info()
            st.json(info)
        else:
            st.error(f"🔴 Cannot connect to `{config.QDRANT_HOST}:{config.QDRANT_PORT}`")

    with col_s2:
        active_model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.OLLAMA_MODEL
        st.markdown(f"""
        <div class="glass-card">
            <h3>LLM ({config.LLM_PROVIDER.upper()})</h3>
        </div>
        """, unsafe_allow_html=True)
        if config.LLM_PROVIDER == "groq":
            st.success(f"🟢 Using Groq Cloud API")
            st.markdown(f"**Active model:** `{config.GROQ_MODEL}`")
        else:
            ollama_ok = check_ollama()
            if ollama_ok:
                st.success(f"🟢 Connected to `{config.OLLAMA_BASE_URL}`")
                try:
                    r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
                    models = [m["name"] for m in r.json().get("models", [])]
                    st.markdown(f"**Available models:** {', '.join(models)}")
                except Exception:
                    pass
                st.markdown(f"**Active model:** `{config.OLLAMA_MODEL}`")
            else:
                st.error(f"🔴 Cannot connect to `{config.OLLAMA_BASE_URL}`")

    st.divider()

    # Model Configuration
    st.markdown("### 🤖 Model Configuration")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        active_model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.OLLAMA_MODEL
        active_endpoint = "Groq Cloud API" if config.LLM_PROVIDER == "groq" else config.OLLAMA_BASE_URL
        st.markdown(f"""
        <div class="glass-card">
            <h3>LLM Settings</h3>
            <p><strong>Provider:</strong> {config.LLM_PROVIDER.upper()}</p>
            <p><strong>Model:</strong> {active_model}</p>
            <p><strong>Endpoint:</strong> {active_endpoint}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="glass-card">
            <h3>Embedding Settings</h3>
            <p><strong>Model:</strong> {config.EMBEDDING_MODEL}</p>
            <p><strong>Collection:</strong> {config.QDRANT_COLLECTION}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # RAG Parameters
    st.markdown("### 🎛️ RAG Parameters")
    st.markdown(f"""
    <div class="glass-card">
        <table style="width: 100%; color: #e2e8f0;">
            <tr><td><strong>Chunk Size</strong></td><td>{config.CHUNK_SIZE} characters</td></tr>
            <tr><td><strong>Chunk Overlap</strong></td><td>{config.CHUNK_OVERLAP} characters</td></tr>
            <tr><td><strong>Top-K Results</strong></td><td>{config.TOP_K}</td></tr>
            <tr><td><strong>Relevance Threshold</strong></td><td>{config.RELEVANCE_THRESHOLD}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Architecture
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  Document    │────▶│  Chunking    │────▶│  Embedding  │
    │  Upload      │     │  Engine      │     │  Model      │
    └─────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
                                                     ▼
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  Generated  │◀────│  LLM         │◀────│  Qdrant     │
    │  Answer     │     │  (Ollama)    │     │  Vector DB  │
    └──────┬──────┘     └──────────────┘     └─────────────┘
           │
           ▼
    ┌─────────────┐     ┌──────────────┐
    │  Metrics    │────▶│  Dashboard   │
    │  Engine     │     │  & Export    │
    └─────────────┘     └──────────────┘
    ```
    """)
