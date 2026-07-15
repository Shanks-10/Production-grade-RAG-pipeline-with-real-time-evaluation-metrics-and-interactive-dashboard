<![CDATA[<div align="center">

# 🧠 RAG System — Retrieval-Augmented Generation Pipeline

**A production-grade RAG system with real-time evaluation metrics, semantic document chunking, and an interactive analytics dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-Cloud_LLM-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)

---

*Upload documents • Ask questions • Get grounded answers • Measure quality — all in one place.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#%EF%B8%8F-configuration)
- [Usage](#-usage)
- [Evaluation Metrics](#-evaluation-metrics)
- [RAG Pipeline Details](#-rag-pipeline-details)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project implements a **complete Retrieval-Augmented Generation (RAG) pipeline** that enables users to upload documents (PDF/text), ask natural-language questions, and receive **context-grounded answers** with full source attribution and quality metrics.

Unlike basic RAG implementations, this system includes:
- A **semantic chunking engine** that preserves document structure
- **LLM-as-Judge faithfulness evaluation** for automated answer quality assessment
- A **6-metric real-time evaluation dashboard** with interactive Plotly visualizations
- **Dual LLM provider support** (Groq Cloud + Ollama) for flexible deployment
- An **LLM-powered context compressor** that extracts only relevant sentences from retrieved chunks

---

## ✨ Key Features

### 📄 Intelligent Document Ingestion
- **PDF and text upload** with automatic text extraction (PyPDF2)
- **Semantic chunking** — splits documents by section headers and paragraphs, preserving coherent context
- Configurable chunk size (default 512 chars), overlap (200 chars), and a 3,000-char hard cap
- Batch embedding generation and vector upsert into Qdrant

### 💬 Context-Grounded Q&A
- Dense retrieval using **bilingual embeddings** (1024-dim, `Lajavaness/bilingual-embedding-large`)
- Cosine similarity search with configurable Top-K and relevance thresholds
- LLM answer generation with **strict context-only prompting** — prevents hallucination
- Source attribution with per-chunk relevance scores and color-coded confidence indicators

### 📊 Real-Time Evaluation Dashboard
- **6 automated quality metrics** computed on every query (see [Evaluation Metrics](#-evaluation-metrics))
- Interactive Plotly visualizations: radar charts, bar charts, latency breakdowns, retrieval heatmaps
- Per-query drill-down with faithfulness reasoning
- One-click export to **JSON and CSV**

### 🗜️ Context Compression
- LLM-powered extraction of **only query-relevant sentences** from retrieved chunks
- Reduces noise fed to the generator, improving answer precision
- Tracks compression ratios per chunk

### ⚙️ Dual LLM Provider Support
- **Groq Cloud API** — low-latency cloud inference
- **Ollama** — self-hosted, on-premise deployment
- Seamless switching via environment configuration

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │ Document  │  │   Ask    │  │ Metrics  │  │ Settings │       │
│   │  Upload   │  │Questions │  │Dashboard │  │  & Info  │       │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘       │
└────────┼──────────────┼─────────────┼────────────────────────────┘
         │              │             │
         ▼              ▼             ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────────┐
│  Ingestion  │  │  RAG        │  │   Evaluation    │
│  Pipeline   │  │  Pipeline   │  │   Engine        │
│             │  │             │  │                 │
│ • PDF Parse │  │ • Retrieve  │  │ • Precision@K   │
│ • Chunk     │  │ • Compress  │  │ • Relevance     │
│ • Embed     │  │ • Generate  │  │ • Faithfulness  │
│ • Upsert    │  │             │  │ • Utilization   │
└──────┬──────┘  └──────┬──────┘  └────────┬────────┘
       │                │                   │
       ▼                ▼                   ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────────┐
│   Qdrant    │  │  LLM        │  │   Plotly         │
│  Vector DB  │  │ Groq/Ollama │  │  Visualizations  │
└─────────────┘  └─────────────┘  └─────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit + Custom CSS | Interactive UI with glassmorphism dark theme |
| **Embeddings** | `Lajavaness/bilingual-embedding-large` | 1024-dim bilingual dense embeddings |
| **Vector Database** | Qdrant | Cosine similarity vector search |
| **LLM (Cloud)** | Groq Cloud API | Low-latency inference |
| **LLM (Local)** | Ollama | Self-hosted on-prem inference |
| **Visualizations** | Plotly | Interactive charts and heatmaps |
| **PDF Processing** | PyPDF2 | Text extraction from PDF documents |
| **ML Framework** | Sentence-Transformers, PyTorch | Embedding model loading and inference |

---

## 📁 Project Structure

```
RAG_project_antigravity/
│
├── app.py                      # Streamlit application (4-tab UI)
├── config.py                   # Environment-based configuration loader
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (LLM keys, DB config)
│
├── rag/                        # Core RAG pipeline modules
│   ├── __init__.py
│   ├── embeddings.py           # Singleton embedding model wrapper
│   ├── ingestion.py            # Document loading, semantic chunking, Qdrant upsert
│   ├── retriever.py            # Vector similarity search against Qdrant
│   ├── compressor.py           # LLM-powered context compression
│   ├── generator.py            # LLM answer generation + faithfulness judging
│   └── pipeline.py             # End-to-end RAG orchestration
│
├── metrics/                    # Evaluation and reporting
│   ├── __init__.py
│   ├── evaluator.py            # 6-metric RAG evaluation engine
│   └── report.py               # Plotly chart generators + export utilities
│
├── scripts/
│   └── generate_metrics.py     # CLI-based evaluation report generator
│
├── data/
│   └── sample/                 # Sample documents for quick testing
│       ├── quantum_computing.pdf
│       ├── ancient_civilizations.pdf
│       └── renewable_energy_climate.txt
│
├── assets/
│   └── style.css               # Dark glassmorphism theme (custom CSS)
│
└── .streamlit/
    └── config.toml             # Streamlit theme and server configuration
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Qdrant** — running locally or remotely ([install guide](https://qdrant.tech/documentation/quick-start/))
- **Groq API Key** (for cloud LLM) or **Ollama** (for local LLM)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/rag-system.git
cd rag-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Qdrant

```bash
# Using Docker (recommended)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or install natively: https://qdrant.tech/documentation/quick-start/
```

### 5. Configure Environment

Create a `.env` file in the project root:

```env
# ─── LLM Configuration ───
LLM_PROVIDER=groq                              # "groq" or "ollama"
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# For Ollama (optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# ─── Embedding Configuration ───
EMBEDDING_MODEL=Lajavaness/bilingual-embedding-large

# ─── Qdrant Configuration ───
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=rag_system

# ─── RAG Parameters ───
CHUNK_SIZE=512
CHUNK_OVERLAP=200
TOP_K=5
RELEVANCE_THRESHOLD=0.15
```

### 6. Launch the Application

```bash
streamlit run app.py
```

The app will be available at **http://localhost:8501**.

---

## ⚙️ Configuration

All settings are managed through the `.env` file and exposed via `config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | LLM backend: `groq` (cloud) or `ollama` (local) |
| `GROQ_API_KEY` | — | Your Groq Cloud API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `EMBEDDING_MODEL` | `Lajavaness/bilingual-embedding-large` | Sentence-Transformers model |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `rag_system` | Qdrant collection name |
| `CHUNK_SIZE` | `512` | Target chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks to retrieve per query |
| `RELEVANCE_THRESHOLD` | `0.15` | Minimum cosine similarity for retrieval |

---

## 📖 Usage

### 1. Upload Documents

Navigate to the **📄 Document Upload** tab:
- **PDF Upload**: Drag and drop PDF files for automatic text extraction and chunking
- **Text Paste**: Directly paste document content with an optional source name
- **Quick Start**: Click "Load Sample Data" in the sidebar to ingest the included sample documents

### 2. Ask Questions

Switch to the **💬 Ask Questions** tab:
- Enter your question in the search box
- The system retrieves relevant chunks, generates a grounded answer, and displays source attribution
- Each answer shows latency breakdown (retrieval, compression, generation)
- Expand the **Sources** section to inspect individual chunk scores

### 3. View Metrics

Go to the **📊 Metrics Dashboard** tab:
- Metrics are automatically computed for every question asked
- View aggregate scores across all queries with interactive charts
- Drill down into per-query details with faithfulness reasoning
- Export results as JSON or CSV for further analysis

### 4. CLI Evaluation

Run the standalone evaluation script:

```bash
python scripts/generate_metrics.py
```

This generates a full metrics report with progress tracking and exports to `data/metrics_report.json` and `data/metrics_report.csv`.

---

## 📊 Evaluation Metrics

The system computes **6 quality metrics** in real-time for every query:

| Metric | Method | What It Measures |
|---|---|---|
| **Precision@K** | Threshold-based (score ≥ 0.3) | Proportion of retrieved chunks that are actually relevant |
| **Context Relevance** | Cosine similarity (query ↔ chunks) | How well retrieved chunks match the query semantically |
| **Answer Relevance** | Cosine similarity (query ↔ answer) | How well the generated answer addresses the question |
| **Faithfulness** | LLM-as-Judge (0.0–1.0 scale) | Whether the answer is grounded in the retrieved context |
| **Chunk Utilization** | Token-level overlap analysis | How much of the retrieved context is actually used in the answer |
| **Latency** | Wall-clock timing per stage | End-to-end performance (retrieval + compression + generation) |

### Visualizations

- **Radar Chart** — Overview of all 5 quality metrics at a glance
- **Horizontal Bar Chart** — Comparative metric scores
- **Stacked Latency Chart** — Retrieval vs. generation time per query
- **Retrieval Heatmap** — Per-chunk relevance scores across all queries
- **Per-Query Line Chart** — Metric trends across consecutive questions

---

## 🔬 RAG Pipeline Details

### Semantic Chunking Algorithm

Unlike naive fixed-window chunking, the system uses a **3-stage semantic chunking strategy**:

1. **Header Detection** — Identifies section boundaries using pattern matching (numbered headers, title-case lines, ALL CAPS)
2. **Section Merging/Splitting** — Merges small sections for context coherence; splits sections exceeding the 3,000-char cap by paragraphs
3. **Overlap Stitching** — Applies configurable character overlap between consecutive chunks to preserve cross-boundary context

### Context Compression

Before answer generation, an optional compression step:
1. Sends each retrieved chunk to the LLM with a focused extraction prompt
2. Extracts only the sentences relevant to the query
3. Falls back to the original chunk if compression fails
4. Tracks compression ratios for observability

### Answer Generation

- Uses a **strict context-only system prompt** to prevent the LLM from using its parametric knowledge
- Formats retrieved chunks with source labels and relevance scores
- Post-processes output to fix common LLM formatting issues (bullet points, headers, whitespace)

### Faithfulness Evaluation (LLM-as-Judge)

- Sends the context, question, and answer to the LLM with a structured evaluation prompt
- Returns a JSON score (0.0–1.0) with reasoning
- Handles response parsing edge cases (code blocks, malformed JSON)

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using Python, Streamlit, Qdrant, and Sentence-Transformers**

</div>
]]>
