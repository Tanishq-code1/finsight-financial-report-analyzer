# Financial Report Analyzer (FinSight)

A modular, source-grounded Financial Report Analyzer that ingests annual (10-K) and quarterly (10-Q) PDF filings to generate executive summaries, structured KPI extractions, and natural-language question answering with verifiable citations and source excerpts.

---

## Key Features

1. **Page-Aware Document Ingestion**: In-memory parsing via PyMuPDF with 1-based page index preservation and scan detection.
2. **Session-Isolated Ephemeral Vector Store**: Ephemeral ChromaDB indexing per user session; no global cross-user document leakage.
3. **Strictly Grounded RAG Q&A**: Answers questions solely from retrieved passages, with defense against embedded prompt injections.
4. **Verifiable Citations**: Every factual claim is traceable to a specific chunk and page, accompanied by verified source excerpts.
5. **Structured Financial Intelligence**: Extracts reported financial metrics (Revenue, Operating Income, Net Income, Cash Flow, Capex, Cash, Debt, EPS) without inventing figures. Missing metrics are never represented as zero.
6. **Disclosed Risks & Management Outlook**: Separates factual risk disclosures and forward-looking guidance from model speculation.
7. **Multi-Format Exports**: Download results in JSON, CSV, or Markdown formats.

---

## Architecture Overview

```text
Browser / Streamlit UI (app.py)
        │
        ▼
Upload Validation & Session Controller
        │
        ▼
PDF Parser (PyMuPDF) ─────────────────► Parse Warnings / Page Counts
        │
        ▼
Page Records (1-based page numbers + text)
        │
        ▼
Section & Sentence-Aware Chunker (700–1,000 tokens)
        │
        ▼
Embedding Service (Google GenAI: text-embedding-004)
        │
        ▼
Session-Isolated Ephemeral ChromaDB Client
        │
        ├───────────────────────────────┐
        ▼                               ▼
RAG Question Answering          Structured Extraction Service
        │                               │
        ▼                               ▼
Retrieve Passages (Top-k)       Pydantic Validation
        │                               │
        ▼                               ▼
Generate Grounded Answer        Compute YoY Growth (Deterministic Python)
        │                               │
        └───────────────┬───────────────┘
                        ▼
Citations, Metrics Table, Risk/Outlook, JSON/CSV Exports
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.11, 3.12, or 3.13
- A Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd financial-report-analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to supply your `GEMINI_API_KEY`:
   ```dotenv
   GEMINI_API_KEY=your_actual_gemini_api_key
   GEMINI_GENERATION_MODEL=gemini-2.5-flash
   GEMINI_EMBEDDING_MODEL=text-embedding-004
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Configuration Reference

All settings can be configured via environment variables or `.streamlit/secrets.toml`:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *None* | Required for AI operations (Embeddings, RAG, Extraction). |
| `GEMINI_GENERATION_MODEL` | `gemini-2.5-flash` | Gemini model used for synthesis, Q&A, and extraction. |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Model used for vector embeddings. |
| `MAX_UPLOAD_MB` | `25` | Maximum PDF file upload size limit in megabytes. |
| `MAX_PDF_PAGES` | `250` | Maximum page count allowed per document. |
| `CHUNK_SIZE_TOKENS` | `850` | Target token length for document chunks. |
| `CHUNK_OVERLAP_TOKENS` | `120` | Token overlap between consecutive chunks. |
| `RETRIEVAL_TOP_K` | `5` | Number of context chunks retrieved per query. |
| `MIN_EVIDENCE_THRESHOLD` | `0.35` | Minimum cosine relevance score required before abstaining. |
| `LOG_LEVEL` | `INFO` | Application logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Running Tests

Unit tests are written with `pytest` and mock external API dependencies so tests execute quickly and reliably offline without incurring API billing:

```bash
pytest tests -v
```

---

## Privacy & Security Notice

- **Public Filings Only**: The application is intended for publicly available corporate reports (e.g. SEC EDGAR filings).
- **External API Processing**: Document excerpts are sent to Google Gemini APIs for embedding and generation.
- **In-Memory Ephemeral Storage**: Uploaded files and vector indices are created ephemerally for the duration of the active session and are not permanently stored on disk.
- **No Hard-coded Secrets**: Credentials are read strictly from environment variables or deployment secret managers.

---

## Disclaimer

**Not Investment Advice**: This software is an experimental document analysis and research tool. It does not provide investment advice, trading recommendations, or financial audit opinions. Extracted metrics and summaries may be incomplete or subject to parsing nuances. Always verify figures directly with official regulatory filings.
