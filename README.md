# Financial Report Analyzer (FinSight)

A modular, source-grounded Financial Report Analyzer that ingests annual (10-K) and quarterly (10-Q) PDF filings to generate executive summaries, structured KPI extractions, and natural-language question answering with verifiable citations and source excerpts.

---

## Key Features

1. **Page-Aware Document Ingestion**: In-memory parsing via PyMuPDF with 1-based page index preservation and scan detection.
2. **Page-Scoped Sentence-Aware Chunking**: Splits text along sentence and paragraph boundaries (~850 target tokens, 120 overlap) while keeping chunks strictly within their source page for unambiguous citation provenance.
3. **Session-Isolated Ephemeral Vector Store**: Ephemeral ChromaDB indexing per active user session; no global cross-user document leakage.
4. **Google GenAI Embeddings**: Embeds chunks and search queries using Google's `gemini-embedding-2` model (768-dimensional vectors) with asymmetric retrieval prefixes (`title: ... | text: ...` and `task: question answering | query: ...`).
5. **Strictly Grounded RAG Q&A**: Answers questions solely from retrieved passages, with defense against embedded prompt injections (Phase 4).
6. **Verifiable Citations**: Every factual claim is traceable to a specific chunk and page, accompanied by verified source excerpts.
7. **Structured Financial Intelligence**: Extracts reported financial metrics (Revenue, Operating Income, Net Income, Cash Flow, Capex, Cash, Debt, EPS) without inventing figures. Missing metrics are never represented as zero.
8. **Disclosed Risks & Management Outlook**: Separates factual risk disclosures and forward-looking guidance from model speculation.
9. **Multi-Format Exports**: Download results in JSON, CSV, or Markdown formats with CSV formula injection defense.
10. **Reproducible Evaluation Benchmark**: 110-case versioned gold dataset (1.0.0) measuring retrieval Recall@k, MRR, citation precision, evidence abstention, Decimal arithmetic, and export fidelity.

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
Sentence-Aware Chunker (chunker.py: ~850 target tokens, 120 overlap)
        │
        ▼
Embedding Service (embedding_service.py: Google GenAI gemini-embedding-2)
        │
        ▼
Session-Isolated Ephemeral ChromaDB Client (vector_store.py: cosine metric)
        │
        ├───────────────────────────────┐
        ▼                               ▼
RAG Question Answering (Phase 4) Structured Extraction Service (Phase 5)
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
   GEMINI_GENERATION_MODEL=gemini-3.8-flash
   GEMINI_EMBEDDING_MODEL=gemini-embedding-2
   GEMINI_EMBEDDING_DIMENSION=768
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
| `GEMINI_GENERATION_MODEL` | `gemini-3.8-flash` | Gemini model used for synthesis, Q&A, and extraction. |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Google embedding model used for vector representations. |
| `GEMINI_EMBEDDING_DIMENSION` | `768` | Vector dimensionality (768D). |
| `MAX_UPLOAD_MB` | `25` | Maximum PDF file upload size limit in megabytes. |
| `MAX_PDF_PAGES` | `250` | Maximum page count allowed per document. |
| `CHUNK_SIZE_TOKENS` | `850` | Target token length for document chunks. |
| `CHUNK_OVERLAP_TOKENS` | `120` | Token overlap between consecutive chunks on a page. |
| `RETRIEVAL_TOP_K` | `5` | Number of context chunks retrieved per query. |
| `MIN_EVIDENCE_THRESHOLD` | `0.35` | Minimum cosine relevance score required before abstaining. |
| `LOG_LEVEL` | `INFO` | Application logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## How the In-Memory Vector Store & Isolation Work

- **Ephemeral ChromaDB**: Uses `chromadb.EphemeralClient()` purely in memory. No SQLite files, parquet tables, or vector directories are persisted to disk.
- **Session Isolation**: Each indexed report generates a unique, unpredictable collection name (e.g., `doc_<digest>_<uuid8>`). Collections are tracked in Streamlit `st.session_state`.
- **Precomputed Vectors**: Chroma is supplied with explicit precomputed vectors from `EmbeddingService`, completely bypassing any default embedding functions.
- **Reset Semantics**: Uploading a new document or clicking **Clear Report** immediately invokes `collection.delete()`, removing the ephemeral collection and freeing memory.

---

## Multi-Format Exports & Security

FinSight allows downloading active session analysis results in three formats directly from the browser without disk persistence:

1. **Structured JSON (`export_analysis_json`)**:
   - Versioned schema contract (`export_schema_version: "1.0.0"`).
   - ISO 8601 UTC export timestamp (`generated_at_utc`).
   - Full provenance: document metadata, processing metrics, coverage stats, structured metrics, risks, and outlook.
   - Non-investment advice disclaimer embedded.
2. **Spreadsheet CSVs (`export_metrics_csv`, `export_risks_csv`, `export_outlook_csv`, `export_analysis_csv`)**:
   - RFC 4180 standard formatting with dedicated headers.
   - **CSV Formula Injection Defense**: Text fields beginning with `=`, `+`, `-`, `@`, `\t`, or `\r` are prepended with `'` to prevent DDE code execution in Microsoft Excel and Google Sheets. Dedicated numeric columns are preserved for native spreadsheet calculations.
3. **Executive Markdown Report (`export_analysis_markdown`)**:
   - Polished report layout with executive summary, strategic takeaways, formatted KPI tables (with escaped table pipes `\|`), risk factors, and forward guidance.
4. **Zero Server Disk Persistence**:
   - Exports are generated strictly in-memory and delivered via Streamlit `st.download_button`. No file artifacts or customer documents are written to the server filesystem.
   - Filenames are sanitized against path traversal (`../`) and illegal filesystem characters.

---

## Running Tests

### Offline Test Suite
The complete test suite runs offline using synthetic programmatic PDFs and mocked Google GenAI client calls (requires no network connection or API billing):

```bash
.venv\Scripts\python -m pytest tests -v
```

### Optional Live Integration Smoke Test
An opt-in smoke test verifies real connectivity with the Google Gemini API when credentials are provided:

```bash
# Set environment variables:
set RUN_LIVE_GEMINI_TESTS=1
set GEMINI_API_KEY=your_actual_key

# Run live smoke test:
.venv\Scripts\python -m pytest tests/test_embedding_service.py -k test_live_gemini_embedding_smoke -v
```

---

## Evaluation Benchmark & Quality Scorecard

FinSight includes a reproducible, versioned evaluation benchmark (`evals/`) that measures retrieval, source-grounded QA, citation faithfulness, abstention, financial arithmetic, and export integrity:

### 1. Run Offline Evaluation
Runs the complete 110-assertion gold dataset offline without requiring an API key or network connectivity:

```bash
.venv\Scripts\python -m evals.runner
```

### 2. Save Benchmark Baseline
```bash
.venv\Scripts\python -m evals.runner --save-baseline evals/results/baseline_scorecard.json
```

### 3. Compare Against Baseline (Regression Detection)
```bash
.venv\Scripts\python -m evals.runner --compare evals/results/baseline_scorecard.json
```

### 4. Optional Live Evaluation
```bash
set RUN_LIVE_EVALS=1
set GEMINI_API_KEY=your_actual_key
.venv\Scripts\python -m evals.runner --mode live
```

Detailed methodology, metric definitions, and provisional quality gates are documented in [`evals/README.md`](evals/README.md).

---

## Data Retention & Known Limitations

- **Temporary In-Memory Retention**: Extracted text, page records, and Chroma collections reside in Python memory for the duration of the browser session. They are purged upon session clear or browser close.
- **Re-indexing on Upload**: Because indices are ephemeral, uploading a new file discards the prior collection and builds a fresh index for the active report.
- **Token Estimation**: Chunker token counting uses a local deterministic heuristic (~4 characters per token). It does not perform a remote API call per candidate chunk.
- **API Quotas & Rate Limits**: Gemini API calls include bounded exponential backoff (up to 3 retries) for HTTP 429 rate limits and 5xx transient server errors.

---

## Privacy & Security Notice

- **Public Filings Only**: The application is intended for publicly available corporate reports (e.g. SEC EDGAR filings).
- **External API Processing**: Document excerpts are sent to Google Gemini APIs for embedding and generation.
- **In-Memory Ephemeral Storage**: Uploaded files and vector indices are created ephemerally for the duration of the active session and are not permanently stored on disk.
- **No Hard-coded Secrets**: Credentials are read strictly from environment variables or deployment secret managers.

---

## Disclaimer

**Not Investment Advice**: This software is an experimental document analysis and research tool. It does not provide investment advice, trading recommendations, or financial audit opinions. Extracted metrics and summaries may be incomplete or subject to parsing nuances. Always verify figures directly with official regulatory filings.
