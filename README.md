# Financial Report Analyzer (FinSight)

A modular, source-grounded Financial Report Analyzer that ingests annual (10-K) and quarterly (10-Q) PDF filings to generate executive summaries, structured KPI extractions, and natural-language question answering with verifiable citations and source excerpts.

> **Notice & Non-Investment Disclaimer:**  
> FinSight is an AI-assisted financial-report analysis portfolio application that provides source-linked answers and structured financial disclosures. Outputs can be incomplete or incorrect and should be checked against the original report. **NOT investment advice or an audit opinion**. Upload only public corporate filings.

---

## Key Features

1. **Page-Aware Document Ingestion**: In-memory parsing via PyMuPDF with 1-based page index preservation and scan detection.
2. **Page-Scoped Sentence-Aware Chunking**: Splits text along sentence and paragraph boundaries (~850 target tokens, 120 overlap) while keeping chunks strictly within their source page for unambiguous citation provenance.
3. **Session-Isolated Ephemeral Vector Store**: Ephemeral ChromaDB indexing per active user session; zero cross-session or persistent disk document leakage.
4. **Google GenAI Embeddings**: Embeds chunks and search queries using Google's `gemini-embedding-2` model (768-dimensional vectors) with asymmetric retrieval prefixes (`title: ... | text: ...` and `task: question answering | query: ...`).
5. **Strictly Grounded RAG Q&A**: Answers questions solely from retrieved passages, with defense against embedded prompt injections.
6. **Verifiable Citations**: Every factual claim is traceable to a specific chunk and page, accompanied by verified source excerpts.
7. **Structured Financial Intelligence**: Extracts reported financial metrics (Revenue, Operating Income, Net Income, Cash Flow, Capex, Cash, Debt, EPS) without inventing figures. Missing metrics are never represented as zero.
8. **Disclosed Risks & Management Outlook**: Separates factual risk disclosures and forward-looking guidance from model speculation.
9. **Deterministic Decimal Arithmetic**: Python `Decimal` computes year-over-year (YoY) percentage changes, scale adjustments, and Free Cash Flow (`Operating Cash Flow - Capex`).
10. **Multi-Format In-Memory Exports**: Download results in JSON, CSV, or Markdown formats with spreadsheet formula injection defense (OWASP compliant).
11. **Demo Abuse & Cost Safeguards**: Per-session Q&A budgets (`SESSION_QA_BUDGET`), analysis budgets (`SESSION_ANALYSIS_BUDGET`), and bounded exponential backoff on Gemini API calls.
12. **Reproducible Evaluation Benchmark**: 110-case versioned gold dataset (`1.0.0`) measuring retrieval Recall@k, MRR, citation precision, evidence abstention, Decimal arithmetic, and export fidelity.

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
RAG Question Answering           Structured Extraction Service
        │                               │
        ▼                               ▼
Retrieve Passages (Top-k)       Pydantic Schema Validation
        │                               │
        ▼                               ▼
Generate Grounded Answer        Compute YoY Growth (Deterministic Python Decimal)
        │                               │
        └───────────────┬───────────────┘
                        ▼
Citations, Metrics Table, Risk/Outlook, In-Memory JSON/CSV/MD Exports
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.11, 3.12, or 3.13 (Python 3.12 is the primary cloud deployment target)
- A Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd financial-report-analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
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
   SESSION_QA_BUDGET=25
   SESSION_ANALYSIS_BUDGET=5
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Streamlit Community Cloud Deployment Guide

Follow this step-by-step checklist to deploy FinSight to **Streamlit Community Cloud**:

### 1. Confirm the Local Repository is Clean
Ensure all code, tests, and configuration changes are committed, and no untracked or secret files are present:
```bash
git status --short --branch
# Ensure output indicates a clean working tree on master
```

### 2. Create or Select the GitHub Repository
1. Log in to [GitHub](https://github.com/).
2. Create a new repository (e.g., `finsight-financial-report-analyzer`). Set visibility to **Public** (for Community Cloud hosting) or **Private** (if using connected private repo permissions).
3. Do **not** initialize with a default README or `.gitignore` (this repository already provides both).

### 3. Add Remote and Push to GitHub
```bash
git remote add origin https://github.com/<your-username>/finsight-financial-report-analyzer.git
git branch -M master
git push -u origin master
```

### 4. Open Streamlit Community Cloud
Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

### 5. Connect the Repository
1. Click **New app**.
2. Select your repository: `<your-username>/finsight-financial-report-analyzer`.
3. Set the branch to `master`.

### 6. Specify the Application Entry Point
In the **Main file path** field, enter:
```text
app.py
```

### 7. Choose Python Version in Advanced Settings
Click **Advanced settings...** and ensure the runtime is set to **Python 3.12** (supported default).

### 8. Provide Secrets in Streamlit Cloud Settings
Under **Secrets** in Advanced Settings (or in the App Dashboard under **Settings -> Secrets**), paste your configuration:
```toml
# =====================================================================
# FinSight — Streamlit Community Cloud Secrets
# =====================================================================

GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
GEMINI_GENERATION_MODEL = "gemini-3.8-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_DIMENSION = 768

# Document & Ingestion Limits
MAX_UPLOAD_MB = 25
MAX_PDF_PAGES = 250
RETRIEVAL_TOP_K = 5
MIN_EVIDENCE_THRESHOLD = 0.35

# Public Demo Abuse & Cost Safeguards
SESSION_QA_BUDGET = 25
SESSION_ANALYSIS_BUDGET = 5
LOG_LEVEL = "INFO"
```
> **Security Rule:** Never commit secrets to source control. Streamlit Community Cloud securely injects these secrets into the application runtime without writing them to disk.

### 9. Deploy and Launch the Application
Click **Deploy!** Streamlit Community Cloud will provision the container, install packages from `requirements.txt`, and start the app at `https://<your-subdomain>.streamlit.app`.

### 10. Inspect Deployment Logs
Click the **Manage app** tab in the bottom right corner of the hosted app to view build and runtime logs:
- Verify `pip install -r requirements.txt` succeeds.
- Confirm `app.py` starts without import errors.
- Confirm that no stack traces or credentials appear in the logs.

### 11. Diagnose and Troubleshoot Common Issues
- **`ModuleNotFoundError: No module named 'financial_report_analyzer'`:** The application entry point dynamically prepends `src/` to `sys.path`. If this error occurs, verify that the repository structure has not been modified.
- **`Authentication failed: Invalid or unauthorized Gemini API key`:** Ensure `GEMINI_API_KEY` in Streamlit Secrets has no extraneous quotes or whitespace.
- **`ResourceExhausted / 429`:** Free-tier Gemini keys have request-per-minute (RPM) quotas. Bounded exponential backoff handles brief spikes; for high-traffic demos, configure spend limits in Google AI Studio.
- **`File size exceeds limit`:** Documents over 25 MB or 250 pages are rejected by design to prevent cloud container memory exhaustion.

### 12. Verify Live Hosted Operations
Once launched:
1. Confirm the **API Key: Configured** indicator is green in the sidebar.
2. Upload a sample corporate report (e.g., a public 10-K).
3. Click **Analyze Report** to parse pages, generate chunks, and build the ephemeral vector store.
4. In **Ask the Report**, submit a question (e.g., *"What was total revenue in the latest fiscal year?"*) and verify that a grounded answer with verified page citations is returned.
5. Click **Extract Financials (KPIs & Risks)** to verify structured balance sheet, cash flow, risk, and guidance extraction.
6. In **Overview**, download JSON, CSV, and Markdown exports to confirm in-memory generation.
7. Click **Clear Report** to ensure all active document state and vector collections are cleared from memory.

### 13. Rotate or Update Credentials Safely
To update an API key:
1. Open [Google AI Studio](https://aistudio.google.com/) and create a new key.
2. Open the Streamlit App Settings -> **Secrets**.
3. Replace the `GEMINI_API_KEY` value and click **Save**.
4. Streamlit Cloud automatically reboots the application with the new secret.

### 14. Reboot or Unpublish
To pause or delete the hosted application:
- In the Streamlit Cloud dashboard, click the app's three dots (`⋮`) and select **Reboot app** (to refresh state) or **Delete app** (to immediately unpublish).

---

## Configuration Reference

All application settings are loaded hierarchically: environment variables take first priority, followed by Streamlit Secrets (`st.secrets`), with safe application defaults:

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
| `SESSION_QA_BUDGET` | `25` | Per-session question budget to guard against quota exhaustion. |
| `SESSION_ANALYSIS_BUDGET` | `5` | Per-session full report financial extraction budget. |
| `LOG_LEVEL` | `INFO` | Application logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Public Demo Abuse, Safeguards & Cost Controls

Publicly accessible portfolio demos calling paid or quota-limited LLM APIs require bounded controls:

1. **Pre-Processing Validation**: Files are strictly validated for extension (`.pdf`), PDF signature (`%PDF-`), non-zero byte size, 25 MB size cap, and 250 page count cap before invoking any API.
2. **Per-Session Inquiry Budgets**: Bounded session counters track user Q&A inquiries (default: 25) and full financial analyses (default: 5). When reached, clear notices instruct the visitor to reset their session.
   > **Note:** In-memory session limits manage portfolio demo usage within active browser sessions; they do not replace global provider-level budget caps.
3. **Provider Spend Controls**: The application owner can set hard spending limits and alert thresholds directly in [Google Cloud Console Billing](https://console.cloud.google.com/billing) or Google AI Studio to prevent unexpected billing.
4. **Rate Limit Handling**: Exponential backoff (up to 3 retries) gracefully handles transient 429 rate limit errors without crashing the interface.
5. **Session Reset Semantics**: Clicking **Clear Report** immediately invokes `collection.delete()`, clearing all ephemeral vectors, text chunks, and cached analysis.

---

## Memory Safety & Ephemeral Processing

- **Pure In-Memory Processing**: Uploaded PDF bytes are parsed via PyMuPDF in memory. Neither original PDF documents nor temporary text files are written to server disk storage during normal user operation.
- **Session-Isolated Ephemeral ChromaDB**: Vector collections are instantiated using `chromadb.EphemeralClient()`. Each report session generates an isolated, unpredictable collection name (e.g. `doc_<digest>_<uuid8>`).
- **Zero Cross-Session Leakage**: One visitor cannot access another visitor's vector store or document chunks.
- **In-Memory Exports**: Downloadable JSON, CSV, and Markdown files are serialized on demand in memory and streamed via Streamlit `st.download_button`.

---

## Evaluation Benchmark & Quality Scorecard

FinSight provides an offline, reproducible evaluation framework (`evals/`) that runs against a gold benchmark dataset of 3 synthetic corporate filings (`manifest.json` v1.0.0):

### 1. Run Offline Benchmark
Runs the complete 110-assertion gold dataset offline in ~0.5s with zero external API calls or billing costs:
```bash
.venv\Scripts\python -m evals.runner
```

### 2. Compare Against Baseline (Regression Detection)
```bash
.venv\Scripts\python -m evals.runner --compare evals/results/baseline_scorecard.json
```

### 3. Save Baseline Scorecard
```bash
.venv\Scripts\python -m evals.runner --save-baseline evals/results/baseline_scorecard.json
```

### 4. Optional Live Evaluation
```bash
set RUN_LIVE_EVALS=1
set GEMINI_API_KEY=your_actual_key
.venv\Scripts\python -m evals.runner --mode live
```

### Understanding Evaluation Types & Accuracy Disclaimers
It is essential to distinguish between the four levels of testing in FinSight:
- **Unit & Regression Tests (`pytest`):** 144 passing offline unit tests validating individual functions, schemas, error paths, and security defenses.
- **Offline Synthetic Benchmark (`evals.runner`):** 110 deterministic assertions running against mock embeddings and synthetic PDFs. Verifies that pipeline logic, citations, abstentions, and arithmetic meet quality gates under controlled conditions.
- **Optional Live-Model Evaluation:** Opt-in evaluation executing real Gemini API calls against synthetic filings.
- **Hosted Cloud Smoke Testing:** Live manual testing in Streamlit Community Cloud against real corporate filings.

> **Important Real-World AI Accuracy Notice:**  
> Offline benchmark pass rates demonstrate architectural soundness and deterministic contract adherence; **they do not establish 100% real-world accuracy on complex, non-standard, or novel corporate filings**. Real-world financial figures must always be verified against the original filing.

---

## Running Automated Tests

```bash
# Run all 144 offline unit and regression tests:
.venv\Scripts\python -m pytest tests -v

# Run deployment and security hardening tests:
.venv\Scripts\python -m pytest tests/test_deployment_hardening.py -v
```

---

## Known Limitations

- **OCR is Not Supported**: Scanned, image-only PDFs without extractable text layers trigger a warning. Optical Character Recognition (OCR) is intentionally out of scope for this version.
- **Single-Document Focus**: FinSight is designed for deep single-document analysis per session. Cross-document comparison across hundreds of filings simultaneously is out of scope.
- **Currency Conversion**: Year-over-year calculations require identical reporting currencies; foreign exchange conversions are intentionally withheld.

---

## License & Attribution

FinSight is built for educational, portfolio, and research purposes.
Developed using Streamlit, PyMuPDF, Google GenAI SDK (`google-genai`), ChromaDB, and Pydantic.
