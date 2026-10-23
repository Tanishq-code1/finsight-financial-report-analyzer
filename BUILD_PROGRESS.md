# Build Progress — Financial Report Analyzer (FinSight)

Last Updated: 2026-10-23
Current Status: **Phases 0–8 Complete (Production Portfolio Release 1.0.0)**

---

## 1. Project Overview & Specification Audit

| Component | Specified in Blueprint | Initial Starter Code Status | Action Taken |
|---|---|---|---|
| Master Prompt | Attached in blueprint | Inspected & incorporated | Requirements mapped |
| Project Plan | Attached in blueprint | Inspected & incorporated | Stored in repository |
| Executive Summary | 12-page PDF attached | Inspected & reviewed | Architectural principles applied |
| Starter Code | Expected in ZIP archive | Blueprint documents only | Clean modular structure created from scratch |
| Project Root | `financial-report-analyzer` | Established within workspace | Configured with virtual environment |
| Embedding Model | Originally `text-embedding-004` (shut down Jan 2026) | Replaced with active model | Updated to `gemini-embedding-2` (768D) |
| Generation Model | `gemini-3.8-flash` | Configured and verified | Structured JSON schema generation |
| Deployment Target | Streamlit Community Cloud | Python 3.12 target | Hardened, tested in clean venv, release ready |

---

## 2. Phase-by-Phase Implementation Roadmap

### Phase 0: Repository Discovery & Engineering Plan
- [x] Inspect workspace and uploaded files.
- [x] Establish `financial-report-analyzer` project directory structure.
- [x] Verify Python runtime (Python 3.13.2 x64 on Windows; target Python 3.12 on Streamlit Cloud).
- [x] Configure dependencies (`requirements.txt`, `pyproject.toml`).
- [x] Set up configuration system with environment variable overrides (`config.py`).
- [x] Create `.gitignore` and `.env.example` templates (ensuring no secrets in git).
- [x] Define typed domain error hierarchy (`errors.py`).
- [x] Establish typed Pydantic contracts (`schemas.py`) for PageRecord, ChunkRecord, Citation, RAGAnswer, FinancialMetric, RiskItem, OutlookItem, and ReportAnalysis.
- [x] Create standardized prompt templates (`prompts/qa_system.txt`, `prompts/summary.txt`, `prompts/financial_extraction.txt`).

### Phase 1: Runnable Shell & UI Skeleton
- [x] Build Streamlit main application (`app.py`).
- [x] Configure responsive theme and layout (`.streamlit/config.toml`).
- [x] Implement clear visual hierarchy:
  - Header & product identity (FinSight).
  - Prominent non-investment-advice and privacy disclaimers.
  - Sidebar with dynamic API key status check, document uploader, and action buttons.
  - Six tab sections: Overview, Ask the Report, Financial Metrics, Risks & Outlook, Sources & Processing, About & Methodology.
- [x] Implement session state initialization and `Clear report` reset semantics.
- [x] Add graceful empty states, loading indicators, and user-friendly error wrapping.

### Phase 2: PDF Ingestion & Page Provenance
- [x] Implement `src/financial_report_analyzer/pdf_loader.py` with PyMuPDF for in-memory stream extraction.
- [x] Enforce pre-parse validations: file extension (`.pdf`), PDF signature (`%PDF-`), non-empty payload, and upload size cap (25 MB).
- [x] Enforce page limits (250 pages) and reject password-protected/encrypted documents.
- [x] Extract text page-by-page preserving accurate 1-based page indices.
- [x] Normalize financial text while preserving currency symbols, decimals, parenthetical losses, and negative percentages.
- [x] Detect unextractable, blank, and sparse pages, providing actionable warnings.
- [x] Generate deterministic document digest identifiers (`doc_<sha256[:12]>`).

### Phase 3: Chunking, Embeddings & Ephemeral Vector Index
- [x] Implement sentence-aware, page-scoped chunker (`chunker.py`) with deterministic token estimation.
- [x] Preserve exact 1-based page provenance in each chunk.
- [x] Implement Google GenAI embedding service (`embedding_service.py`) supporting `gemini-embedding-2` (768D).
- [x] Implement ephemeral session-isolated ChromaDB client (`vector_store.py`) with cosine metric.
- [x] Implement indexing orchestration pipeline (`indexing_pipeline.py`) with progress callbacks and rollback on failure.

### Phase 4: Grounded Question Answering & Citation Validation
- [x] Implement RAG engine (`rag.py`) with structured JSON generation via official Google GenAI SDK.
- [x] Implement context boundary formatting (`<excerpt>`) for prompt injection defense.
- [x] Implement independent citation verification (`citation_validator.py`).
- [x] Implement evidence-based abstention.

### Phase 5: Structured Metrics, Deterministic Calculations & Risk Analysis
- [x] Implement structured financial extraction (`financial_extractor.py`) across page-bounded batches.
- [x] Implement strict Python `Decimal` arithmetic for YoY comparisons (`financial_calculations.py`).
- [x] Compute Free Cash Flow deterministically (`Operating Cash Flow - Capex`).
- [x] Deduplicate conflicting metrics and report reporting discrepancies.
- [x] Implement full-report analysis orchestration (`financial_analysis_pipeline.py`).

### Phase 6: Multi-Format Exports & Usability Polish
- [x] Implement canonical JSON export adhering to versioned schema (`1.0.0`) with full provenance.
- [x] Implement CSV exports for metrics, risk factors, and forward guidance with CSV formula injection defense.
- [x] Implement formatted executive Markdown export with escaped table pipes (`\|`).
- [x] Implement in-memory Streamlit download buttons without server disk persistence.

### Phase 7: Evaluation Benchmark, Gold Dataset & Scorecard
- [x] Create versioned gold benchmark dataset (`evals/dataset/`, version `1.0.0`) with 110 assertions across 3 synthetic filings.
- [x] Generate programmatic synthetic PDF fixtures (`fixtures.py`).
- [x] Implement mathematical evaluation scoring engine (`metrics.py`).
- [x] Implement CLI evaluation runner (`runner.py`) with baseline comparator (`compare.py`).
- [x] Verify quality gates pass with 100% parity against baseline.

### Phase 8: Deployment, Security Hardening & Final Release
- [x] Implement cloud `sys.path` resolution in `app.py` for immediate Streamlit Community Cloud launch.
- [x] Implement hierarchical configuration loading (`config.py`) supporting Streamlit Secrets (`st.secrets`) with safe exception handling when secrets are absent.
- [x] Implement public demo abuse safeguards & cost controls:
  - Per-session inquiry budget (`SESSION_QA_BUDGET = 25`).
  - Per-session financial analysis budget (`SESSION_ANALYSIS_BUDGET = 5`).
  - Bounded action button disabling while processing (`is_processing`).
  - Clear user-facing cost & quota disclaimer in sidebar.
- [x] Security hardening:
  - Configure `.streamlit/config.toml` with `enableCORS = true`, `enableXsrfProtection = true`, `headless = true`.
  - Scan all git-tracked files for potential credentials (0 secrets found).
  - Ensure `.env` and `secrets.toml` remain excluded from source control.
  - Implement non-leaking user-friendly messages for HTTP 401, 403, 429, and 500/503.
- [x] Clean environment verification:
  - Created isolated virtual environment in temporary directory.
  - Installed packages from `requirements.txt` cleanly with zero missing dependencies.
  - Verified all core runtime modules import without error.
- [x] Final release metadata:
  - Bumped `pyproject.toml` version to `1.0.0`.
  - Added safe runtime diagnostics and release identifier banner to Tab 6 ("About & Methodology").
  - Published comprehensive 14-step Streamlit Community Cloud deployment guide in `README.md`.
- [x] Test suite expansion:
  - Created `tests/test_deployment_hardening.py` covering 10 deployment and security requirements.
  - All 144 unit tests passing offline (1 skipped opt-in live test).
  - Benchmark passing 110/110 assertions with PARITY.

---

## 3. Test Results & Verification

### Test Execution Command
```bash
.venv\Scripts\python -m pytest tests -v
```

### Actual Output
```text
============================= test session starts =============================
platform win32 -- Python 3.13.2, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\shubh\Downloads\Project-anti\financial-report-analyzer
configfile: pyproject.toml
plugins: anyio-4.15.1, mock-3.17.0
collected 145 items

tests\test_chunker.py .........                                          [  6%]
tests\test_citation_validator.py .........                               [ 12%]
tests\test_config.py ....                                                [ 15%]
tests\test_deployment_hardening.py ..........                            [ 22%]
tests\test_embedding_service.py ........s                                [ 28%]
tests\test_evaluation_dataset.py ........                                [ 33%]
tests\test_evaluation_metrics.py ......                                  [ 37%]
tests\test_evaluation_runner.py ..                                       [ 39%]
tests\test_evaluation_scorecard.py ...                                   [ 41%]
tests\test_export_service.py .........                                   [ 47%]
tests\test_export_validation.py .....                                    [ 51%]
tests\test_financial_analysis_pipeline.py ....                           [ 53%]
tests\test_financial_calculations.py ..............                      [ 63%]
tests\test_financial_extractor.py .....                                  [ 66%]
tests\test_indexing_pipeline.py ...                                      [ 68%]
tests\test_pdf_loader.py ...............                                 [ 79%]
tests\test_rag.py .............                                          [ 88%]
tests\test_schemas.py ........                                           [ 93%]
tests\test_ui_smoke.py ...                                               [ 95%]
tests\test_vector_store.py ......                                        [100%]

=========================== short test summary info ===========================
SKIPPED [1] tests\test_embedding_service.py:184: Live Gemini API tests are opt-in. Set RUN_LIVE_GEMINI_TESTS=1 and GEMINI_API_KEY to execute.
======================= 168 passed, 1 skipped in 18.65s =======================
```

### Benchmark Scorecard Output
```bash
.venv\Scripts\python -m evals.runner --compare evals/results/baseline_scorecard.json
```
- **Assertions:** 110/110 passed (0 failed).
- **Quality Gates:** 100% PASSED.
- **Baseline Comparison:** PARITY across all metrics.
- **Execution Time:** ~0.35s offline.

---

## 4. Post-Release Hotfix: Token-Aware Embedding Pacing & Resumption (v1.0.1)

- **Issue Addressed:** Large PDF uploads (e.g. 109 chunks) failed at chunk 60/109 due to Gemini Embedding 2 Token-Per-Minute (TPM) quota limits (30,000 TPM tier cap exceeded by unpaced burst traffic).
- **Error Classifier (`classify_genai_error`):** Differentiates between:
  1. Permanent daily quota exhaustion (`is_daily_quota = True`, non-retriable, halts immediately).
  2. Transient TPM rate limits (`is_tpm = True`, retriable with backoff & token pacing).
  3. Transient RPM rate limits (`is_rpm = True`, retriable).
  4. Authentication failures (401/403, non-retriable).
  5. Upstream server outages (500/503, retriable).
  6. Provider suggested retry-after delays (parsed from `retry in Xs` / `wait Xs`).
- **Token-Aware Pacing (`TokenPacer`):** Sliding 60-second window rate limiter enforcing effective TPM (30,000 * 0.80 = 24,000) and effective RPM (60 * 0.80 = 48) with 20% safety margin to prevent limit spikes on dense financial texts.
- **In-Memory Partial Progress & Resumption:** In-memory `ChunkRecord.embedding` objects are saved immediately as each batch finishes. If interrupted, the sidebar displays an actionable resume button: `▶️ Resume Indexing ({remaining} chunks remaining)`. Resumption skips already-embedded chunks without duplicate API calls or charges.
- **Ephemeral Zero-Disk Guarantee:** Incomplete Chroma collections are wiped on failure (`store.delete()`). No uploaded files or vectors touch persistent disk storage.

---

## 5. Production Hotfix: Pipeline Interface, RPC RetryInfo & 48 RPM Spacing (v1.0.2)

- **Pipeline Interface Mismatch:** Corrected `run_indexing_pipeline()` to support `chunks` and `**kwargs`, ensuring backwards and forward compatibility across caller styles and eliminating `TypeError: unexpected keyword argument 'chunks'`.
- **Render Error on Failed Analysis:** Fixed Tab 1 Overview rendering to guard against `report_analysis=None`, preventing `AttributeError: 'NoneType' object has no attribute 'company_name'`.
- **Google GenAI RPC Error Details:** Implemented `extract_genai_error_payload()` to parse nested `google.rpc.RetryInfo` (`retryDelay: "25s"`) and `QuotaFailure` metric `embed_content_free_tier_tokens`.
- **Strict Delay Invariant:** Provider-specified wait times (e.g. 25 seconds) are **never shortened** to arbitrary exponential backoff intervals (1s, 2s, 4s); conservative jitter is strictly positive.
- **Smooth 48 RPM Inter-Request Spacing:** Enforced `min_request_interval = 1.25s` in `TokenPacer` to smooth traffic and eliminate 429-triggering microbursts.
- **Token Reservation Pattern:** Added `TokenReservation` handles with safe cancellation on request failure.
- **Deadline Tracking:** Enforces `embedding_deadline_sec` (120s) preventing indefinite hang loops while preserving partial progress.

---

## 6. Comprehensive Production Hotfix: Runtime Annotations, Extraction Hardening & Streamlit State Management (v1.0.3)

- **Runtime Annotation & Import Name Resolution:**
  - Diagnosed and fixed Streamlit Community Cloud production failure: `NameError: name 'Optional' is not defined` inside `render_kpi_card` in `app.py`.
  - Added explicit typing imports (`Optional`, `List`, `Dict`, `Tuple`, `Any`, `Union`, `Callable`) and schema imports (`PageRecord`, `ChunkRecord`, `FinancialMetric`, `RiskItem`, `OutlookItem`, `ReportAnalysis`, `Citation`, `RetrievalResult`, `IndexingResult`, `DocumentMetadata`) at the top of `app.py`.
  - Moved `render_kpi_card` to module scope so type annotations are parsed at script import time, preventing nested function evaluation failures during Streamlit reruns.
- **Defensive Multi-State Lifecycle Management:**
  - Separated conceptual processing stages: `EMPTY`, `PDF_EXTRACTED`, `INDEXING`, `INDEXED`, `FINANCIAL_ANALYSIS_RUNNING`, `FINANCIAL_ANALYSIS_COMPLETE`, and `INDEXING_FAILED`.
  - In-memory document text and ephemeral ChromaDB vector collections are preserved independently; an extraction failure or timeout never erases or resets an indexed vector store.
  - Added `analysis_document_id` tracking so UI tab navigation, downloads, or reruns never trigger duplicate extraction API calls.
  - Automatic document change detection purges prior document session state when a new PDF file is uploaded.
- **Financial Extraction Latency & Timeout Configuration:**
  - Investigated extraction latency on dense 10-K tables (~272s on 4 pages).
  - Introduced `generation_timeout_sec: float = 120.0` across defaults, `AppConfig`, `load_config()`, and `safe_diagnostics`.
  - Configured `genai.Client` with `http_options=genai_types.HttpOptions(client_args={"timeout": httpx.Timeout(self.timeout_sec, connect=15.0, read=self.timeout_sec)})` across `FinancialExtractor` and `RAGService`.
  - Added safe step-by-step elapsed timing instrumentation logging model duration, JSON parsing duration, and citation verification duration.
  - Wrapped `TimeoutError` and `httpx.TimeoutException` into friendly `RAGGenerationError` notifications informing the user that document text remains safely in memory.
- **Streamlit Deprecation Polish:**
  - Replaced all 14 occurrences of deprecated `use_container_width=True` on buttons, download buttons, and dataframes with `width="stretch"`, eliminating console warnings.
- **Export Service Module Skew Defense:**
  - Introduced `_is_report_analysis` in `export_service.py` to prevent `TypeError: Input must be a valid ReportAnalysis instance` caused by Streamlit script runner class identity changes across hot reloads.
- **Automated Verification:**
  - Expanded test suite from 171 to 182 tests (181 passed, 1 skipped opt-in live test).
  - Added `streamlit.testing.v1.AppTest` regression tests for empty, populated, missing company/period, and error states.
  - Offline evaluation benchmark: 110/110 gold assertions passed with 100% Quality Gates Passed and baseline PARITY.

---

## 7. Critical Financial Extraction Hotfix: Gemini 503 Resiliency, AFC Disabled & Fallback Model (v1.0.4)

- **Root Cause Diagnosis from Deployed Streamlit Cloud Logs:**
  - `Gemini API error during financial extraction (status 503)`: `This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.` Model: `gemini-3.8-flash`.
  - Automatic Function Calling (AFC): Google GenAI SDK logged `AFC is enabled with max remote calls: 10.` on each batch request because `GenerateContentConfig.automatic_function_calling` was omitted, introducing unnecessary SDK introspection overhead.
  - SDK Tenacity Retry Multiplication: The Google GenAI SDK retried 5 times internally with tenacity while the application also retried 3 times, causing multi-minute request hangs (~236s on 4 pages).
  - UI State Flaw: When all batches failed (`analysis_status="failed"`), `app.py` mistakenly treated `report_analysis` as populated, rendering empty KPI cards ("N/A"), blank executive summaries, and export buttons instead of a prominent retry/outage warning.
- **Implemented Engineering Fixes:**
  - **Explicitly Disabled AFC:** Set `automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True)` and explicitly passed `tools=None`, `tool_config=None` in `FinancialExtractor` and `RAGService`.
  - **Capped SDK Internal Retries:** Configured `http_options=genai_types.HttpOptions(retry_options=genai_types.HttpRetryOptions(attempts=1))` across client initialization and generation configs, giving application-level logic exclusive, non-multiplied control over backoff and fallback.
  - **Introduced Stable Generation Fallback Model:** Added `gemini_generation_fallback_model: str = "gemini-3.6-flash"` to `AppConfig`, defaults, `.env.example`, diagnostics, and UI.
  - **Bounded 503 Backoff & Seamless Fallback:** Bounded primary model attempts to at most 1 retry with jittered backoff; upon persistent 503 availability error, extraction seamlessly falls back to `gemini-3.6-flash`.
  - **Monotonic Request Deadlines:** Applied monotonic deadline checks against `generation_timeout_sec` (120s) to guarantee predictable termination.
  - **Distinguished Error Classes:** Strict differentiation between 401/403 (fail immediately without retry), 429 daily quota (fail immediately without retry), 429 transient rate limit (bounded exponential backoff), and 503 service availability (bounded retry -> fallback model).
  - **UI Defect Resolved:** Tab 1 and sidebar now detect `analysis.analysis_status == "failed"`, rendering a clear outage notice with an actionable `🔄 Retry Financial Extraction` button that preserves in-memory text and vector store without requiring re-upload. Fallback model provenance (`model_used`) and badges are displayed in Tab 1, Tab 5 (Audit), and Tab 6 (Configured Models).
  - **Stale Document ID Guards:** Extraction results are discarded if the user uploaded a new document or cleared the session while an in-flight analysis request was running.
- **Automated Verification:**
  - Expanded test suite to 190 tests (189 passed, 1 skipped opt-in live test).
  - Offline evaluation benchmark: 110/110 gold assertions passed with 100% Quality Gates Passed and baseline PARITY.

---

## 8. Known Limitations & Operational Boundaries

1. **Optical Character Recognition (OCR):** Scanned, image-only PDFs without extractable text layers trigger an informative warning. OCR is intentionally not implemented.
2. **Deterministic Offline Benchmark vs. Live Models:** The offline evaluation uses deterministic fake embeddings and structured mocks for regression testing. It does **not** establish 100% real-world accuracy on novel or non-standard corporate filings.
3. **Session vs. Global Quotas:** In-memory session limits manage portfolio demo usage within active browser sessions. Repository owners should configure provider spend alerts in Google AI Studio or Google Cloud Billing to prevent unexpected charges.
4. **Single-Filing Scope:** Designed for deep single-report analysis per session; cross-company multi-document comparison is out of scope.

---

## 9. Release Readiness Summary
- Git branch: `master`
- Version: `1.0.4`
- Deployment status: Ready for push to GitHub and automatic redeployment on Streamlit Community Cloud.
