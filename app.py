"""Financial Report Analyzer — Main Streamlit Application."""

import logging
import streamlit as st
import pandas as pd

from financial_report_analyzer.config import load_config, setup_logging
from financial_report_analyzer.errors import (
    FinancialAnalyzerError,
    DocumentValidationError,
    PDFParsingError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    RAGGenerationError,
    CitationValidationError,
    ConfigurationError,
)
from financial_report_analyzer.pdf_loader import load_pdf
from financial_report_analyzer.chunker import chunk_pages, ChunkerConfig
from financial_report_analyzer.embedding_service import EmbeddingService
from financial_report_analyzer.indexing_pipeline import run_indexing_pipeline
from financial_report_analyzer.rag import RAGService
from financial_report_analyzer.citation_validator import CitationValidator
from financial_report_analyzer.financial_analysis_pipeline import run_financial_analysis_pipeline
from financial_report_analyzer.financial_extractor import FinancialExtractor
from financial_report_analyzer.financial_calculations import (
    calculate_yoy_comparison,
    calculate_free_cash_flow,
    parse_financial_number,
)

# Initialize logging
setup_logging()
logger = logging.getLogger("financial_report_analyzer.app")

# Page Configuration
st.set_page_config(
    page_title="Financial Report Analyzer | FinSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished financial tool interface
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .disclaimer-badge {
        background-color: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.3);
        border-radius: 6px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #93C5FD;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .empty-card {
        background-color: #1E293B;
        border: 1px dashed #475569;
        border-radius: 8px;
        padding: 2.5rem;
        text-align: center;
        color: #94A3B8;
        margin: 1rem 0;
    }
    .retrieval-card {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .citation-tag {
        background-color: #0F172A;
        border: 1px solid #3B82F6;
        color: #60A5FA;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .answer-card {
        background-color: #1E293B;
        border: 1px solid #3B82F6;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .citation-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-left: 4px solid #10B981;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .verified-badge {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .abstention-card {
        background-color: #1E293B;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .kpi-metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        height: 100%;
    }
    .kpi-title {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }
    .kpi-badge-pos {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        font-size: 0.8rem;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .kpi-badge-neg {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        font-size: 0.8rem;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .kpi-badge-neu {
        background-color: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        font-size: 0.8rem;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .conflict-banner {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        color: #FCA5A5;
        font-size: 0.9rem;
    }
    .tag-pill {
        background-color: #0F172A;
        border: 1px solid #475569;
        border-radius: 4px;
        padding: 0.15rem 0.4rem;
        font-size: 0.75rem;
        color: #CBD5E1;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize active session variables."""
    if "document_id" not in st.session_state:
        st.session_state.document_id = None
    if "report_metadata" not in st.session_state:
        st.session_state.report_metadata = None
    if "page_records" not in st.session_state:
        st.session_state.page_records = None
    if "ingestion_result" not in st.session_state:
        st.session_state.ingestion_result = None
    if "chunk_records" not in st.session_state:
        st.session_state.chunk_records = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "indexing_result" not in st.session_state:
        st.session_state.indexing_result = None
    if "is_indexed" not in st.session_state:
        st.session_state.is_indexed = False
    if "report_analysis" not in st.session_state:
        st.session_state.report_analysis = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "status_message" not in st.session_state:
        st.session_state.status_message = None


def clear_report_session() -> None:
    """Clear active report data and reset session state cleanly."""
    logger.info("Resetting session state for active user.")
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        try:
            st.session_state.vector_store.delete()
        except Exception as e:
            logger.warning("Error deleting vector store during reset: %s", str(e))

    st.session_state.document_id = None
    st.session_state.report_metadata = None
    st.session_state.page_records = None
    st.session_state.ingestion_result = None
    st.session_state.chunk_records = None
    st.session_state.vector_store = None
    st.session_state.indexing_result = None
    st.session_state.is_indexed = False
    st.session_state.report_analysis = None
    st.session_state.chat_history = []
    st.session_state.search_results = None
    st.session_state.is_processing = False
    st.session_state.status_message = None
    st.rerun()


def main() -> None:
    """Streamlit application entrypoint."""
    init_session_state()
    config = load_config()

    # Sidebar
    with st.sidebar:
        st.title("FinSight 📊")
        st.caption("AI-Powered Financial Report Analysis")
        st.divider()

        # API Key Status Banner
        if config.has_api_key:
            st.success("API Key: Configured", icon="✅")
        else:
            st.warning("API Key: Not Configured", icon="⚠️")
            st.caption(
                "Gemini API key is required for embeddings, Q&A, and metric extraction. "
                "Set `GEMINI_API_KEY` in `.env` or Streamlit Secrets."
            )

        st.subheader("Document Ingestion")
        uploaded_file = st.file_uploader(
            "Upload Financial Report (PDF)",
            type=["pdf"],
            help=f"Accepts public annual/quarterly reports up to {config.max_upload_mb} MB and {config.max_pdf_pages} pages.",
            key="pdf_uploader",
        )

        has_valid_upload = False
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            st.caption(f"📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

            if file_size_mb > config.max_upload_mb:
                st.error(f"File size exceeds maximum upload limit of {config.max_upload_mb} MB.")
            elif len(file_bytes) == 0:
                st.error("Uploaded file is empty (0 bytes).")
            else:
                has_valid_upload = True

        col_analyze, col_clear = st.columns(2)
        with col_analyze:
            analyze_clicked = st.button(
                "Analyze Report",
                type="primary",
                use_container_width=True,
                disabled=not has_valid_upload,
                help="Extract text, generate chunks, and build ephemeral vector index.",
            )
        with col_clear:
            clear_clicked = st.button("Clear Report", on_click=clear_report_session, use_container_width=True)

        # Explicit Financial Disclosures Analysis action button
        has_extracted_doc = bool(st.session_state.page_records)
        fin_btn_clicked = st.button(
            "📊 Extract Financials (KPIs & Risks)",
            type="secondary",
            use_container_width=True,
            disabled=not has_extracted_doc or st.session_state.is_processing,
            help="Extract structured KPIs, balance sheet items, disclosed risks, management outlook, and calculate YoY changes.",
        )

        if fin_btn_clicked and has_extracted_doc:
            if not config.has_api_key:
                st.error("Gemini API key is required to analyze financials. Please set GEMINI_API_KEY in .env or settings.")
            else:
                st.session_state.is_processing = True
                p_bar = st.progress(0.0)
                s_text = st.empty()

                def update_fin_p(frac: float, msg: str) -> None:
                    p_bar.progress(frac)
                    s_text.text(msg)

                try:
                    analysis = run_financial_analysis_pipeline(
                        pages=st.session_state.page_records,
                        metadata=st.session_state.report_metadata,
                        config=config,
                        progress_callback=update_fin_p,
                    )
                    st.session_state.report_analysis = analysis
                    s_text.text("Financial extraction complete!")
                    st.success(
                        f"Financial Analysis Ready: Extracted {len(analysis.metrics)} metrics, "
                        f"{len(analysis.risks)} risks, and {len(analysis.outlook)} forward outlook items."
                    )
                except Exception as fin_err:
                    logger.error("Financial analysis pipeline error: %s", str(fin_err), exc_info=True)
                    st.error(f"Financial Analysis Error: {str(fin_err)}")
                finally:
                    st.session_state.is_processing = False

        if uploaded_file and analyze_clicked and has_valid_upload:
            st.session_state.is_processing = True
            # Invalidate any prior financial analysis from previous document
            st.session_state.report_analysis = None

            # Safely delete any previous session vector collection
            if st.session_state.vector_store is not None:
                try:
                    st.session_state.vector_store.delete()
                except Exception as del_err:
                    logger.debug("Previous collection cleanup: %s", str(del_err))
                st.session_state.vector_store = None
                st.session_state.is_indexed = False

            # Progress widgets
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                # Step 1: PDF Extraction
                status_text.text("1/3: Extracting text page-by-page...")
                progress_bar.progress(0.2)
                result = load_pdf(
                    pdf_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    config=config,
                )
                st.session_state.document_id = result.metadata.document_id
                st.session_state.report_metadata = result.metadata
                st.session_state.page_records = result.pages
                st.session_state.ingestion_result = result

                # Step 2: Sentence-Aware Chunking
                status_text.text("2/3: Generating sentence-aware chunks...")
                progress_bar.progress(0.4)
                chunker_cfg = ChunkerConfig(
                    target_tokens=config.chunk_size_tokens,
                    overlap_tokens=config.chunk_overlap_tokens,
                )
                chunks = chunk_pages(result.pages, config=chunker_cfg)
                st.session_state.chunk_records = chunks

                # Step 3: Embeddings & Ephemeral ChromaDB Vector Indexing
                if config.has_api_key:
                    status_text.text("3/3: Generating embeddings & vector index...")

                    def update_progress(desc: str, frac: float) -> None:
                        status_text.text(f"3/3: {desc}")
                        progress_bar.progress(min(0.95, 0.4 + 0.55 * frac))

                    pipeline_output = run_indexing_pipeline(
                        pages=result.pages,
                        config=config,
                        progress_callback=update_progress,
                    )
                    st.session_state.vector_store = pipeline_output.vector_store
                    st.session_state.indexing_result = pipeline_output.indexing_result
                    st.session_state.is_indexed = True

                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    st.success(
                        f"Indexed {result.metadata.total_pages} pages ({len(chunks)} chunks, "
                        f"{pipeline_output.indexing_result.embedding_dimension}D vectors)!"
                    )
                else:
                    st.session_state.is_indexed = False
                    progress_bar.progress(1.0)
                    status_text.text("PDF extracted (Vector indexing paused)")
                    st.info(
                        f"Parsed {result.metadata.total_pages} pages into {len(chunks)} chunks. "
                        "Vector indexing paused: `GEMINI_API_KEY` is not configured."
                    )

                if result.has_warnings:
                    st.warning(
                        f"{len(result.metadata.warnings)} parsing notice(s) detected. Check 'Sources & Processing' tab."
                    )

            except (DocumentValidationError, PDFParsingError, EmbeddingError, VectorStoreError) as err:
                logger.error("Pipeline error: %s", str(err))
                st.error(f"Processing Error: {err.user_friendly_message}")
            except Exception as err:
                logger.critical("Unexpected pipeline crash: %s", str(err), exc_info=True)
                st.error("An unexpected error occurred during report processing.")
            finally:
                st.session_state.is_processing = False

        st.divider()
        st.subheader("Runtime Configuration")
        st.caption(f"**Generation Model:** `{config.gemini_generation_model}`")
        st.caption(f"**Embedding Model:** `{config.gemini_embedding_model}` ({config.embedding_dimension}D)")
        st.caption(f"**Max Size:** `{config.max_upload_mb} MB` | **Max Pages:** `{config.max_pdf_pages}`")
        st.caption(f"**Chunk Size:** `{config.chunk_size_tokens} tokens` (overlap {config.chunk_overlap_tokens})")
        st.caption(f"**Retrieval Top-K:** `{config.retrieval_top_k}`")

    # Main Canvas Header
    st.markdown('<div class="main-header">Financial Report Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Source-grounded RAG intelligence for 10-K, 10-Q, and annual reports.</div>',
        unsafe_allow_html=True,
    )

    # Privacy & Non-Investment Advice Disclaimers
    st.markdown(
        """
        <div class="disclaimer-badge">
            <strong>Notice & Disclaimer:</strong> FinSight is an informational research aid and document-understanding tool, 
            <strong>NOT investment advice or an audit opinion</strong>. Upload only public corporate filings. 
            All extracted insights should be cross-verified with official regulatory filings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Navigation Tabs
    tab_overview, tab_qa, tab_metrics, tab_risks, tab_sources, tab_about = st.tabs(
        [
            "📑 Overview",
            "💬 Ask the Report",
            "📈 Financial Metrics",
            "⚠️ Risks & Outlook",
            "🔍 Sources & Processing",
            "ℹ️ About & Methodology",
        ]
    )

    # Tab 1: Overview
    with tab_overview:
        st.subheader("Executive Summary & Overview")
        if not st.session_state.report_metadata:
            st.markdown(
                """
                <div class="empty-card">
                    <h4>No Active Financial Report</h4>
                    <p>Upload a public company report (PDF) in the sidebar and click <strong>Analyze Report</strong> to generate an executive summary, strategic takeaways, and key operational drivers.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            meta = st.session_state.report_metadata
            chunks_count = len(st.session_state.chunk_records) if st.session_state.chunk_records else 0
            index_badge = "✅ Vector Index Ready" if st.session_state.is_indexed else "⚠️ Text Extracted (Index Paused)"
            analysis = st.session_state.report_analysis

            status_badge = "⏳ Awaiting Extraction"
            if analysis:
                if analysis.analysis_status == "complete":
                    status_badge = f"✅ Complete Analysis ({analysis.pages_analyzed}/{analysis.total_pages} pages)"
                elif analysis.analysis_status == "partial":
                    status_badge = f"⚠️ Partial Analysis ({analysis.batches_succeeded} ok, {analysis.batches_failed} failed)"
                else:
                    status_badge = "❌ Extraction Failed"

            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0;">{analysis.company_name or 'Document'}: {meta.filename}</h4>
                        <span class="tag-pill" style="font-weight: 600;">{status_badge}</span>
                    </div>
                    <p style="margin-top: 0.5rem; color: #94A3B8; font-size: 0.9rem;">
                        Period: <strong>{analysis.fiscal_period if analysis and analysis.fiscal_period else 'Report Period'}</strong> | 
                        Pages: <strong>{meta.total_pages}</strong> | 
                        Chunks: <strong>{chunks_count}</strong> | 
                        Vector Store: <strong>{index_badge}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if analysis:
                # Display reporting discrepancies if any
                if analysis.data_conflicts:
                    st.markdown(
                        f"""
                        <div class="conflict-banner">
                            <strong>⚠️ Reporting Conflicts Detected ({len(analysis.data_conflicts)}):</strong>
                            <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
                                {''.join(f'<li>{c}</li>' for c in analysis.data_conflicts)}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Key KPI Highlight Cards (4 Columns)
                rev = next((m for m in analysis.metrics if m.normalized_name == "revenue" or "revenue" in m.name.lower()), None)
                ni = next((m for m in analysis.metrics if m.normalized_name == "net_income" or "net income" in m.name.lower()), None)
                ocf = next((m for m in analysis.metrics if m.normalized_name == "operating_cash_flow" or "operating cash" in m.name.lower()), None)
                fcf = next((m for m in analysis.metrics if m.name == "Free Cash Flow"), None)

                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

                def render_kpi_card(title: str, m: Optional[FinancialMetric]) -> str:
                    if not m or m.value is None:
                        return f"""
                        <div class="kpi-metric-box">
                            <div class="kpi-title">{title}</div>
                            <div class="kpi-value" style="color: #64748B;">N/A</div>
                            <small style="color: #94A3B8;">Not reported in filing</small>
                        </div>
                        """
                    curr_str = f"{m.currency} " if m.currency else "$"
                    unit_str = f" {m.unit_scale}" if m.unit_scale else ""
                    val_str = f"{curr_str}{m.value:,}{unit_str}" if isinstance(m.value, (int, float)) else f"{curr_str}{m.value}{unit_str}"

                    # YoY Badge
                    badge_html = ""
                    if m.yoy_percentage_change is not None:
                        if m.yoy_percentage_change > 0:
                            badge_html = f'<span class="kpi-badge-pos">+{m.yoy_percentage_change:.2f}% YoY</span>'
                        elif m.yoy_percentage_change < 0:
                            badge_html = f'<span class="kpi-badge-neg">{m.yoy_percentage_change:.2f}% YoY</span>'
                        else:
                            badge_html = f'<span class="kpi-badge-neu">0.00% YoY</span>'
                    elif m.yoy_calculation_note and "YoY" in m.yoy_calculation_note:
                        badge_html = f'<span class="kpi-badge-neu">{m.yoy_calculation_note}</span>'

                    page_str = f'<span class="citation-tag" style="float: right;">p. {m.source_page}</span>' if m.source_page else ""

                    return f"""
                    <div class="kpi-metric-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                            <span class="kpi-title">{title}</span>
                            {page_str}
                        </div>
                        <div class="kpi-value">{val_str}</div>
                        <div style="margin-top: 0.4rem;">{badge_html}</div>
                    </div>
                    """

                with kpi_col1:
                    st.markdown(render_kpi_card("Total Revenue", rev), unsafe_allow_html=True)
                with kpi_col2:
                    st.markdown(render_kpi_card("Net Income", ni), unsafe_allow_html=True)
                with kpi_col3:
                    st.markdown(render_kpi_card("Operating Cash Flow", ocf), unsafe_allow_html=True)
                with kpi_col4:
                    st.markdown(render_kpi_card("Free Cash Flow (Python)", fcf), unsafe_allow_html=True)

                st.markdown("### 📝 Executive Overview")
                st.markdown(
                    f"""
                    <div class="answer-card">
                        <p style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 0;">{analysis.executive_summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if analysis.key_takeaways:
                    st.markdown("**Key Strategic Takeaways:**")
                    for t in analysis.key_takeaways:
                        st.markdown(f"- {t}")

                if analysis.limitations:
                    with st.expander(f"ℹ️ Scope & Extraction Limitations ({len(analysis.limitations)})", expanded=False):
                        for lim in analysis.limitations:
                            st.caption(f"• {lim}")
            else:
                st.markdown(
                    """
                    <div class="empty-card">
                        <h4>Financial Disclosures Not Yet Extracted</h4>
                        <p>Document text is ready. Click <strong>Extract Financial Disclosures</strong> to extract key KPIs, itemize risk factors, and compute deterministic YoY growth.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("📊 Extract Financial Disclosures Now", type="primary", key="btn_run_fin_tab1"):
                    if not config.has_api_key:
                        st.error("Gemini API key is required to analyze financials. Please set GEMINI_API_KEY in .env or settings.")
                    else:
                        with st.spinner("Analyzing document batches and calculating YoY metrics in Python..."):
                            try:
                                analysis_res = run_financial_analysis_pipeline(
                                    pages=st.session_state.page_records,
                                    metadata=st.session_state.report_metadata,
                                    config=config,
                                )
                                st.session_state.report_analysis = analysis_res
                                st.rerun()
                            except Exception as run_err:
                                st.error(f"Extraction failed: {str(run_err)}")

    # Tab 2: Ask the Report (Source-Grounded RAG Q&A & Diagnostic Vector Search)
    with tab_qa:
        st.subheader("Ask the Report (Source-Grounded Q&A)")
        st.caption(
            "Ask natural-language questions about the active filing. FinSight answers strictly from retrieved "
            "evidence with verifiable 1-based page citations and explicit evidence abstention."
        )

        if not st.session_state.report_metadata:
            st.markdown(
                """
                <div class="empty-card">
                    <h4>Document Not Yet Loaded</h4>
                    <p>Upload and process a public company financial report (PDF) in the sidebar to enable source-grounded question answering.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not st.session_state.is_indexed:
            st.markdown(
                """
                <div class="empty-card">
                    <h4>Vector Index Not Available</h4>
                    <p>Document text is extracted, but vector retrieval and grounded Q&A require configuring <code>GEMINI_API_KEY</code>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Suggested Questions Quick Bar
            st.markdown("**Suggested Research Questions:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                q1 = st.button("📊 Revenue Drivers", use_container_width=True)
            with col2:
                q2 = st.button("💵 Cash Flow vs Net Income", use_container_width=True)
            with col3:
                q3 = st.button("⚠️ Disclosed Risk Factors", use_container_width=True)
            with col4:
                q4 = st.button("🔮 Forward Guidance", use_container_width=True)

            preset_question = None
            if q1:
                preset_question = "What were the main revenue drivers and growth rates in the latest fiscal period?"
            elif q2:
                preset_question = "How did operating cash flow compare to net income, and what were the main drivers of the difference?"
            elif q3:
                preset_question = "What are the primary operational, macroeconomic, and regulatory risk factors disclosed?"
            elif q4:
                preset_question = "What forward-looking statements or management guidance were provided for upcoming periods?"

            # Input form
            with st.form(key="rag_qa_form", clear_on_submit=False):
                user_question = st.text_input(
                    "Ask a question about this financial report:",
                    value=preset_question or "",
                    placeholder="e.g., What were the principal drivers of revenue growth in fiscal year 2024?",
                    key="rag_question_input",
                )
                col_btn1, col_btn2 = st.columns([5, 1])
                with col_btn1:
                    submit_qa = st.form_submit_button("Ask Question", type="primary")
                with col_btn2:
                    clear_history = st.form_submit_button("Clear History")

            if clear_history:
                st.session_state.chat_history = []
                st.rerun()

            active_q = None
            if submit_qa and user_question.strip():
                active_q = user_question.strip()
            elif preset_question:
                active_q = preset_question

            if active_q:
                with st.spinner("Analyzing report evidence and generating grounded response..."):
                    try:
                        embedder = EmbeddingService(config=config)
                        rag_service = RAGService(config=config)
                        answer, retrieved = rag_service.answer_question(
                            question=active_q,
                            vector_store=st.session_state.vector_store,
                            embedding_service=embedder,
                            document_id=st.session_state.document_id,
                            top_k=config.retrieval_top_k,
                            min_evidence_threshold=config.min_evidence_threshold,
                        )
                        # Prepend to chat history for chronological reverse display
                        st.session_state.chat_history.insert(
                            0,
                            {
                                "question": active_q,
                                "answer": answer,
                                "retrieved_chunks": retrieved,
                            },
                        )
                    except (ConfigurationError, RetrievalError, RAGGenerationError) as rag_err:
                        logger.error("RAG error: %s", str(rag_err))
                        st.error(f"Question Answering Error: {rag_err.user_friendly_message}")
                    except Exception as exc:
                        logger.critical("Unexpected error in RAG Q&A: %s", str(exc), exc_info=True)
                        st.error("An unexpected error occurred while processing your question.")

            # Render Chat History
            if st.session_state.chat_history:
                st.markdown("### Grounded Answers")
                for idx, entry in enumerate(st.session_state.chat_history):
                    q = entry["question"]
                    ans = entry["answer"]
                    retrieved_chunks = entry["retrieved_chunks"]

                    st.markdown(f"#### ❓ {q}")

                    if ans.insufficient_evidence:
                        st.markdown(
                            f"""
                            <div class="abstention-card">
                                <h5 style="color: #F59E0B; margin-top: 0;">⚠️ Insufficient Evidence in Report</h5>
                                <p style="margin-bottom: 0.5rem; font-size: 0.95rem;">{ans.answer}</p>
                                {f'<small style="color: #D97706;"><strong>Reason:</strong> {ans.abstention_reason}</small>' if ans.abstention_reason else ''}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if ans.limitations:
                            for lim in ans.limitations:
                                st.caption(f"• {lim}")
                    else:
                        st.markdown(
                            f"""
                            <div class="answer-card">
                                <p style="font-size: 1.02rem; line-height: 1.6; color: #F8FAFC; margin-bottom: 0; white-space: pre-wrap;">{ans.answer}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if ans.limitations:
                            with st.expander("ℹ️ Data Caveats & Scope Limitations", expanded=False):
                                for lim in ans.limitations:
                                    st.markdown(f"- {lim}")

                        # Citations Panel
                        if ans.citations:
                            st.markdown(f"**📚 Verifiable Source Citations ({len(ans.citations)}):**")
                            for cit in ans.citations:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                                            <span class="citation-tag">📄 Page {cit.page_number}</span>
                                            <span class="verified-badge">✓ Verified in Report Text</span>
                                        </div>
                                        <p style="font-style: italic; color: #E2E8F0; margin: 0.4rem 0; font-size: 0.93rem;">"{cit.quote}"</p>
                                        <div style="font-size: 0.78rem; color: #64748B;">
                                            Source: <code>{cit.source_filename or 'Report PDF'}</code> | Chunk ID: <code>{cit.chunk_id}</code>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Collapsible diagnostic inspection of retrieved candidate passages
                    if retrieved_chunks:
                        with st.expander(f"🔍 Inspect {len(retrieved_chunks)} Retrieved Passages for this Query", expanded=False):
                            for rank, r in enumerate(retrieved_chunks, start=1):
                                st.markdown(
                                    f"""
                                    <div class="retrieval-card">
                                        <span class="citation-tag">Rank #{rank} | Page {r.page_number}</span>
                                        <span style="float: right; font-size: 0.8rem; color: #94A3B8;">
                                            Distance: <strong>{r.distance}</strong> | Similarity Metric: <strong>{r.similarity}</strong>
                                        </span>
                                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #E2E8F0; white-space: pre-wrap;">{r.text}</p>
                                        <small style="color: #64748B;">Chunk ID: <code>{r.chunk_id}</code></small>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    st.divider()

            # Standalone Vector Diagnostic Search Tool (Preserving Phase 3 tool)
            with st.expander("🛠️ Standalone Vector Search Diagnostic Tool (Phase 3)", expanded=False):
                st.caption("Inspect raw cosine distances and vector similarity matches without LLM generation.")
                diag_query = st.text_input("Enter diagnostic search phrase:", key="standalone_vector_diag_input")
                if st.button("Run Diagnostic Search", key="btn_standalone_vector_diag"):
                    if diag_query.strip():
                        with st.spinner("Searching vector index..."):
                            try:
                                embedder = EmbeddingService(config=config)
                                q_vec = embedder.embed_query(diag_query.strip())
                                diag_results = st.session_state.vector_store.query(
                                    query_embedding=q_vec,
                                    top_k=config.retrieval_top_k,
                                )
                                st.markdown(f"**Found {len(diag_results)} passages:**")
                                for rank, r in enumerate(diag_results, start=1):
                                    st.markdown(
                                        f"""
                                        <div class="retrieval-card">
                                            <span class="citation-tag">Rank #{rank} | Page {r.page_number}</span>
                                            <span style="float: right; font-size: 0.8rem; color: #94A3B8;">
                                                Cosine Distance: <strong>{r.distance}</strong> | Similarity: <strong>{r.similarity}</strong>
                                            </span>
                                            <p style="margin-top: 0.4rem; font-size: 0.9rem; color: #CBD5E1;">{r.text}</p>
                                            <small style="color: #64748B;">Chunk ID: <code>{r.chunk_id}</code></small>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                            except Exception as diag_err:
                                st.error(f"Diagnostic retrieval failed: {str(diag_err)}")

    # Tab 3: Financial Metrics
    with tab_metrics:
        st.subheader("Key Financial Performance Indicators (KPIs)")
        st.caption(
            "Structured extraction of revenue, profitability, cash flow, debt, and balance-sheet metrics with Python-computed YoY comparisons."
        )

        analysis = st.session_state.report_analysis
        if not analysis:
            st.markdown(
                """
                <div class="empty-card">
                    <h4>No Extracted Financial Metrics</h4>
                    <p>Click <strong>Extract Financials</strong> in the sidebar or Overview tab to extract verified metrics and compute YoY growth.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if not analysis.metrics:
                st.info("No structured financial metrics were detected in the analyzed pages.")
            else:
                # Tabular summary dataframe
                df_rows = []
                for m in analysis.metrics:
                    curr_str = f"{m.currency} " if m.currency else ""
                    val_str = f"{curr_str}{m.value:,}" if isinstance(m.value, (int, float)) else (f"{curr_str}{m.value}" if m.value else "N/A")
                    prior_str = f"{curr_str}{m.prior_period_value:,}" if isinstance(m.prior_period_value, (int, float)) else (f"{curr_str}{m.prior_period_value}" if m.prior_period_value else "—")
                    yoy_pct = f"{'+' if m.yoy_percentage_change and m.yoy_percentage_change > 0 else ''}{m.yoy_percentage_change:.2f}%" if m.yoy_percentage_change is not None else (m.yoy_calculation_note or "—")
                    yoy_abs = f"{'+' if m.yoy_absolute_change and m.yoy_absolute_change > 0 else ''}{m.yoy_absolute_change:,.2f}" if m.yoy_absolute_change is not None else "—"

                    df_rows.append(
                        {
                            "Metric": m.name,
                            "Reported Value": val_str,
                            "Scale": m.unit_scale or "units",
                            "Period": m.period_label or "Current",
                            "Prior Value": prior_str,
                            "YoY Change (%)": yoy_pct,
                            "YoY Change (Abs)": yoy_abs,
                            "Source Page": f"Page {m.source_page}" if m.source_page else "—",
                            "Status": "✅ Verified" if m.is_verified else "⚠️ Needs Review",
                            "Classification": "Calculated (Python)" if m.metric_type == "calculated" else "Reported",
                        }
                    )

                st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

                st.markdown("#### Itemized Metric Provenance & Quotations")
                for m in analysis.metrics:
                    v_badge = (
                        '<span class="verified-badge">✓ Verified in Source</span>'
                        if m.is_verified
                        else '<span style="color: #F87171; font-size: 0.75rem;">⚠️ Citation Unverified</span>'
                    )
                    type_badge = f'<span class="tag-pill">{m.metric_type.title()}</span>'
                    conflict_badge = (
                        '<span style="color: #F87171; font-weight: 600; margin-left: 0.5rem;">[CONFLICT DETECTED]</span>'
                        if m.has_conflict
                        else ""
                    )

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong>{m.name}</strong> {conflict_badge}
                                <div>
                                    {type_badge}
                                    <span class="citation-tag" style="margin-left: 0.4rem;">Page {m.source_page or 'N/A'}</span>
                                    {v_badge}
                                </div>
                            </div>
                            <p style="margin: 0.4rem 0 0.2rem 0; font-size: 0.95rem; color: #E2E8F0;">
                                Reported: <strong>{m.currency or ''} {m.value} {m.unit_scale or ''}</strong> ({m.period_label or 'Current'})
                                {f" | Prior: <strong>{m.prior_period_value}</strong> ({m.prior_period_label})" if m.prior_period_value else ""}
                                {f" | YoY: <strong>{m.yoy_calculation_note}</strong>" if m.yoy_calculation_note else ""}
                            </p>
                            {f'<p style="color: #FCA5A5; font-size: 0.85rem; margin-top: 0.25rem;"><strong>Conflict Note:</strong> {m.confidence_note}</p>' if m.has_conflict else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if m.source_quote:
                        with st.expander(f"Verbatim Source Excerpt for {m.name}", expanded=False):
                            st.caption(f'"{m.source_quote}"')

    # Tab 4: Risks & Outlook
    with tab_risks:
        st.subheader("Disclosed Risk Factors & Management Outlook")
        st.caption(
            "Itemized disclosure of operational, regulatory, and financial risks alongside forward-looking guidance and management expectations."
        )

        analysis = st.session_state.report_analysis
        if not analysis:
            st.markdown(
                """
                <div class="empty-card">
                    <h4>No Disclosed Risks or Outlook Extracted</h4>
                    <p>Click <strong>Extract Financials</strong> in the sidebar or Overview tab to extract risk disclosures and management forward guidance.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            col_risk, col_outlook = st.columns(2)

            with col_risk:
                st.markdown(f"#### ⚠️ Disclosed Risk Factors ({len(analysis.risks)})")
                if not analysis.risks:
                    st.info("No material risk factors identified in analyzed pages.")
                for r in analysis.risks:
                    cat_pill = f'<span class="tag-pill">{r.risk_category}</span>' if r.risk_category else ""
                    pages_label = f"Page {', '.join(str(p) for p in r.source_pages)}" if r.source_pages else "N/A"
                    v_badge = '<span class="verified-badge">✓ Verified</span>' if r.is_verified else '<span style="color: #F87171;">⚠️ Unverified</span>'

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                                <strong>{r.risk_title}</strong>
                                <div>
                                    {cat_pill}
                                    <span class="citation-tag">{pages_label}</span>
                                    {v_badge}
                                </div>
                            </div>
                            <p style="font-size: 0.92rem; color: #CBD5E1; margin-bottom: 0.4rem;">{r.description}</p>
                            {f'<p style="font-size: 0.85rem; color: #6EE7B7; margin-bottom: 0;"><strong>Company-Stated Mitigation:</strong> {r.company_stated_mitigation}</p>' if r.company_stated_mitigation else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if r.source_quotes:
                        with st.expander(f"Supporting Quotations ({len(r.source_quotes)})", expanded=False):
                            for q in r.source_quotes:
                                st.caption(f'"{q}"')

            with col_outlook:
                st.markdown(f"#### 🔮 Management Outlook & Guidance ({len(analysis.outlook)})")
                if not analysis.outlook:
                    st.info("No forward-looking guidance or outlook items identified.")
                for o in analysis.outlook:
                    type_pill = f'<span class="tag-pill">{o.statement_type.replace("_", " ").title()}</span>'
                    pages_label = f"Page {', '.join(str(p) for p in o.source_pages)}" if o.source_pages else "N/A"
                    v_badge = '<span class="verified-badge">✓ Verified</span>' if o.is_verified else '<span style="color: #F87171;">⚠️ Unverified</span>'

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                                <span style="font-weight: 600; color: #93C5FD;">Period: {o.period or 'Upcoming Period'}</span>
                                <div>
                                    {type_pill}
                                    <span class="citation-tag">{pages_label}</span>
                                    {v_badge}
                                </div>
                            </div>
                            <p style="font-size: 0.92rem; color: #CBD5E1; margin-bottom: 0.4rem;">{o.statement}</p>
                            {f'<p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0;"><strong>Target / Timeframe:</strong> {o.time_horizon or o.target_value}</p>' if (o.time_horizon or o.target_value) else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if o.source_quotes:
                        with st.expander(f"Supporting Excerpts ({len(o.source_quotes)})", expanded=False):
                            for q in o.source_quotes:
                                st.caption(f'"{q}"')

    # Tab 5: Sources & Processing Diagnostics
    with tab_sources:
        st.subheader("Processing Diagnostics & Retrieval Provenance")
        st.caption("Inspect parsing metrics, warnings, chunk counts, and retrieval context.")

        if st.session_state.report_analysis:
            analysis = st.session_state.report_analysis
            st.subheader("Financial Analysis Execution & Provenance Audit")
            v_rate_pct = analysis.verification_rate * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <h5>Analysis Pipeline Status: <strong>{analysis.analysis_status.upper()}</strong></h5>
                    <p>Pages Analyzed: <strong>{analysis.pages_analyzed}</strong> of <strong>{analysis.total_pages}</strong> ({analysis.batches_succeeded} batches succeeded, {analysis.batches_failed} failed)</p>
                    <p>KPI Metrics Extracted: <strong>{len(analysis.metrics)}</strong> | Disclosed Risks: <strong>{len(analysis.risks)}</strong> | Outlook Items: <strong>{len(analysis.outlook)}</strong></p>
                    <p>Citation Verification Rate: <strong>{v_rate_pct:.1f}%</strong> of claims confirmed against source page text</p>
                    <p>Generated At: <code>{analysis.generated_at}</code> | Model: <code>{config.gemini_generation_model}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if analysis.data_conflicts:
                st.warning(f"**Discrepancy Audit ({len(analysis.data_conflicts)} conflicts detected):**")
                for c in analysis.data_conflicts:
                    st.markdown(f"- {c}")

        if not st.session_state.report_metadata:
            st.markdown(
                """
                <div class="metric-card">
                    <h5>Document Ingestion Status</h5>
                    <p>Status: <strong>Awaiting Document</strong></p>
                    <p>Pages Extracted: 0 | Meaningful Text Pages: 0 | Total Characters: 0</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            meta = st.session_state.report_metadata
            ingestion = st.session_state.ingestion_result
            chunks = st.session_state.chunk_records or []
            indexing = st.session_state.indexing_result
            total_chars = ingestion.total_characters if ingestion else 0

            indexed_status = "Active Ephemeral ChromaDB Collection" if st.session_state.is_indexed else "Paused (No API Key)"
            collection_label = indexing.collection_name if indexing else "None"
            emb_model = indexing.embedding_model if indexing else config.gemini_embedding_model
            emb_dim = indexing.embedding_dimension if indexing else config.embedding_dimension

            st.markdown(
                f"""
                <div class="metric-card">
                    <h5>Document Provenance & Extraction Statistics</h5>
                    <p>Filename: <strong>{meta.filename}</strong> | Size: <strong>{meta.file_size_bytes / 1024:.1f} KB</strong></p>
                    <p>Total Pages: <strong>{meta.total_pages}</strong> | Meaningful Pages: <strong>{meta.meaningful_pages}</strong> | Total Extracted Characters: <strong>{total_chars:,}</strong></p>
                    <p>Total Chunks: <strong>{len(chunks)}</strong> | Indexed Vectors: <strong>{indexing.total_embeddings if indexing else 0}</strong></p>
                    <p>Embedding Model: <code>{emb_model}</code> ({emb_dim}D) | Vector Store: <strong>{indexed_status}</strong></p>
                    <p>Collection ID: <code>{collection_label}</code> | Digest ID: <code>{meta.document_id}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display Ingestion Warnings if any
            if meta.warnings:
                st.subheader(f"Extraction Notices & Warnings ({len(meta.warnings)})")
                for w in meta.warnings:
                    page_label = f"Page {w.page_number}: " if w.page_number is not None else "Document-Level: "
                    st.warning(f"**[{w.warning_type}]** {page_label}{w.message}")

            # Display Chunks Breakdown Table
            if chunks:
                st.subheader(f"Chunk Breakdown ({len(chunks)} chunks)")
                chunk_data = [
                    {
                        "Chunk ID": c.chunk_id,
                        "Page": c.page_numbers[0] if c.page_numbers else 1,
                        "Est. Tokens": len(c.text.split()),
                        "Char Count": len(c.text),
                        "Preview": (c.text[:90] + "...") if len(c.text) > 90 else c.text,
                    }
                    for c in chunks[:100]  # Display up to 100 for responsive UI
                ]
                st.dataframe(pd.DataFrame(chunk_data), use_container_width=True, hide_index=True)

            # Display Page Breakdown Table
            if st.session_state.page_records:
                st.subheader("Page-Level Extraction Breakdown")
                breakdown_data = [
                    {
                        "Page Number": p.page_number,
                        "Character Count": p.char_count,
                        "Has Meaningful Text": "✅ Yes" if p.has_meaningful_text else "⚠️ Low / Empty",
                        "Text Preview": (p.text[:90] + "...") if len(p.text) > 90 else (p.text or "(blank)"),
                    }
                    for p in st.session_state.page_records
                ]
                st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

    # Tab 6: About & Methodology
    with tab_about:
        st.subheader("Architecture & Methodology")
        st.markdown(
            """
            ### System Design
            The **Financial Report Analyzer** transforms unstructured corporate PDFs into verifiable financial intelligence:
            - **PDF Extraction**: Page-aware in-memory parsing via `PyMuPDF` preserving 1-based page indices.
            - **Chunking**: Sentence-boundary-aware chunking (~850 target tokens with 120 token overlap).
            - **Vector Index**: Isolated in-memory ephemeral ChromaDB collection per active user session.
            - **Embeddings**: Google GenAI `gemini-embedding-2` embedding vectors (768 dimensions).
            - **RAG & Citations**: Strict prompt grounding with exact page and quotation verification.
            - **Data Integrity**: Financial metrics retain period, currency, scale, and calculation flags; missing metrics are never represented as zero.

            ### Privacy & Security
            - Uploaded files are processed in-memory and not stored persistently in the MVP.
            - Active vector indices are ephemeral and isolated to the browser session.
            - Uploaded text is transmitted to the configured Google Gemini API for embedding and generation.

            ### Disclaimer
            This tool is designed for research and study of public company filings. It does not provide investment advice, stock recommendations, or audit opinions. Always consult original filings submitted to regulatory authorities.
            """
        )


if __name__ == "__main__":
    try:
        main()
    except FinancialAnalyzerError as e:
        logger.error("Financial analyzer domain error: %s", str(e), exc_info=True)
        st.error(f"Application Error: {e.user_friendly_message}")
    except Exception as e:
        logger.critical("Unexpected application crash: %s", str(e), exc_info=True)
        st.error("An unexpected error occurred. Please refresh the page or clear the active session.")
