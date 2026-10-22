"""Financial Report Analyzer — Main Streamlit Application."""

import logging
import streamlit as st
import pandas as pd

from financial_report_analyzer.config import load_config, setup_logging
from financial_report_analyzer.errors import FinancialAnalyzerError, DocumentValidationError, PDFParsingError
from financial_report_analyzer.pdf_loader import load_pdf

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
    .warning-card {
        background-color: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        color: #FDE047;
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
    if "report_analysis" not in st.session_state:
        st.session_state.report_analysis = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "status_message" not in st.session_state:
        st.session_state.status_message = None


def clear_report_session() -> None:
    """Clear active report data and reset session state cleanly."""
    logger.info("Resetting session state for active user.")
    st.session_state.document_id = None
    st.session_state.report_metadata = None
    st.session_state.page_records = None
    st.session_state.ingestion_result = None
    st.session_state.report_analysis = None
    st.session_state.chat_history = []
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
            )
        with col_clear:
            clear_clicked = st.button("Clear Report", on_click=clear_report_session, use_container_width=True)

        if uploaded_file and analyze_clicked and has_valid_upload:
            st.session_state.is_processing = True
            with st.spinner("Parsing document and extracting page provenance..."):
                try:
                    result = load_pdf(
                        pdf_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        config=config,
                    )
                    st.session_state.document_id = result.metadata.document_id
                    st.session_state.report_metadata = result.metadata
                    st.session_state.page_records = result.pages
                    st.session_state.ingestion_result = result
                    st.session_state.is_processing = False

                    st.success(
                        f"Parsed {result.metadata.total_pages} pages ({result.total_characters:,} characters)!"
                    )
                    if result.has_warnings:
                        st.warning(
                            f"{len(result.metadata.warnings)} parsing notice(s) detected. Check 'Sources & Processing' tab."
                        )
                except (DocumentValidationError, PDFParsingError) as err:
                    st.session_state.is_processing = False
                    logger.error("PDF Ingestion error: %s", str(err))
                    st.error(f"Validation Error: {err.user_friendly_message}")
                except Exception as err:
                    st.session_state.is_processing = False
                    logger.critical("Unexpected ingestion error: %s", str(err), exc_info=True)
                    st.error("An unexpected error occurred while processing the PDF.")

        st.divider()
        st.subheader("Runtime Configuration")
        st.caption(f"**Generation Model:** `{config.gemini_generation_model}`")
        st.caption(f"**Embedding Model:** `{config.gemini_embedding_model}`")
        st.caption(f"**Max Size:** `{config.max_upload_mb} MB` | **Max Pages:** `{config.max_pdf_pages}`")
        st.caption(f"**Chunk Size:** `{config.chunk_size_tokens} tokens`")
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
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Document Ingested: {meta.filename}</h4>
                    <p>Total Pages: <strong>{meta.total_pages}</strong> | Meaningful Text Pages: <strong>{meta.meaningful_pages}</strong></p>
                    <p>Document ID: <code>{meta.document_id}</code></p>
                    <p>Status: <strong>Text Extracted with 1-based page provenance (Ready for chunking & embeddings in Phase 3)</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not st.session_state.report_analysis:
                st.info("Executive summary and key takeaways generation will be connected in subsequent phases.")

    # Tab 2: Ask the Report (RAG Q&A)
    with tab_qa:
        st.subheader("Ask the Report (Retrieval-Augmented Q&A)")
        st.caption("Ask natural-language questions grounded strictly in the report's text with verifiable citations.")

        # Suggested Questions
        st.markdown("**Suggested Questions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("What were the main revenue drivers?", disabled=st.session_state.report_metadata is None)
        with col2:
            st.button("How did operating cash flow compare to net income?", disabled=st.session_state.report_metadata is None)
        with col3:
            st.button("What are the primary operational risks?", disabled=st.session_state.report_metadata is None)

        if not st.session_state.report_metadata:
            st.markdown(
                """
                <div class="empty-card">
                    <h4>Document Not Yet Loaded</h4>
                    <p>Upload and process a filing to query revenue, margins, capital expenditures, or disclosed risks.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            user_query = st.chat_input("Ask a question about this financial filing...")
            if user_query:
                st.write(f"Query: {user_query}")
                st.info("Retrieval-augmented Q&A pipeline will be connected in Phase 3 & 4.")

    # Tab 3: Financial Metrics
    with tab_metrics:
        st.subheader("Key Performance Indicators (KPIs)")
        st.caption("Structured extraction of revenue, operating income, net income, cash flow, capex, cash, debt, and EPS.")

        if not st.session_state.report_analysis:
            st.markdown(
                """
                <div class="empty-card">
                    <h4>Financial Metrics Table</h4>
                    <p>Extracted metrics will be presented with period labels, currency, reported vs. calculated classification, and source page citations.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Metrics table will be rendered here.")

    # Tab 4: Risks & Outlook
    with tab_risks:
        st.subheader("Disclosed Risks & Management Outlook")
        st.caption("Separate company-stated disclosures from forward-looking commentary.")

        col_risk, col_outlook = st.columns(2)
        with col_risk:
            st.markdown("#### Disclosed Risk Factors")
            st.markdown(
                """
                <div class="empty-card">
                    <p>Itemized risk disclosures with cited excerpts and stated mitigations.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_outlook:
            st.markdown("#### Management Outlook & Guidance")
            st.markdown(
                """
                <div class="empty-card">
                    <p>Forward-looking statements, target periods, and qualitative commentary.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tab 5: Sources & Processing Diagnostics
    with tab_sources:
        st.subheader("Processing Diagnostics & Retrieval Provenance")
        st.caption("Inspect parsing metrics, warnings, chunk counts, and retrieval context.")

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
            total_chars = ingestion.total_characters if ingestion else 0

            st.markdown(
                f"""
                <div class="metric-card">
                    <h5>Document Provenance & Extraction Statistics</h5>
                    <p>Filename: <strong>{meta.filename}</strong> | Size: <strong>{meta.file_size_bytes / 1024:.1f} KB</strong></p>
                    <p>Total Pages: <strong>{meta.total_pages}</strong> | Meaningful Pages: <strong>{meta.meaningful_pages}</strong> | Total Extracted Characters: <strong>{total_chars:,}</strong></p>
                    <p>Document Digest ID: <code>{meta.document_id}</code></p>
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
            - **Chunking**: Sentence-boundary-aware chunking (~700–1,000 tokens with 100–150 token overlap).
            - **Vector Index**: Isolated in-memory ephemeral ChromaDB collection per active user session.
            - **Embeddings**: Google GenAI `gemini-embedding-2` embedding vectors.
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
