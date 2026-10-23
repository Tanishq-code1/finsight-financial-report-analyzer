"""Smoke test verifying imports, modules, and basic entry points."""

import importlib
from pathlib import Path


def test_core_module_imports():
    """Verify that all core project modules can be cleanly imported."""
    modules = [
        "financial_report_analyzer",
        "financial_report_analyzer.config",
        "financial_report_analyzer.schemas",
        "financial_report_analyzer.errors",
        "financial_report_analyzer.pdf_loader",
        "financial_report_analyzer.chunker",
        "financial_report_analyzer.embedding_service",
        "financial_report_analyzer.vector_store",
        "financial_report_analyzer.indexing_pipeline",
        "financial_report_analyzer.citation_validator",
        "financial_report_analyzer.rag",
        "financial_report_analyzer.financial_calculations",
        "financial_report_analyzer.financial_extractor",
        "financial_report_analyzer.financial_analysis_pipeline",
        "financial_report_analyzer.export_service",
    ]
    for mod in modules:
        loaded = importlib.import_module(mod)
        assert loaded is not None


def test_streamlit_app_loads():
    """Verify app.py syntax and import compatibility."""
    import app

    assert hasattr(app, "main")
    assert hasattr(app, "init_session_state")
    assert hasattr(app, "clear_report_session")


def test_pdf_loader_integration_smoke():
    """Verify that pdf_loader output integrates seamlessly into app data contracts."""
    import pymupdf
    from financial_report_analyzer.pdf_loader import load_pdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Sample Financial Smoke Test Report")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = load_pdf(pdf_bytes, "smoke_test.pdf")
    assert result.metadata.total_pages == 1
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Sample Financial" in result.pages[0].text
    assert result.metadata.document_id.startswith("doc_")


def test_render_kpi_card_with_none_and_empty():
    """Verify render_kpi_card handles None metric, None value, and missing fields without exceptions."""
    import app
    from financial_report_analyzer.schemas import FinancialMetric

    html_none = app.render_kpi_card("Total Revenue", None)
    assert "Total Revenue" in html_none
    assert "N/A" in html_none
    assert "Not reported in filing" in html_none

    metric_val_none = FinancialMetric(name="Total Revenue", value=None)
    html_val_none = app.render_kpi_card("Total Revenue", metric_val_none)
    assert "N/A" in html_val_none
    assert "Not reported in filing" in html_val_none


def test_render_kpi_card_positive_and_negative_yoy():
    """Verify YoY badges render correctly for positive, negative, zero, and calculated notes."""
    import app
    from financial_report_analyzer.schemas import FinancialMetric

    m_pos = FinancialMetric(name="Revenue", value=1500.0, yoy_percentage_change=12.34, source_page=5)
    html_pos = app.render_kpi_card("Revenue", m_pos)
    assert "+12.34% YoY" in html_pos
    assert "kpi-badge-pos" in html_pos
    assert "p. 5" in html_pos

    m_neg = FinancialMetric(name="Net Income", value=500.0, yoy_percentage_change=-5.67, source_page=6)
    html_neg = app.render_kpi_card("Net Income", m_neg)
    assert "-5.67% YoY" in html_neg
    assert "kpi-badge-neg" in html_neg

    m_zero = FinancialMetric(name="Operating Income", value=200.0, yoy_percentage_change=0.0)
    html_zero = app.render_kpi_card("Operating Income", m_zero)
    assert "0.00% YoY" in html_zero
    assert "kpi-badge-neu" in html_zero

    m_note = FinancialMetric(name="Free Cash Flow", value=300.0, yoy_calculation_note="+10% YoY (Python)")
    html_note = app.render_kpi_card("Free Cash Flow", m_note)
    assert "+10% YoY (Python)" in html_note


def test_render_kpi_card_with_currency_and_scale():
    """Verify proper formatting of currency and unit scale."""
    import app
    from financial_report_analyzer.schemas import FinancialMetric

    m = FinancialMetric(name="Total Revenue", value=1250000, currency="EUR", unit_scale="thousands", source_page=2)
    html = app.render_kpi_card("Total Revenue", m)
    assert "EUR" in html
    assert "1,250,000" in html
    assert "thousands" in html


def test_session_state_initialization_and_clearing():
    """Verify init_session_state and clear_report_session handle state variables properly."""
    import streamlit as st
    import app

    app.init_session_state()
    assert st.session_state.pipeline_stage == "EMPTY"
    assert st.session_state.document_id is None
    assert st.session_state.financial_error is None

    # Simulate loaded state
    st.session_state.document_id = "doc_123"
    st.session_state.pipeline_stage = "INDEXED"
    st.session_state.is_indexed = True

    app.clear_report_session()
    assert st.session_state.document_id is None
    assert st.session_state.pipeline_stage == "EMPTY"
    assert st.session_state.is_indexed is False


_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_apptest_runs_cleanly_initial_state():
    """Verify Streamlit AppTest runs cleanly with no errors on fresh initialization."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0


def test_apptest_runs_cleanly_with_populated_analysis():
    """Regression test: verify no NameError: name 'Optional' is not defined when report_analysis is populated."""
    from streamlit.testing.v1 import AppTest
    from financial_report_analyzer.schemas import (
        DocumentMetadata,
        ReportAnalysis,
        FinancialMetric,
        RiskItem,
        OutlookItem,
    )

    at = AppTest.from_file(_APP_PATH, default_timeout=30)
    at.session_state.document_id = "doc_test_10k"
    at.session_state.report_metadata = DocumentMetadata(
        filename="acme_10k.pdf",
        file_size_bytes=2048,
        total_pages=4,
        meaningful_pages=4,
        document_id="doc_test_10k",
    )
    at.session_state.report_analysis = ReportAnalysis(
        document_id="doc_test_10k",
        company_name="Acme Corp",
        fiscal_period="FY2024",
        total_pages=4,
        pages_analyzed=4,
        batches_succeeded=1,
        batches_failed=0,
        analysis_status="complete",
        executive_summary="Acme Corp reported record sales in FY2024.",
        key_takeaways=["Revenue grew 15%", "Operating margin expanded"],
        metrics=[
            FinancialMetric(name="Total Revenue", normalized_name="revenue", value=96770.0, yoy_percentage_change=15.0, source_page=1),
            FinancialMetric(name="Net Income", normalized_name="net_income", value=22100.0, yoy_percentage_change=8.5, source_page=2),
            FinancialMetric(name="Operating Cash Flow", normalized_name="operating_cash_flow", value=28500.0, source_page=3),
            FinancialMetric(name="Free Cash Flow", value=19500.0, yoy_calculation_note="+10% YoY (Calculated)", source_page=3),
        ],
        risks=[
            RiskItem(
                risk_title="Supply Chain",
                description="Supply chain disruptions",
                risk_category="Operational",
                company_stated_mitigation="Dual sourcing",
                source_pages=[4],
                source_quotes=["Dual sourcing active"],
                is_verified=True,
            )
        ],
        outlook=[
            OutlookItem(
                statement="Expecting strong growth in FY2025",
                period="FY2025",
                statement_type="guidance",
                source_pages=[4],
                source_quotes=["Expecting strong growth"],
                is_verified=True,
            )
        ],
        verification_rate=1.0,
        generated_at="2026-10-24T00:00:00Z",
    )
    at.run()
    assert len(at.exception) == 0


def test_apptest_runs_cleanly_with_missing_company_or_period():
    """Verify dashboard renders gracefully even if company name or period are not resolved."""
    from streamlit.testing.v1 import AppTest
    from financial_report_analyzer.schemas import DocumentMetadata, ReportAnalysis

    at = AppTest.from_file(_APP_PATH, default_timeout=30)
    at.session_state.document_id = "doc_test_unknown"
    at.session_state.report_metadata = DocumentMetadata(
        filename="unknown_report.pdf",
        file_size_bytes=1024,
        total_pages=2,
        meaningful_pages=2,
        document_id="doc_test_unknown",
    )
    at.session_state.report_analysis = ReportAnalysis(
        document_id="doc_test_unknown",
        company_name=None,
        fiscal_period=None,
        total_pages=2,
        pages_analyzed=2,
        batches_succeeded=1,
        batches_failed=0,
        analysis_status="complete",
        executive_summary="Summary without company header.",
        key_takeaways=[],
        metrics=[],
        risks=[],
        outlook=[],
        verification_rate=1.0,
        generated_at="2026-10-24T00:00:00Z",
    )
    at.run()
    assert len(at.exception) == 0


def test_apptest_runs_cleanly_with_partial_and_error_states():
    """Verify dashboard displays notices and does not crash when indexing or extraction errors occur."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=30)
    at.session_state.indexing_error = "Gemini embedding API quota exceeded. Delay 25s."
    at.session_state.financial_error = "Model request timed out after 120s."
    at.session_state.pipeline_stage = "INDEXING_FAILED"
    at.run()
    assert len(at.exception) == 0
