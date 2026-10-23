"""Smoke test verifying imports, modules, and basic entry points."""

import importlib


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
