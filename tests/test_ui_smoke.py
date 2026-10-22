"""Smoke test verifying imports, modules, and basic entry points."""

import importlib


def test_core_module_imports():
    """Verify that all core project modules can be cleanly imported."""
    modules = [
        "financial_report_analyzer",
        "financial_report_analyzer.config",
        "financial_report_analyzer.schemas",
        "financial_report_analyzer.errors",
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
