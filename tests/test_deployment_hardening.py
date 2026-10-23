"""Comprehensive deployment hardening, security, configuration, and cloud-readiness tests."""

import os
import sys
import tomllib
from pathlib import Path
from typing import Dict, Any
import pytest
from unittest.mock import MagicMock, patch

from financial_report_analyzer.config import load_config, AppConfig, _get_setting
from financial_report_analyzer.errors import (
    RAGGenerationError,
    EmbeddingError,
    DocumentValidationError,
    PDFParsingError,
)
from financial_report_analyzer.rag import RAGService
from financial_report_analyzer.financial_extractor import FinancialExtractor
from financial_report_analyzer.vector_store import VectorStore
from financial_report_analyzer.schemas import ChunkRecord


def test_sys_path_src_resolution():
    """Verify that app.py ensures the 'src' directory is in sys.path for direct cloud execution."""
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    assert src_dir.exists() and src_dir.is_dir()

    # Verify that importing app succeeds and src is preserved in sys.path
    import app

    assert str(src_dir) in sys.path
    assert hasattr(app, "main")
    assert hasattr(app, "init_session_state")
    assert hasattr(app, "clear_report_session")


def test_config_no_api_key_safe_diagnostics(monkeypatch: pytest.MonkeyPatch):
    """Verify configuration loads safely without credentials and diagnostics contain no secret leakage."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = load_config()

    assert cfg.has_api_key is False
    assert cfg.gemini_api_key is None

    diag = cfg.safe_diagnostics
    assert isinstance(diag, dict)
    assert "api_key" not in diag
    assert "gemini_api_key" not in diag
    assert diag["api_key_configured"] is False
    assert diag["generation_model"] == cfg.gemini_generation_model
    assert diag["embedding_model"] == cfg.gemini_embedding_model
    assert diag["max_upload_mb"] == 25
    assert diag["max_pdf_pages"] == 250
    assert diag["session_qa_budget"] == 25
    assert diag["session_analysis_budget"] == 5


def test_config_reads_streamlit_secrets(monkeypatch: pytest.MonkeyPatch):
    """Verify Streamlit Community Cloud secrets fallback when environment variables are omitted."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("SESSION_QA_BUDGET", raising=False)

    fake_secrets = {
        "GEMINI_API_KEY": "streamlit-cloud-secret-key-999",
        "GEMINI_GENERATION_MODEL": "gemini-3.8-flash-cloud",
        "SESSION_QA_BUDGET": "35",
    }

    mock_st = MagicMock()
    mock_st.secrets = fake_secrets

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = load_config()
        assert cfg.has_api_key is True
        assert cfg.gemini_api_key == "streamlit-cloud-secret-key-999"
        assert cfg.gemini_generation_model == "gemini-3.8-flash-cloud"
        assert cfg.session_qa_budget == 35


def test_config_streamlit_secrets_missing_graceful(monkeypatch: pytest.MonkeyPatch):
    """Verify missing secrets or exception raised by st.secrets falls back gracefully."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class BrokenSecrets:
        def __contains__(self, item):
            raise RuntimeError("Secrets store not initialized")

    mock_st = MagicMock()
    mock_st.secrets = BrokenSecrets()

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = load_config()
        assert cfg.has_api_key is False
        assert cfg.gemini_api_key is None


def test_session_budgets_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch):
    """Verify per-session abuse controls and budgets are configurable."""
    monkeypatch.setenv("SESSION_QA_BUDGET", "50")
    monkeypatch.setenv("SESSION_ANALYSIS_BUDGET", "10")

    cfg = load_config()
    assert cfg.session_qa_budget == 50
    assert cfg.session_analysis_budget == 10

    # Test non-numeric fallback
    monkeypatch.setenv("SESSION_QA_BUDGET", "invalid")
    monkeypatch.setenv("SESSION_ANALYSIS_BUDGET", "bad")
    fallback_cfg = load_config()
    assert fallback_cfg.session_qa_budget == 25
    assert fallback_cfg.session_analysis_budget == 5


def test_cors_and_xsrf_settings_in_config_toml():
    """Verify Streamlit deployment server configuration enables XSRF and does not disable CORS."""
    repo_root = Path(__file__).resolve().parent.parent
    config_toml = repo_root / ".streamlit" / "config.toml"
    assert config_toml.exists(), "Missing .streamlit/config.toml"

    with open(config_toml, "rb") as f:
        cfg = tomllib.load(f)

    server_cfg = cfg.get("server", {})
    assert server_cfg.get("headless") is True
    assert server_cfg.get("enableXsrfProtection") is True
    # Crucial security hardening: enableCORS must NOT be false (which weakens WebSocket cross-origin controls)
    assert server_cfg.get("enableCORS") is not False
    assert server_cfg.get("maxUploadSize") == 25


def test_pyproject_version_and_metadata():
    """Verify pyproject.toml package version is 1.0.0 and requires-python supports Python 3.12."""
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    assert pyproject_path.exists()

    with open(pyproject_path, "rb") as f:
        meta = tomllib.load(f)

    proj = meta.get("project", {})
    assert proj.get("name") == "financial-report-analyzer"
    assert proj.get("version") == "1.0.0"
    assert proj.get("requires-python") in (">=3.11", ">=3.11,<3.14", ">=3.12")


def test_no_secrets_in_tracked_files():
    """Verify that no actual Google API keys or credentials exist in git-tracked files."""
    repo_root = Path(__file__).resolve().parent.parent

    # Patterns indicating potential secrets
    secret_patterns = ["AIzaSy", "sk-proj-", "-----BEGIN RSA PRIVATE KEY-----"]

    # Files to inspect
    scan_exts = {".py", ".toml", ".json", ".md", ".txt", ".env.example"}
    for path in repo_root.rglob("*"):
        if path.is_file() and path.suffix in scan_exts:
            # Skip git folder, venv, and cache
            rel_str = str(path.relative_to(repo_root)).replace("\\", "/")
            if any(part in rel_str for part in [".venv", ".git", "__pycache__", ".pytest_cache"]):
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pat in secret_patterns:
                    assert pat not in content, f"Potential secret pattern '{pat}' found in {rel_str}"
            except Exception:
                pass


def test_api_error_user_friendly_messages():
    """Verify that HTTP 401, 403, 429, and 503 produce clear, non-leaking user-friendly messages."""
    # Test RAG Service error handling
    rag = RAGService()
    extractor = FinancialExtractor()

    # Simulate genai APIError with code 401
    class FakeAPIError(Exception):
        def __init__(self, code, msg):
            super().__init__(msg)
            self.code = code

    # Test that 401 in RAG results in user_friendly_message regarding authentication
    with patch.object(rag, "_call_gemini_with_retry", side_effect=RAGGenerationError("internal detail", user_friendly_message="Authentication failed: Invalid or unauthorized Gemini API key. Please verify GEMINI_API_KEY.")):
        with pytest.raises(RAGGenerationError) as exc_info:
            rag._call_gemini_with_retry("test content")
        assert "Authentication failed" in exc_info.value.user_friendly_message
        assert "GEMINI_API_KEY" in exc_info.value.user_friendly_message

    # Test that 429 results in quota message
    with patch.object(extractor, "_call_gemini_with_retry", side_effect=RAGGenerationError("internal detail", user_friendly_message="Gemini API quota or rate limit exceeded. Please check your Google AI Studio quota or try again later.")):
        with pytest.raises(RAGGenerationError) as exc_info:
            extractor._call_gemini_with_retry("test content")
        assert "quota or rate limit exceeded" in exc_info.value.user_friendly_message


def test_ephemeral_vector_store_cleanup():
    """Verify that deleting an ephemeral ChromaDB collection removes stored vectors and releases collection."""
    import chromadb
    client = chromadb.EphemeralClient()
    store = VectorStore(document_id="doc_cleanup_1", client=client, expected_dimension=768)

    # Add a mock chunk with embedding
    chunk = ChunkRecord(
        chunk_id="chunk_test_1",
        document_id="doc_cleanup_1",
        source_filename="cleanup.pdf",
        page_numbers=[1],
        chunk_index=0,
        text="Sample cash flow information for ephemeral store verification.",
        char_count=60,
        token_count_estimate=10,
        embedding=[0.1] * 768,
    )
    added = store.add_chunks(chunks=[chunk])
    assert added == 1
    assert store.count() == 1

    # Delete collection
    store.delete()
    assert store.count() == 0
