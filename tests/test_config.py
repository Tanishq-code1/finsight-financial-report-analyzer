"""Unit tests for application configuration management."""

import os
import pytest
from financial_report_analyzer.config import load_config, setup_logging


def test_default_config(monkeypatch: pytest.MonkeyPatch):
    """Verify safe fallback defaults when environment variables are omitted."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    monkeypatch.delenv("MAX_PDF_PAGES", raising=False)
    monkeypatch.delenv("CHUNK_SIZE_TOKENS", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP_TOKENS", raising=False)
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
    monkeypatch.delenv("MIN_EVIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    cfg = load_config()
    assert cfg.gemini_api_key is None
    assert cfg.has_api_key is False
    assert cfg.gemini_generation_model == "gemini-3.8-flash"
    assert cfg.gemini_embedding_model == "gemini-embedding-2"
    assert cfg.embedding_dimension == 768
    assert cfg.max_upload_mb == 25
    assert cfg.max_upload_bytes == 25 * 1024 * 1024
    assert cfg.max_pdf_pages == 250
    assert cfg.chunk_size_tokens == 850
    assert cfg.chunk_overlap_tokens == 120
    assert cfg.retrieval_top_k == 5
    assert cfg.min_evidence_threshold == 0.35
    assert cfg.log_level == "INFO"
    assert cfg.embedding_tpm_limit == 30000
    assert cfg.embedding_rpm_limit == 60
    assert cfg.embedding_safety_margin == 0.80
    assert cfg.embedding_max_retries == 5
    assert cfg.embedding_initial_backoff_sec == 2.0
    assert cfg.embedding_max_backoff_sec == 60.0
    assert cfg.embedding_batch_size == 1


def test_custom_config(sample_config_env):
    """Verify custom environment values are loaded accurately."""
    cfg = load_config()
    assert cfg.has_api_key is True
    assert cfg.gemini_api_key == "test-api-key-12345"
    assert cfg.max_upload_mb == 15
    assert cfg.max_pdf_pages == 100
    assert cfg.chunk_size_tokens == 500
    assert cfg.chunk_overlap_tokens == 100
    assert cfg.retrieval_top_k == 4
    assert cfg.min_evidence_threshold == 0.40
    assert cfg.log_level == "DEBUG"


def test_embedding_rate_limit_env_config(monkeypatch: pytest.MonkeyPatch):
    """Verify custom embedding rate limits loaded correctly from environment variables."""
    monkeypatch.setenv("EMBEDDING_TPM_LIMIT", "45000")
    monkeypatch.setenv("EMBEDDING_RPM_LIMIT", "100")
    monkeypatch.setenv("EMBEDDING_SAFETY_MARGIN", "0.75")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "5")

    cfg = load_config()
    assert cfg.embedding_tpm_limit == 45000
    assert cfg.embedding_rpm_limit == 100
    assert cfg.embedding_safety_margin == 0.75
    assert cfg.embedding_batch_size == 5


def test_invalid_numeric_env_falls_back(monkeypatch: pytest.MonkeyPatch):
    """Ensure non-numeric environment values fall back gracefully without crashing."""
    monkeypatch.setenv("MAX_UPLOAD_MB", "invalid_number")
    monkeypatch.setenv("MAX_PDF_PAGES", "bad_pages")
    monkeypatch.setenv("LOG_LEVEL", "NONEXISTENT")

    cfg = load_config()
    assert cfg.max_upload_mb == 25
    assert cfg.max_pdf_pages == 250
    assert cfg.log_level == "INFO"


def test_setup_logging():
    """Verify logger setup does not raise exceptions."""
    setup_logging("DEBUG")
    setup_logging("INFO")
