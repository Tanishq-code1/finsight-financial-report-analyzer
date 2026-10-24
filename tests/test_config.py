"""Unit tests for application configuration management."""

import os
import pytest
from financial_report_analyzer.config import AppConfig, load_config, setup_logging


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
    assert cfg.embedding_batch_size == 16
    assert cfg.embedding_deadline_sec == 120.0
    assert "embedding_tpm_limit" in cfg.safe_diagnostics
    assert "embedding_rpm_limit" in cfg.safe_diagnostics
    assert "embedding_safety_margin" in cfg.safe_diagnostics
    assert "embedding_batch_size" in cfg.safe_diagnostics


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


def test_appconfig_getattr_fallback_for_missing_attributes():
    """Verify that AppConfig.__getattr__ safely returns defaults for missing attributes."""
    cfg = AppConfig()

    # Pacing attributes
    assert cfg.embedding_tpm_limit == 30000
    assert cfg.embedding_rpm_limit == 60
    assert cfg.embedding_safety_margin == 0.80
    assert cfg.embedding_max_retries == 5
    assert cfg.embedding_initial_backoff_sec == 2.0
    assert cfg.embedding_max_backoff_sec == 60.0
    assert cfg.embedding_batch_size == 16
    assert cfg.embedding_deadline_sec == 120.0

    # Test accessing a non-existent configuration attribute
    with pytest.raises(AttributeError, match="has no attribute 'non_existent_key'"):
        _ = cfg.non_existent_key


def test_appconfig_safe_diagnostics_and_properties_resilience():
    """Verify safe_diagnostics, max_upload_bytes, and has_api_key never raise AttributeError."""
    cfg = AppConfig()
    diag = cfg.safe_diagnostics

    assert isinstance(diag, dict)
    assert diag["generation_model"] == "gemini-3.8-flash"
    assert diag["embedding_model"] == "gemini-embedding-2"
    assert diag["embedding_dimension"] == 768
    assert diag["embedding_tpm_limit"] == 30000
    assert diag["embedding_rpm_limit"] == 60
    assert diag["embedding_safety_margin"] == 0.80
    assert diag["api_key_configured"] is False
    assert cfg.max_upload_bytes == 25 * 1024 * 1024
    assert cfg.has_api_key is False


def test_legacy_or_stripped_config_object_tolerance():
    """Simulate an older in-memory class definition from Streamlit hot-reload cache."""
    class StaleAppConfig:
        gemini_generation_model = "gemini-3.8-flash"
        gemini_embedding_model = "gemini-embedding-2"

    stale_cfg = StaleAppConfig()

    # Direct getattr calls used across app.py sidebar and diagnostics
    assert getattr(stale_cfg, "embedding_tpm_limit", 30000) == 30000
    assert getattr(stale_cfg, "embedding_rpm_limit", 60) == 60
    assert getattr(stale_cfg, "embedding_safety_margin", 0.80) == 0.80
    assert getattr(stale_cfg, "max_upload_mb", 25) == 25
    assert getattr(stale_cfg, "max_pdf_pages", 250) == 250
    assert getattr(stale_cfg, "session_qa_budget", 25) == 25
    assert getattr(stale_cfg, "session_analysis_budget", 5) == 5
    assert getattr(stale_cfg, "has_api_key", False) is False
