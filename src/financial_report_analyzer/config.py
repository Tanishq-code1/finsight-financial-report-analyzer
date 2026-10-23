"""Application configuration management with environment variable and Streamlit secrets support."""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()


def _get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve setting from environment variable or Streamlit secrets if running within Streamlit."""
    val = os.getenv(key)
    if val is not None and val.strip():
        return val.strip()

    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            secret_val = st.secrets[key]
            if secret_val is not None and str(secret_val).strip():
                return str(secret_val).strip()
    except Exception:
        # Ignore StreamlitSecretNotFoundError or missing secrets.toml
        pass

    return default


_DEFAULT_CONFIG_VALUES: Dict[str, Any] = {
    "gemini_api_key": None,
    "gemini_generation_model": "gemini-3.8-flash",
    "gemini_generation_fallback_model": "gemini-3.6-flash",
    "gemini_embedding_model": "gemini-embedding-2",
    "embedding_dimension": 768,
    "max_upload_mb": 25,
    "max_pdf_pages": 250,
    "chunk_size_tokens": 850,
    "chunk_overlap_tokens": 120,
    "retrieval_top_k": 5,
    "min_evidence_threshold": 0.35,
    "log_level": "INFO",
    "session_qa_budget": 25,
    "session_analysis_budget": 5,
    "embedding_tpm_limit": 30000,
    "embedding_rpm_limit": 60,
    "embedding_safety_margin": 0.80,
    "embedding_max_retries": 5,
    "embedding_initial_backoff_sec": 2.0,
    "embedding_max_backoff_sec": 60.0,
    "embedding_batch_size": 1,
    "embedding_deadline_sec": 120.0,
    "generation_timeout_sec": 120.0,
}


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration container."""

    gemini_api_key: Optional[str] = None
    gemini_generation_model: str = "gemini-3.8-flash"
    gemini_generation_fallback_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768
    max_upload_mb: int = 25
    max_pdf_pages: int = 250
    chunk_size_tokens: int = 850
    chunk_overlap_tokens: int = 120
    retrieval_top_k: int = 5
    min_evidence_threshold: float = 0.35
    log_level: str = "INFO"
    session_qa_budget: int = 25
    session_analysis_budget: int = 5
    embedding_tpm_limit: int = 30000
    embedding_rpm_limit: int = 60
    embedding_safety_margin: float = 0.80
    embedding_max_retries: int = 5
    embedding_initial_backoff_sec: float = 2.0
    embedding_max_backoff_sec: float = 60.0
    embedding_batch_size: int = 1
    embedding_deadline_sec: float = 120.0
    generation_timeout_sec: float = 120.0

    def __getattr__(self, name: str) -> Any:
        """Provide safe fallback default values if an attribute is missing.

        Guards against Streamlit hot-reload module skew, legacy cached instances,
        or partially initialized AppConfig objects.
        """
        if name in _DEFAULT_CONFIG_VALUES:
            return _DEFAULT_CONFIG_VALUES[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def max_upload_bytes(self) -> int:
        """Maximum allowed upload file size in bytes."""
        mb = getattr(self, "max_upload_mb", 25)
        return mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        """Check if a non-empty Gemini API key is configured."""
        val = getattr(self, "gemini_api_key", None)
        return bool(val and str(val).strip())

    @property
    def safe_diagnostics(self) -> Dict[str, Any]:
        """Safe runtime diagnostic summary containing no secrets or local filesystem paths."""
        return {
            "generation_model": getattr(self, "gemini_generation_model", "gemini-3.8-flash"),
            "generation_fallback_model": getattr(self, "gemini_generation_fallback_model", "gemini-3.6-flash"),
            "embedding_model": getattr(self, "gemini_embedding_model", "gemini-embedding-2"),
            "embedding_dimension": getattr(self, "embedding_dimension", 768),
            "max_upload_mb": getattr(self, "max_upload_mb", 25),
            "max_pdf_pages": getattr(self, "max_pdf_pages", 250),
            "chunk_size_tokens": getattr(self, "chunk_size_tokens", 850),
            "chunk_overlap_tokens": getattr(self, "chunk_overlap_tokens", 120),
            "retrieval_top_k": getattr(self, "retrieval_top_k", 5),
            "min_evidence_threshold": getattr(self, "min_evidence_threshold", 0.35),
            "session_qa_budget": getattr(self, "session_qa_budget", 25),
            "session_analysis_budget": getattr(self, "session_analysis_budget", 5),
            "embedding_tpm_limit": getattr(self, "embedding_tpm_limit", 30000),
            "embedding_rpm_limit": getattr(self, "embedding_rpm_limit", 60),
            "embedding_safety_margin": getattr(self, "embedding_safety_margin", 0.80),
            "embedding_max_retries": getattr(self, "embedding_max_retries", 5),
            "embedding_initial_backoff_sec": getattr(self, "embedding_initial_backoff_sec", 2.0),
            "embedding_max_backoff_sec": getattr(self, "embedding_max_backoff_sec", 60.0),
            "embedding_batch_size": getattr(self, "embedding_batch_size", 1),
            "embedding_deadline_sec": getattr(self, "embedding_deadline_sec", 120.0),
            "generation_timeout_sec": getattr(self, "generation_timeout_sec", 120.0),
            "api_key_configured": getattr(self, "has_api_key", False),
        }


def load_config() -> AppConfig:
    """Load configuration from environment variables or Streamlit secrets with safe defaults and validation."""
    api_key = _get_setting("GEMINI_API_KEY")

    generation_model = _get_setting("GEMINI_GENERATION_MODEL", "gemini-3.8-flash")
    generation_fallback_model = _get_setting("GEMINI_GENERATION_FALLBACK_MODEL", "gemini-3.6-flash")
    embedding_model = _get_setting("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")

    try:
        embedding_dimension = int(_get_setting("GEMINI_EMBEDDING_DIMENSION", "768"))
    except ValueError:
        embedding_dimension = 768

    try:
        max_upload_mb = int(_get_setting("MAX_UPLOAD_MB", "25"))
    except ValueError:
        max_upload_mb = 25

    try:
        max_pdf_pages = int(_get_setting("MAX_PDF_PAGES", "250"))
    except ValueError:
        max_pdf_pages = 250

    try:
        chunk_size_tokens = int(_get_setting("CHUNK_SIZE_TOKENS", "850"))
    except ValueError:
        chunk_size_tokens = 850

    try:
        chunk_overlap_tokens = int(_get_setting("CHUNK_OVERLAP_TOKENS", "120"))
    except ValueError:
        chunk_overlap_tokens = 120

    try:
        retrieval_top_k = int(_get_setting("RETRIEVAL_TOP_K", "5"))
    except ValueError:
        retrieval_top_k = 5

    try:
        min_evidence_threshold = float(_get_setting("MIN_EVIDENCE_THRESHOLD", "0.35"))
    except ValueError:
        min_evidence_threshold = 0.35

    log_level = _get_setting("LOG_LEVEL", "INFO").upper().strip()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level = "INFO"

    try:
        session_qa_budget = int(_get_setting("SESSION_QA_BUDGET", "25"))
    except ValueError:
        session_qa_budget = 25

    try:
        session_analysis_budget = int(_get_setting("SESSION_ANALYSIS_BUDGET", "5"))
    except ValueError:
        session_analysis_budget = 5

    try:
        embedding_tpm_limit = max(1, int(_get_setting("EMBEDDING_TPM_LIMIT", "30000")))
    except ValueError:
        embedding_tpm_limit = 30000

    try:
        embedding_rpm_limit = max(1, int(_get_setting("EMBEDDING_RPM_LIMIT", "60")))
    except ValueError:
        embedding_rpm_limit = 60

    try:
        margin_val = float(_get_setting("EMBEDDING_SAFETY_MARGIN", "0.80"))
        embedding_safety_margin = max(0.05, min(1.0, margin_val))
    except ValueError:
        embedding_safety_margin = 0.80

    try:
        embedding_max_retries = max(0, int(_get_setting("EMBEDDING_MAX_RETRIES", "5")))
    except ValueError:
        embedding_max_retries = 5

    try:
        embedding_initial_backoff_sec = max(0.1, float(_get_setting("EMBEDDING_INITIAL_BACKOFF_SEC", "2.0")))
    except ValueError:
        embedding_initial_backoff_sec = 2.0

    try:
        embedding_max_backoff_sec = max(
            embedding_initial_backoff_sec,
            float(_get_setting("EMBEDDING_MAX_BACKOFF_SEC", "60.0")),
        )
    except ValueError:
        embedding_max_backoff_sec = 60.0

    try:
        embedding_batch_size = max(1, int(_get_setting("EMBEDDING_BATCH_SIZE", "1")))
    except ValueError:
        embedding_batch_size = 1

    try:
        embedding_deadline_sec = max(10.0, float(_get_setting("EMBEDDING_DEADLINE_SEC", "120.0")))
    except ValueError:
        embedding_deadline_sec = 120.0

    try:
        generation_timeout_sec = max(5.0, float(_get_setting("GENERATION_TIMEOUT_SEC", "120.0")))
    except ValueError:
        generation_timeout_sec = 120.0

    return AppConfig(
        gemini_api_key=api_key,
        gemini_generation_model=generation_model,
        gemini_generation_fallback_model=generation_fallback_model,
        gemini_embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        max_upload_mb=max_upload_mb,
        max_pdf_pages=max_pdf_pages,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        retrieval_top_k=retrieval_top_k,
        min_evidence_threshold=min_evidence_threshold,
        log_level=log_level,
        session_qa_budget=session_qa_budget,
        session_analysis_budget=session_analysis_budget,
        embedding_tpm_limit=embedding_tpm_limit,
        embedding_rpm_limit=embedding_rpm_limit,
        embedding_safety_margin=embedding_safety_margin,
        embedding_max_retries=embedding_max_retries,
        embedding_initial_backoff_sec=embedding_initial_backoff_sec,
        embedding_max_backoff_sec=embedding_max_backoff_sec,
        embedding_batch_size=embedding_batch_size,
        embedding_deadline_sec=embedding_deadline_sec,
        generation_timeout_sec=generation_timeout_sec,
    )


def setup_logging(level: str = "INFO") -> None:
    """Configure standardized application logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
