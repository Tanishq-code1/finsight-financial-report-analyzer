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


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration container."""

    gemini_api_key: Optional[str]
    gemini_generation_model: str
    gemini_embedding_model: str
    embedding_dimension: int
    max_upload_mb: int
    max_pdf_pages: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    retrieval_top_k: int
    min_evidence_threshold: float
    log_level: str
    session_qa_budget: int = 25
    session_analysis_budget: int = 5

    @property
    def max_upload_bytes(self) -> int:
        """Maximum allowed upload file size in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        """Check if a non-empty Gemini API key is configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def safe_diagnostics(self) -> Dict[str, Any]:
        """Safe runtime diagnostic summary containing no secrets or local filesystem paths."""
        return {
            "generation_model": self.gemini_generation_model,
            "embedding_model": self.gemini_embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "max_upload_mb": self.max_upload_mb,
            "max_pdf_pages": self.max_pdf_pages,
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "min_evidence_threshold": self.min_evidence_threshold,
            "session_qa_budget": self.session_qa_budget,
            "session_analysis_budget": self.session_analysis_budget,
            "api_key_configured": self.has_api_key,
        }


def load_config() -> AppConfig:
    """Load configuration from environment variables or Streamlit secrets with safe defaults and validation."""
    api_key = _get_setting("GEMINI_API_KEY")

    generation_model = _get_setting("GEMINI_GENERATION_MODEL", "gemini-3.8-flash")
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

    return AppConfig(
        gemini_api_key=api_key,
        gemini_generation_model=generation_model,
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
    )


def setup_logging(level: str = "INFO") -> None:
    """Configure standardized application logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
