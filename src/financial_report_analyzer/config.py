"""Application configuration management with environment variable support."""

import os
import logging
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()


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

    @property
    def max_upload_bytes(self) -> int:
        """Maximum allowed upload file size in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        """Check if a non-empty Gemini API key is configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


def load_config() -> AppConfig:
    """Load configuration from environment variables with safe defaults and validation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        api_key = api_key.strip()
        if not api_key:
            api_key = None

    generation_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.8-flash").strip()
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()

    try:
        embedding_dimension = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))
    except ValueError:
        embedding_dimension = 768

    try:
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "25"))
    except ValueError:
        max_upload_mb = 25

    try:
        max_pdf_pages = int(os.getenv("MAX_PDF_PAGES", "250"))
    except ValueError:
        max_pdf_pages = 250

    try:
        chunk_size_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", "850"))
    except ValueError:
        chunk_size_tokens = 850

    try:
        chunk_overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "120"))
    except ValueError:
        chunk_overlap_tokens = 120

    try:
        retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    except ValueError:
        retrieval_top_k = 5

    try:
        min_evidence_threshold = float(os.getenv("MIN_EVIDENCE_THRESHOLD", "0.35"))
    except ValueError:
        min_evidence_threshold = 0.35

    log_level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level = "INFO"

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
    )


def setup_logging(level: str = "INFO") -> None:
    """Configure standardized application logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
