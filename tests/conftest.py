"""Global test fixtures and configuration for Financial Report Analyzer."""

import os
import sys
from pathlib import Path
import pytest

# Ensure `src` is in the Python search path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_config_env(monkeypatch: pytest.MonkeyPatch):
    """Provide a predictable test environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    monkeypatch.setenv("MAX_UPLOAD_MB", "15")
    monkeypatch.setenv("MAX_PDF_PAGES", "100")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "500")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "100")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "4")
    monkeypatch.setenv("MIN_EVIDENCE_THRESHOLD", "0.40")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
