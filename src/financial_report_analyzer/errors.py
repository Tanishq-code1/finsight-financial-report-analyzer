"""Domain exception hierarchy for Financial Report Analyzer."""

from typing import Optional


class FinancialAnalyzerError(Exception):
    """Base exception for all domain errors in Financial Report Analyzer."""

    def __init__(self, message: str, user_friendly_message: Optional[str] = None) -> None:
        super().__init__(message)
        self.user_friendly_message = user_friendly_message or message


class ConfigurationError(FinancialAnalyzerError):
    """Raised when environment variables or application configuration are invalid."""


class DocumentValidationError(FinancialAnalyzerError):
    """Raised when an uploaded document violates size, page count, format, or readability constraints."""


class PDFParsingError(FinancialAnalyzerError):
    """Raised when PyMuPDF fails to extract or parse document structure."""


class EmbeddingError(FinancialAnalyzerError):
    """Raised when embedding generation fails or vector counts do not match."""


class VectorStoreError(FinancialAnalyzerError):
    """Raised when an ephemeral vector collection operation fails."""


class RetrievalError(FinancialAnalyzerError):
    """Raised when similarity search or candidate filtering fails."""


class RAGGenerationError(FinancialAnalyzerError):
    """Raised when LLM answer generation or structured parsing fails."""


class CitationValidationError(FinancialAnalyzerError):
    """Raised when returned citations fail provenance, chunk ID, or substring verification."""
