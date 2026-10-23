"""Unit tests for FinancialExtractor, structured output parsing, and citation verification."""

import json
from unittest.mock import MagicMock
import pytest

from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.errors import ConfigurationError, RAGGenerationError
from financial_report_analyzer.schemas import (
    PageRecord,
    FinancialMetric,
    RiskItem,
    OutlookItem,
    BatchExtractionResult,
)
from financial_report_analyzer.financial_extractor import (
    FinancialExtractor,
    format_batch_pages,
)


@pytest.fixture
def mock_config():
    """Predictable configuration for testing."""
    return AppConfig(
        gemini_api_key="mock-key-for-extractor",
        gemini_generation_model="gemini-3.8-flash",
        gemini_embedding_model="gemini-embedding-2",
        embedding_dimension=768,
        max_upload_mb=25,
        max_pdf_pages=250,
        chunk_size_tokens=850,
        chunk_overlap_tokens=120,
        retrieval_top_k=5,
        min_evidence_threshold=0.35,
        log_level="DEBUG",
    )


@pytest.fixture
def sample_pages():
    """Synthetic PageRecord fixtures representing financial statements."""
    p1 = PageRecord(
        document_id="doc_test_10k",
        source_filename="acme_10k.pdf",
        page_number=14,
        text=(
            "ACME CORP - CONSOLIDATED STATEMENTS OF OPERATIONS\n"
            "Total net sales was $96,770 million for FY2024, compared to $84,150 million for FY2023.\n"
            "Operating income was $28,500 million in FY2024.\n"
            "Net income was $22,100 million in FY2024."
        ),
        char_count=230,
        has_meaningful_text=True,
    )
    p2 = PageRecord(
        document_id="doc_test_10k",
        source_filename="acme_10k.pdf",
        page_number=28,
        text=(
            "ITEM 1A. RISK FACTORS\n"
            "Disruptions in semiconductor supply chains could adversely affect operating results. "
            "We maintain multi-source vendor agreements to mitigate supply single-points of failure.\n"
            "OUTLOOK: Management expects high single-digit revenue growth in FY2025."
        ),
        char_count=270,
        has_meaningful_text=True,
    )
    return [p1, p2]


def test_format_batch_pages(sample_pages):
    """Verify page formatting wraps text in page boundary tags."""
    formatted = format_batch_pages(sample_pages)
    assert '<page number="14" source="acme_10k.pdf">' in formatted
    assert '<page number="28" source="acme_10k.pdf">' in formatted
    assert "</page>" in formatted
    assert "$96,770 million" in formatted


def test_unconfigured_extractor_raises_configuration_error(sample_pages):
    """Verify extractor raises ConfigurationError if API key is missing."""
    cfg = AppConfig(
        gemini_api_key=None,
        gemini_generation_model="gemini-3.8-flash",
        gemini_embedding_model="gemini-embedding-2",
        embedding_dimension=768,
        max_upload_mb=25,
        max_pdf_pages=250,
        chunk_size_tokens=850,
        chunk_overlap_tokens=120,
        retrieval_top_k=5,
        min_evidence_threshold=0.35,
        log_level="INFO",
    )
    ext = FinancialExtractor(config=cfg)
    assert ext.is_configured is False
    with pytest.raises(ConfigurationError):
        ext.extract_batch(sample_pages)


def test_successful_batch_extraction(mock_config, sample_pages):
    """Verify successful structured extraction and citation verification."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "report_title": "Acme Corp Form 10-K",
            "company_name": "Acme Corp",
            "fiscal_period": "FY2024",
            "metrics": [
                {
                    "name": "Revenue",
                    "normalized_name": "revenue",
                    "value": 96770.0,
                    "currency": "USD",
                    "unit_scale": "millions",
                    "period_label": "FY2024",
                    "prior_period_value": 84150.0,
                    "prior_period_label": "FY2023",
                    "source_page": 14,
                    "source_quote": "Total net sales was $96,770 million for FY2024",
                    "status": "found",
                }
            ],
            "risks": [
                {
                    "risk_title": "Semiconductor Supply Chain Disruption",
                    "description": "Disruptions in semiconductor supply chains could adversely affect operating results.",
                    "risk_category": "Supply Chain",
                    "company_stated_mitigation": "We maintain multi-source vendor agreements to mitigate supply single-points of failure.",
                    "source_pages": [28],
                    "source_quotes": ["Disruptions in semiconductor supply chains could adversely affect operating results."],
                }
            ],
            "outlook": [
                {
                    "statement": "Management expects high single-digit revenue growth in FY2025.",
                    "period": "FY2025",
                    "statement_type": "guidance",
                    "source_pages": [28],
                    "source_quotes": ["Management expects high single-digit revenue growth in FY2025."],
                }
            ],
            "executive_summary_excerpts": ["Acme Corp reported record sales in FY2024."],
        }
    )
    mock_client.models.generate_content.return_value = mock_response

    ext = FinancialExtractor(config=mock_config, client=mock_client)
    res = ext.extract_batch(sample_pages, document_id="doc_test_10k")

    assert res.company_name == "Acme Corp"
    assert len(res.metrics) == 1
    assert res.metrics[0].is_verified is True
    assert res.metrics[0].value == 96770.0

    assert len(res.risks) == 1
    assert res.risks[0].is_verified is True
    assert "multi-source vendor" in res.risks[0].company_stated_mitigation

    assert len(res.outlook) == 1
    assert res.outlook[0].is_verified is True


def test_citation_verification_rejects_hallucinations(mock_config, sample_pages):
    """Verify that ungrounded quotations or numbers are flagged as unverified."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "report_title": "Acme Corp Form 10-K",
            "company_name": "Acme Corp",
            "fiscal_period": "FY2024",
            "metrics": [
                {
                    "name": "Revenue",
                    "normalized_name": "revenue",
                    "value": 500000.0,  # Fabricated number not on page 14
                    "currency": "USD",
                    "unit_scale": "millions",
                    "period_label": "FY2024",
                    "source_page": 14,
                    "source_quote": "Fabricated revenue sentence",
                    "status": "found",
                }
            ],
            "risks": [
                {
                    "risk_title": "Asteroid Collision Risk",
                    "description": "An asteroid will destroy the factory.",
                    "source_pages": [14],
                    "source_quotes": ["Asteroid impact imminent"],  # Not on page 14
                }
            ],
            "outlook": [],
            "executive_summary_excerpts": [],
        }
    )
    mock_client.models.generate_content.return_value = mock_response

    ext = FinancialExtractor(config=mock_config, client=mock_client)
    res = ext.extract_batch(sample_pages, document_id="doc_test_10k")

    # Fabricated metric must be flagged as unverified
    assert res.metrics[0].is_verified is False
    assert "could not be confirmed" in res.metrics[0].confidence_note

    # Fabricated risk must be flagged as unverified
    assert res.risks[0].is_verified is False


def test_markdown_code_fence_handling(mock_config, sample_pages):
    """Verify JSON wrapped in markdown ```json ... ``` fences is parsed."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    raw_payload = json.dumps(
        {
            "report_title": "Acme 10-K",
            "company_name": "Acme",
            "fiscal_period": "FY2024",
            "metrics": [],
            "risks": [],
            "outlook": [],
            "executive_summary_excerpts": [],
        }
    )
    mock_response.text = f"```json\n{raw_payload}\n```"
    mock_client.models.generate_content.return_value = mock_response

    ext = FinancialExtractor(config=mock_config, client=mock_client)
    res = ext.extract_batch(sample_pages, document_id="doc_test_10k")
    assert res.company_name == "Acme"


def test_extractor_handles_timeout_gracefully(mock_config, sample_pages):
    """Verify that timeout exceptions from client generate friendly RAGGenerationError."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("The read operation timed out")

    ext = FinancialExtractor(config=mock_config, client=mock_client, timeout_sec=10.0)
    with pytest.raises(RAGGenerationError) as exc_info:
        ext.extract_batch(sample_pages, document_id="doc_test_10k")

    assert "timed out" in str(exc_info.value).lower()
    assert hasattr(exc_info.value, "user_friendly_message")
    assert "timed out" in exc_info.value.user_friendly_message.lower()


def test_extractor_respects_custom_timeout(mock_config):
    """Verify custom timeout setting is retained on FinancialExtractor."""
    custom_cfg = AppConfig(
        gemini_api_key="mock-key",
        generation_timeout_sec=45.0,
    )
    ext = FinancialExtractor(config=custom_cfg)
    assert ext.timeout_sec == 45.0
