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


def test_generation_config_disables_afc_and_caps_http_retries(mock_config):
    """Verify GenerateContentConfig explicitly disables Automatic Function Calling and sets HttpRetryOptions attempts=1."""
    ext = FinancialExtractor(config=mock_config)
    cfg = ext._build_generation_config()

    # Automatic Function Calling must be explicitly disabled
    assert cfg.automatic_function_calling is not None
    assert cfg.automatic_function_calling.disable is True
    # Tools and tool_config must be None
    assert cfg.tools is None
    assert cfg.tool_config is None
    # HTTP retries must be capped at 1 attempt to avoid internal tenacity loops
    assert cfg.http_options is not None
    assert cfg.http_options.retry_options is not None
    assert cfg.http_options.retry_options.attempts == 1


def test_503_primary_retry_succeeds(mock_config, sample_pages):
    """Verify that a transient 503 on primary model is retried and succeeds without fallback."""
    from google.genai import errors as genai_errors

    mock_client = MagicMock()
    err_503 = genai_errors.APIError(503, {"error": {"message": "High demand. Please try again later."}})

    mock_response = MagicMock()
    mock_response.text = json.dumps(
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

    # First call 503, second call succeeds
    mock_client.models.generate_content.side_effect = [err_503, mock_response]

    ext = FinancialExtractor(config=mock_config, client=mock_client, initial_backoff=0.01)
    res = ext.extract_batch(sample_pages, document_id="doc_test_10k")

    assert res.company_name == "Acme"
    assert res.fallback_used is False
    assert res.model_used == "gemini-3.8-flash"
    assert mock_client.models.generate_content.call_count == 2


def test_503_primary_exhausted_fallback_succeeds(mock_config, sample_pages):
    """Verify that persistent 503 on primary switches to fallback model and succeeds."""
    from google.genai import errors as genai_errors

    mock_client = MagicMock()
    err_503 = genai_errors.APIError(503, {"error": {"message": "Model gemini-3.8-flash high demand"}})

    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "report_title": "Acme 10-K",
            "company_name": "Acme Fallback",
            "fiscal_period": "FY2024",
            "metrics": [],
            "risks": [],
            "outlook": [],
            "executive_summary_excerpts": [],
        }
    )

    # Primary attempt 1 and 2 fail with 503; fallback attempt succeeds
    mock_client.models.generate_content.side_effect = [err_503, err_503, mock_response]

    ext = FinancialExtractor(
        config=mock_config,
        client=mock_client,
        initial_backoff=0.01,
        fallback_model="gemini-3.6-flash",
    )
    res = ext.extract_batch(sample_pages, document_id="doc_test_10k")

    assert res.company_name == "Acme Fallback"
    assert res.fallback_used is True
    assert res.model_used == "gemini-3.6-flash"
    assert mock_client.models.generate_content.call_count == 3
    # Verify the third call used fallback model
    third_call_kwargs = mock_client.models.generate_content.call_args_list[2].kwargs
    assert third_call_kwargs["model"] == "gemini-3.6-flash"


def test_503_primary_and_fallback_both_fail(mock_config, sample_pages):
    """Verify that if both primary and fallback experience 503, a clean RAGGenerationError is raised."""
    from google.genai import errors as genai_errors

    mock_client = MagicMock()
    err_503 = genai_errors.APIError(503, {"error": {"message": "Service unavailable across models"}})
    mock_client.models.generate_content.side_effect = err_503

    ext = FinancialExtractor(
        config=mock_config,
        client=mock_client,
        initial_backoff=0.01,
        fallback_model="gemini-3.6-flash",
    )
    with pytest.raises(RAGGenerationError) as exc_info:
        ext.extract_batch(sample_pages, document_id="doc_test_10k")

    assert "503" in str(exc_info.value)
    assert "high demand" in exc_info.value.user_friendly_message.lower() or "unavailable" in exc_info.value.user_friendly_message.lower()


def test_429_daily_quota_fails_immediately_without_fallback(mock_config, sample_pages):
    """Verify that HTTP 429 daily quota exhaustion does not retry or fallback."""
    from google.genai import errors as genai_errors

    mock_client = MagicMock()
    err_429 = genai_errors.APIError(
        429,
        {"error": {"message": "Daily quota exceeded for embed/generate", "status": "RESOURCE_EXHAUSTED"}},
    )
    mock_client.models.generate_content.side_effect = err_429

    ext = FinancialExtractor(
        config=mock_config,
        client=mock_client,
        initial_backoff=0.01,
        fallback_model="gemini-3.6-flash",
    )
    with pytest.raises(RAGGenerationError) as exc_info:
        ext.extract_batch(sample_pages, document_id="doc_test_10k")

    assert "daily quota" in str(exc_info.value).lower()
    # Call count should be exactly 1: no retry, no fallback
    assert mock_client.models.generate_content.call_count == 1


def test_401_auth_error_fails_immediately_without_fallback(mock_config, sample_pages):
    """Verify that HTTP 401 authentication error raises ConfigurationError immediately without retrying or fallback."""
    from google.genai import errors as genai_errors

    mock_client = MagicMock()
    err_401 = genai_errors.APIError(401, {"error": {"message": "API key not valid"}})
    mock_client.models.generate_content.side_effect = err_401

    ext = FinancialExtractor(
        config=mock_config,
        client=mock_client,
        initial_backoff=0.01,
        fallback_model="gemini-3.6-flash",
    )
    with pytest.raises(ConfigurationError):
        ext.extract_batch(sample_pages, document_id="doc_test_10k")

    assert mock_client.models.generate_content.call_count == 1
