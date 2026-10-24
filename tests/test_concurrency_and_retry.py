"""Unit tests verifying bounded concurrent extraction and targeted retry behavior."""

from unittest.mock import MagicMock
import pytest
import time
from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.schemas import (
    PageRecord,
    DocumentMetadata,
    BatchExtractionResult,
    FinancialMetric,
)
from financial_report_analyzer.financial_analysis_pipeline import run_financial_analysis_pipeline
from financial_report_analyzer.cache import GLOBAL_CACHE


def make_page(pno: int, text: str) -> PageRecord:
    return PageRecord(
        document_id="doc_conc_test",
        source_filename="doc.pdf",
        page_number=pno,
        text=text,
        char_count=len(text),
        has_meaningful_text=True,
    )


def test_concurrent_batch_extraction_and_cached_retry():
    """Verify concurrent batch execution, partial failure handling, and subsequent instant retry from cache."""
    GLOBAL_CACHE.clear()

    pages = [
        make_page(1, "Executive Summary: FY2024 Total Revenue $1,200M."),
        make_page(2, "Statements: Operating Income $300M, Net Income $200M."),
        make_page(3, "MD&A: Guidance and risk discussion."),
        make_page(4, "Notes: Debt and liquidity disclosures."),
    ]
    meta = DocumentMetadata(
        document_id="doc_conc_test",
        filename="doc.pdf",
        total_pages=4,
        meaningful_pages=4,
        file_size_bytes=2048,
        sha256_hash="hash_conc_test",
    )

    mock_extractor = MagicMock()
    mock_extractor.generation_model = "gemini-3.8-flash"

    # Batch 1 succeeds, Batch 2 fails initially
    res_b1 = BatchExtractionResult(
        report_title="Annual Report",
        company_name="Acme Global",
        fiscal_period="FY2024",
        metrics=[FinancialMetric(name="Revenue", value=1200.0, currency="USD", unit_scale="millions", source_page=1, source_quote="Total Revenue $1,200M", is_verified=True)],
        risks=[],
        outlook=[],
    )
    res_b2 = BatchExtractionResult(
        report_title="Annual Report",
        company_name="Acme Global",
        fiscal_period="FY2024",
        metrics=[FinancialMetric(name="Net Income", value=200.0, currency="USD", unit_scale="millions", source_page=2, source_quote="Net Income $200M", is_verified=True)],
        risks=[],
        outlook=[],
    )

    # First run: Batch 1 succeeds, Batch 2 throws transient error
    mock_extractor.extract_batch.side_effect = [
        res_b1,
        RuntimeError("Transient 503 high demand"),
    ]

    cfg = AppConfig(
        extraction_concurrency=2,
        extraction_batch_page_size=2,
        enable_section_routing=False,
    )

    analysis_1 = run_financial_analysis_pipeline(
        pages=pages,
        metadata=meta,
        config=cfg,
        extractor=mock_extractor,
        batch_page_size=2,
    )

    # Partial analysis recorded
    assert analysis_1.analysis_status == "partial"
    assert analysis_1.batches_succeeded == 1
    assert analysis_1.batches_failed == 1
    assert len(analysis_1.metrics) == 1
    assert analysis_1.metrics[0].name == "Revenue"

    # Second run (targeted retry):
    # Batch 1 is in cache (0 new calls), Batch 2 now succeeds
    mock_extractor.extract_batch.side_effect = [res_b2]

    analysis_2 = run_financial_analysis_pipeline(
        pages=pages,
        metadata=meta,
        config=cfg,
        extractor=mock_extractor,
        batch_page_size=2,
    )

    assert analysis_2.analysis_status == "complete"
    assert analysis_2.batches_succeeded == 2
    assert analysis_2.batches_failed == 0
    # Both metrics now present
    metric_names = [m.name for m in analysis_2.metrics]
    assert "Revenue" in metric_names
    assert "Net Income" in metric_names
    # Exactly 1 new call was made during retry (Batch 2), because Batch 1 hit the cache!
    assert mock_extractor.extract_batch.call_count == 3  # 2 in first run + 1 in retry
