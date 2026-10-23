"""Unit tests for FinancialAnalysisPipeline, page partitioning, and deterministic analysis."""

from unittest.mock import MagicMock
import pytest

from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.schemas import (
    PageRecord,
    DocumentMetadata,
    FinancialMetric,
    RiskItem,
    OutlookItem,
    BatchExtractionResult,
)
from financial_report_analyzer.financial_analysis_pipeline import (
    partition_pages_into_batches,
    run_financial_analysis_pipeline,
)


@pytest.fixture
def mock_config():
    """Predictable configuration for testing."""
    return AppConfig(
        gemini_api_key="mock-key-for-pipeline",
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
def sample_metadata():
    """Document metadata fixture."""
    return DocumentMetadata(
        document_id="doc_acme_10k_test",
        filename="acme_10k.pdf",
        file_size_bytes=1024 * 500,
        total_pages=4,
        meaningful_pages=4,
        indexed=True,
    )


def test_partition_pages_into_batches():
    """Verify page partitioning obeys page count and character thresholds."""
    pages = [
        PageRecord(
            document_id="doc_1",
            source_filename="test.pdf",
            page_number=i,
            text="A" * 1000,
            char_count=1000,
            has_meaningful_text=True,
        )
        for i in range(1, 11)
    ]

    # Partition with max 3 pages per batch
    batches = partition_pages_into_batches(pages, max_pages=3, max_chars=10000)
    assert len(batches) == 4
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 1

    # Partition with max 2500 chars (at most 2 pages per batch)
    batches_chars = partition_pages_into_batches(pages, max_pages=10, max_chars=2500)
    assert len(batches_chars) == 5
    for b in batches_chars:
        assert len(b) == 2


def test_pipeline_zero_readable_pages(mock_config, sample_metadata):
    """Verify pipeline fails gracefully when all pages are unreadable."""
    pages = [
        PageRecord(
            document_id="doc_1",
            source_filename="test.pdf",
            page_number=1,
            text="",
            char_count=0,
            has_meaningful_text=False,
        )
    ]
    analysis = run_financial_analysis_pipeline(
        pages=pages,
        metadata=sample_metadata,
        config=mock_config,
    )
    assert analysis.analysis_status == "failed"
    assert analysis.pages_analyzed == 0
    assert any("zero readable pages" in lim for lim in analysis.limitations)


def test_pipeline_end_to_end_with_fake_extractor(mock_config, sample_metadata):
    """Verify pipeline runs through batches, computes YoY in Python, and derives Free Cash Flow."""
    pages = [
        PageRecord(
            document_id="doc_1",
            source_filename="test.pdf",
            page_number=1,
            text="Income statement: Net sales was $100,000 million in FY2024 and $80,000 million in FY2023.",
            char_count=100,
            has_meaningful_text=True,
        ),
        PageRecord(
            document_id="doc_1",
            source_filename="test.pdf",
            page_number=2,
            text="Cash flows: Operating cash flow was $30,000 million. Capital expenditures were $10,000 million.",
            char_count=100,
            has_meaningful_text=True,
        ),
    ]

    fake_extractor = MagicMock()
    # Batch 1 (page 1) returns Revenue
    res_b1 = BatchExtractionResult(
        company_name="Acme Corporation",
        fiscal_period="FY2024",
        metrics=[
            FinancialMetric(
                name="Total Net Sales",
                normalized_name="revenue",
                value=100000.0,
                currency="USD",
                unit_scale="millions",
                period_label="FY2024",
                prior_period_value=80000.0,
                prior_period_label="FY2023",
                source_page=1,
                source_quote="Net sales was $100,000 million in FY2024",
                is_verified=True,
            )
        ],
        risks=[
            RiskItem(
                risk_title="Supply Chain Risk",
                description="Shortages may increase component costs.",
                source_pages=[1],
                source_quotes=["Shortages may increase component costs."],
                is_verified=True,
            )
        ],
        outlook=[],
        executive_summary_excerpts=["Acme reported robust growth."],
    )
    # Batch 2 (page 2) returns OCF and Capex
    res_b2 = BatchExtractionResult(
        metrics=[
            FinancialMetric(
                name="Operating Cash Flow",
                normalized_name="operating_cash_flow",
                value=30000.0,
                currency="USD",
                unit_scale="millions",
                period_label="FY2024",
                source_page=2,
                source_quote="Operating cash flow was $30,000 million",
                is_verified=True,
            ),
            FinancialMetric(
                name="Capital Expenditures",
                normalized_name="capex",
                value="(10000.0)",
                currency="USD",
                unit_scale="millions",
                period_label="FY2024",
                source_page=2,
                source_quote="Capital expenditures were $10,000 million",
                is_verified=True,
            ),
        ],
        risks=[],
        outlook=[
            OutlookItem(
                statement="Management anticipates 10% operating margin in FY2025.",
                period="FY2025",
                statement_type="guidance",
                source_pages=[2],
                source_quotes=["Management anticipates 10% operating margin"],
                is_verified=True,
            )
        ],
        executive_summary_excerpts=[],
    )
    fake_extractor.extract_batch.side_effect = [res_b1, res_b2]

    progress_calls = []

    def on_progress(pct, msg):
        progress_calls.append((pct, msg))

    analysis = run_financial_analysis_pipeline(
        pages=pages,
        metadata=sample_metadata,
        config=mock_config,
        extractor=fake_extractor,
        progress_callback=on_progress,
        batch_page_size=1,  # Force 2 batches
    )

    assert analysis.analysis_status == "complete"
    assert analysis.company_name == "Acme Corporation"
    assert analysis.batches_succeeded == 2
    assert analysis.batches_failed == 0
    assert len(progress_calls) > 0

    # Verify YoY calculation for Revenue: 100,000 vs 80,000 = +20,000 (+25.00%)
    rev = next(m for m in analysis.metrics if m.normalized_name == "revenue")
    assert rev.yoy_absolute_change == 20000.0
    assert rev.yoy_percentage_change == 25.0
    assert "+25.00%" in rev.yoy_calculation_note

    # Verify deterministic derivation of Free Cash Flow: 30,000 - 10,000 = 20,000
    fcf = next((m for m in analysis.metrics if m.name == "Free Cash Flow"), None)
    assert fcf is not None
    assert fcf.value == 20000.0
    assert fcf.metric_type == "calculated"
    assert "Operating Cash Flow - Capital Expenditures" in fcf.calculation_method

    # Verify risks and outlook
    assert len(analysis.risks) == 1
    assert len(analysis.outlook) == 1
    assert analysis.verification_rate == 1.0


def test_pipeline_handles_batch_failure_as_partial(mock_config, sample_metadata):
    """Verify pipeline continues and reports partial analysis if one batch fails."""
    pages = [
        PageRecord(
            document_id="doc_1",
            source_filename="test.pdf",
            page_number=1,
            text="Net sales was $50,000 million in FY2024.",
            char_count=50,
            has_meaningful_text=True,
        ),
        PageRecord(
            document_id="doc_1",
            source_filename="test.pdf",
            page_number=2,
            text="Risk items disclosures page.",
            char_count=50,
            has_meaningful_text=True,
        ),
    ]

    fake_extractor = MagicMock()
    res_b1 = BatchExtractionResult(
        company_name="Beta Corp",
        metrics=[
            FinancialMetric(
                name="Revenue",
                value=50000.0,
                source_page=1,
                is_verified=True,
            )
        ],
    )
    # Batch 1 succeeds, Batch 2 fails with API rate limit / error
    fake_extractor.extract_batch.side_effect = [res_b1, RuntimeError("API quota exhausted")]

    analysis = run_financial_analysis_pipeline(
        pages=pages,
        metadata=sample_metadata,
        config=mock_config,
        extractor=fake_extractor,
        batch_page_size=1,
    )

    assert analysis.analysis_status == "partial"
    assert analysis.batches_succeeded == 1
    assert analysis.batches_failed == 1
    # Batch 1 metrics must NOT be lost!
    assert len(analysis.metrics) == 1
    assert analysis.metrics[0].value == 50000.0
    assert any("API quota exhausted" in lim for lim in analysis.limitations)
