"""Unit tests for Pydantic data schemas and contract enforcement."""

import pytest
from pydantic import ValidationError
from financial_report_analyzer.schemas import (
    PageRecord,
    ChunkRecord,
    Citation,
    RAGAnswer,
    FinancialMetric,
    RiskItem,
    OutlookItem,
    ReportAnalysis,
    DocumentMetadata,
)


def test_page_record_valid():
    """Verify PageRecord schema validation with 1-based page indices."""
    record = PageRecord(
        document_id="doc-123",
        source_filename="apple_10k.pdf",
        page_number=1,
        text="Sample financial statement text",
        char_count=31,
        has_meaningful_text=True,
    )
    assert record.page_number == 1
    assert record.char_count == 31


def test_page_record_invalid_page():
    """Verify PageRecord rejects non-positive page numbers."""
    with pytest.raises(ValidationError):
        PageRecord(
            document_id="doc-123",
            source_filename="test.pdf",
            page_number=0,  # Invalid: page numbers must be >= 1
            text="Text",
            char_count=4,
            has_meaningful_text=True,
        )


def test_chunk_record():
    """Verify ChunkRecord schema and provenance fields."""
    chunk = ChunkRecord(
        chunk_id="doc-123_c0",
        document_id="doc-123",
        source_filename="report.pdf",
        page_numbers=[1, 2],
        section_title="Management Discussion & Analysis",
        chunk_index=0,
        text="Revenue grew by 8% year-over-year to $400 billion.",
        embedding=[0.1, 0.2, 0.3],
    )
    assert chunk.chunk_id == "doc-123_c0"
    assert chunk.page_numbers == [1, 2]
    assert len(chunk.embedding) == 3


def test_citation_validation():
    """Verify Citation schema structure."""
    cit = Citation(
        chunk_id="c_001",
        page_number=14,
        quote="Net sales increased 5% in FY2024",
        source_filename="report.pdf",
    )
    assert cit.page_number == 14
    assert "Net sales" in cit.quote


def test_rag_answer_abstention():
    """Verify RAGAnswer representation for insufficient evidence."""
    ans = RAGAnswer(
        answer="The report does not provide sufficient evidence regarding Q4 dividend plans.",
        citations=[],
        insufficient_evidence=True,
        limitations=["No forward dividend guidance found in the filing."],
    )
    assert ans.insufficient_evidence is True
    assert len(ans.citations) == 0


def test_financial_metric_missing_not_zero():
    """Verify missing financial metric is not converted to 0.0."""
    metric = FinancialMetric(
        name="Total Debt",
        value=None,
        currency="USD",
        unit_scale="millions",
        period_label="FY2024",
        metric_type="reported",
        source_page=None,
        source_quote=None,
        status="not_found",
    )
    assert metric.value is None
    assert metric.status == "not_found"


def test_financial_metric_reported():
    """Verify reported financial metric validation."""
    metric = FinancialMetric(
        name="Revenue",
        value=383285.0,
        currency="USD",
        unit_scale="millions",
        period_label="FY2024",
        metric_type="reported",
        source_page=32,
        source_quote="Total net sales was $383,285 million in 2024",
        status="found",
    )
    assert metric.value == 383285.0
    assert metric.source_page == 32
    assert metric.status == "found"


def test_report_analysis_complete():
    """Verify top-level ReportAnalysis schema."""
    analysis = ReportAnalysis(
        report_title="Apple Inc. Form 10-K",
        company_name="Apple Inc.",
        report_type="10-K",
        fiscal_period="FY2024",
        executive_summary="Apple delivered record services revenue...",
        key_takeaways=["Services reached an all-time high.", "Gross margin expanded."],
        metrics=[],
        risks=[
            RiskItem(
                risk_title="Foreign Exchange Fluctuations",
                description="Strengthening USD may unfavourably impact international revenues.",
                source_pages=[18],
                source_quotes=["Fluctuations in foreign currencies may adversely impact revenue."],
            )
        ],
        outlook=[
            OutlookItem(
                statement="Management anticipates continued services expansion in FY2025.",
                period="FY2025",
                statement_type="guidance",
                source_pages=[25],
                source_quotes=["We expect services to remain a strong growth driver."],
            )
        ],
        limitations=["Scan quality warning on page 4"],
        generated_at="2026-10-23T01:00:00Z",
    )
    assert analysis.company_name == "Apple Inc."
    assert len(analysis.risks) == 1
    assert len(analysis.outlook) == 1
