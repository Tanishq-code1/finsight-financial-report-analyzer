"""Unit tests for CitationValidator and quote verification."""

import pytest
from financial_report_analyzer.citation_validator import (
    CitationValidator,
    normalize_text_for_matching,
    extract_key_numbers,
    quote_matches_chunk_text,
)
from financial_report_analyzer.schemas import Citation, ChunkRecord, RetrievalResult


@pytest.fixture
def sample_retrieved_chunks():
    """Sample retrieved chunks with typical financial text."""
    c1 = RetrievalResult(
        chunk_id="chunk_rev_01",
        document_id="doc_acme_2024",
        source_filename="acme_corp_10k.pdf",
        page_number=14,
        text=(
            "Total revenue for the fiscal year ended December 31, 2024, was $96,770 million, "
            "an increase of 15% compared to $84,150 million in fiscal year 2023. "
            "The growth was primarily driven by higher enterprise cloud services adoption."
        ),
        distance=0.15,
        similarity=0.925,
        chunk_index=3,
        metadata={"page_number": 14, "page_numbers": [14]},
    )
    c2 = RetrievalResult(
        chunk_id="chunk_risk_02",
        document_id="doc_acme_2024",
        source_filename="acme_corp_10k.pdf",
        page_number=28,
        text=(
            "We face intense competition in our core markets. "
            "Disruptions in global semiconductor supply chains could adversely affect "
            "our hardware gross margins and operating cash flow."
        ),
        distance=0.30,
        similarity=0.85,
        chunk_index=8,
        metadata={"page_number": 28, "page_numbers": [28, 29]},
    )
    return [c1, c2]


def test_valid_exact_citation(sample_retrieved_chunks):
    """Verify that an exact verbatim quote passes validation."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_rev_01",
            page_number=14,
            quote="Total revenue for the fiscal year ended December 31, 2024, was $96,770 million",
            source_filename="acme_corp_10k.pdf",
        )
    ]
    report = validator.validate_citations(citations, sample_retrieved_chunks)
    assert report.is_fully_valid is True
    assert report.total_count == 1
    assert report.valid_count == 1
    assert report.invalid_count == 0
    assert report.verification_rate == 1.0


def test_valid_normalized_quote(sample_retrieved_chunks):
    """Verify that quotes with differing whitespace or smart quotes pass normalization."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_rev_01",
            page_number=14,
            quote="“Total revenue for the fiscal year ended December 31, 2024,   was $96,770 million”",
            source_filename="",
        )
    ]
    report = validator.validate_citations(citations, sample_retrieved_chunks)
    assert report.is_fully_valid is True
    assert report.valid_count == 1
    # Check that source_filename was auto-populated
    assert report.valid_citations[0].source_filename == "acme_corp_10k.pdf"


def test_quote_with_ellipsis(sample_retrieved_chunks):
    """Verify that quotes with an ellipsis between phrases pass validation."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_risk_02",
            page_number=28,
            quote="Disruptions in global semiconductor supply chains ... operating cash flow",
            source_filename="acme_corp_10k.pdf",
        )
    ]
    report = validator.validate_citations(citations, sample_retrieved_chunks)
    assert report.is_fully_valid is True
    assert report.valid_count == 1


def test_invalid_chunk_id(sample_retrieved_chunks):
    """Verify that citations with hallucinated chunk IDs are rejected."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_nonexistent_99",
            page_number=14,
            quote="Total revenue was $96,770 million",
            source_filename="acme_corp_10k.pdf",
        )
    ]
    report = validator.validate_citations(citations, sample_retrieved_chunks)
    assert report.is_fully_valid is False
    assert report.valid_count == 0
    assert report.invalid_count == 1
    assert "not in the retrieved passages" in report.items[0].reason


def test_page_number_mismatch(sample_retrieved_chunks):
    """Verify that citations with incorrect page numbers are rejected."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_rev_01",
            page_number=99,  # Chunk is on page 14
            quote="Total revenue for the fiscal year ended December 31, 2024, was $96,770 million",
            source_filename="acme_corp_10k.pdf",
        )
    ]
    report = validator.validate_citations(citations, sample_retrieved_chunks)
    assert report.is_fully_valid is False
    assert report.valid_count == 0
    assert report.invalid_count == 1
    assert "does not match chunk" in report.items[0].reason


def test_fabricated_number_rejected(sample_retrieved_chunks):
    """Verify that quotes with hallucinated financial numbers are strictly rejected."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_rev_01",
            page_number=14,
            quote="Total revenue for the fiscal year was $150,000 million",  # Actual is $96,770M
            source_filename="acme_corp_10k.pdf",
        )
    ]
    report = validator.validate_citations(citations, sample_retrieved_chunks)
    assert report.is_fully_valid is False
    assert report.valid_count == 0
    assert "Quote contains figure '150000' not found" in report.items[0].reason


def test_fabricated_text_rejected(sample_retrieved_chunks):
    """Verify that completely made-up quotations are rejected."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_risk_02",
            page_number=28,
            quote="The company acquired a space exploration division in Antarctica",
            source_filename="acme_corp_10k.pdf",
        )
    ]
    report = validator.validate_citations(citations, sample_retrieved_chunks)
    assert report.is_fully_valid is False
    assert report.valid_count == 0
    assert report.invalid_count == 1


def test_filter_and_annotate(sample_retrieved_chunks):
    """Verify filtering retains only valid citations and generates limitation notes for excluded ones."""
    validator = CitationValidator()
    citations = [
        Citation(
            chunk_id="chunk_rev_01",
            page_number=14,
            quote="Total revenue for the fiscal year ended December 31, 2024, was $96,770 million",
            source_filename="acme_corp_10k.pdf",
        ),
        Citation(
            chunk_id="chunk_fake_99",
            page_number=14,
            quote="Some fake text",
            source_filename="acme_corp_10k.pdf",
        ),
    ]
    valid_cits, limitations = validator.filter_and_annotate(citations, sample_retrieved_chunks)
    assert len(valid_cits) == 1
    assert valid_cits[0].chunk_id == "chunk_rev_01"
    assert len(limitations) == 1
    assert "1 of 2 citation(s) could not be verified" in limitations[0]


def test_empty_citations(sample_retrieved_chunks):
    """Verify empty citations list returns empty valid report cleanly."""
    validator = CitationValidator()
    report = validator.validate_citations([], sample_retrieved_chunks)
    assert report.is_fully_valid is True
    assert report.total_count == 0
