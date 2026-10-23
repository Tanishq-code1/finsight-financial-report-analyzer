"""Comprehensive unit tests for the sentence-aware, page-scoped chunker."""

import pytest
from financial_report_analyzer.schemas import PageRecord
from financial_report_analyzer.chunker import (
    chunk_pages,
    chunk_single_page,
    estimate_tokens,
    split_into_sentence_units,
    split_long_unit,
    ChunkerConfig,
)


def make_page(
    page_number: int = 1,
    text: str = "",
    doc_id: str = "doc_test123",
    filename: str = "report.pdf",
) -> PageRecord:
    """Helper to construct PageRecord fixtures."""
    return PageRecord(
        document_id=doc_id,
        source_filename=filename,
        page_number=page_number,
        text=text,
        char_count=len(text),
        has_meaningful_text=len(text.strip()) > 30,
    )


def test_estimate_tokens_empty_and_normal():
    """Verify deterministic token estimation."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    # 4 characters ~= 1 token
    assert estimate_tokens("word") == 1
    assert estimate_tokens("a" * 100) == 25


def test_empty_page_produces_no_chunks():
    """Verify empty or whitespace-only page text produces zero chunks without crashing."""
    p_empty = make_page(page_number=1, text="")
    p_whitespace = make_page(page_number=2, text="    \n\n   ")

    chunks = chunk_pages([p_empty, p_whitespace])
    assert len(chunks) == 0


def test_short_paragraph_page():
    """Verify page with short text produces exactly 1 chunk retaining full text and page number."""
    short_text = "Consolidated Statements of Operations: Net revenue was $50,000,000."
    p = make_page(page_number=3, text=short_text)

    chunks = chunk_pages([p])
    assert len(chunks) == 1
    assert chunks[0].page_numbers == [3]
    assert chunks[0].text == short_text
    assert chunks[0].chunk_id == "doc_test123_p3_c0"
    assert chunks[0].chunk_index == 0


def test_long_page_requires_multiple_chunks_with_overlap():
    """Verify dense page is partitioned into multiple chunks with overlap."""
    sentences = [
        f"Sentence number {i} provides detailed discussion regarding financial disclosure and operations."
        for i in range(1, 60)
    ]
    long_text = "\n\n".join(sentences)
    p = make_page(page_number=5, text=long_text)

    # Use small target to force multi-chunk splitting
    cfg = ChunkerConfig(target_tokens=100, max_tokens=150, overlap_tokens=30)
    chunks = chunk_single_page(p, cfg, starting_chunk_index=0)

    assert len(chunks) > 1
    for c in chunks:
        assert c.page_numbers == [5]
        assert c.document_id == "doc_test123"
        assert len(c.text) > 0

    # Verify overlap exists between chunk 0 and chunk 1
    c0_words = set(chunks[0].text.split())
    c1_words = set(chunks[1].text.split())
    common_words = c0_words.intersection(c1_words)
    assert len(common_words) > 5


def test_sentence_boundary_splitting_protects_financial_decimals():
    """Verify sentences split at punctuation without splitting decimal figures."""
    text = (
        "Operating margin was 14.5% in 2024. Total revenue reached $383.28 billion. "
        "EPS expanded from $6.08 to $6.13 per share. Growth remained strong."
    )
    units = split_into_sentence_units(text)
    assert len(units) == 4
    assert "$383.28 billion." in units[1]
    assert "$6.08 to $6.13" in units[2]


def test_long_unbroken_text_partition():
    """Verify long unbroken text or giant strings are safely split."""
    giant_word = "X" * 1200
    pieces = split_long_unit(giant_word, max_chars=400)
    assert len(pieces) == 3
    assert all(len(piece) <= 400 for piece in pieces)
    assert "".join(pieces) == giant_word


def test_financial_table_rows_preserved():
    """Verify financial table lines are kept intact as row units."""
    table_text = (
        "Consolidated Balance Sheet:\n"
        "Cash and equivalents       $29,965     $28,400\n"
        "Marketable securities     $31,590     $31,200\n"
        "Total current assets      $61,555     $59,600\n"
        "Operating Loss            ($1,234)    ($2,100)\n"
        "Decline percentage        -4.5%       -6.2%"
    )
    units = split_into_sentence_units(table_text)
    # Each row should be preserved
    assert any("Cash and equivalents" in u for u in units)
    assert any("($1,234)" in u for u in units)
    assert any("-4.5%" in u for u in units)


def test_page_provenance_isolation():
    """Verify chunks do not span pages (page-scoped chunking contract)."""
    p1 = make_page(page_number=1, text="Page one disclosures and revenue figures.")
    p2 = make_page(page_number=2, text="Page two risk factors and guidance.")

    chunks = chunk_pages([p1, p2])
    assert len(chunks) == 2
    assert chunks[0].page_numbers == [1]
    assert chunks[1].page_numbers == [2]
    assert chunks[0].chunk_id == "doc_test123_p1_c0"
    assert chunks[1].chunk_id == "doc_test123_p2_c0"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_deterministic_chunk_ids():
    """Verify identical input produces identical deterministic chunk IDs."""
    p = make_page(page_number=4, text="Determinism verification text for financial chunking.")
    chunks_1 = chunk_pages([p])
    chunks_2 = chunk_pages([p])

    assert chunks_1[0].chunk_id == chunks_2[0].chunk_id
    assert chunks_1[0].chunk_id == "doc_test123_p4_c0"
