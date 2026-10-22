"""Comprehensive unit tests for PDF ingestion and page provenance extraction."""

import os
from typing import List, Optional
import pytest
import pymupdf

from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.errors import DocumentValidationError, PDFParsingError
from financial_report_analyzer.pdf_loader import (
    load_pdf,
    normalize_financial_text,
    validate_pdf_bytes,
    generate_document_id,
)


def create_synthetic_pdf(
    pages_text: List[str],
    password: Optional[str] = None,
) -> bytes:
    """Helper to programmatically generate an in-memory PDF using PyMuPDF."""
    doc = pymupdf.open()
    font_path = "C:/Windows/Fonts/arial.ttf" if os.path.exists("C:/Windows/Fonts/arial.ttf") else None

    for text in pages_text:
        page = doc.new_page()
        if text:
            if font_path:
                page.insert_font(fontname="f_arial", fontfile=font_path, set_simple=False)
                page.insert_text((50, 72), text, fontname="f_arial")
            else:
                page.insert_text((50, 72), text)

    if password:
        pdf_bytes = doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=password,
            owner_pw="admin_owner_secret_99",
        )
    else:
        pdf_bytes = doc.tobytes()

    doc.close()
    return pdf_bytes


def test_valid_pdf_multi_page_and_1_based_numbering():
    """Verify loading multi-page PDF preserves accurate 1-based page numbering."""
    p1 = "Financial Performance Summary\nRevenue: $100,000,000 for FY2024.\nOperating Margin: 25.4%."
    p2 = "Disclosed Risk Factors\nSupply chain disruptions could negatively impact our operations."
    p3 = "Management Outlook\nWe forecast strong international growth in FY2025."

    pdf_bytes = create_synthetic_pdf([p1, p2, p3])
    result = load_pdf(pdf_bytes, "annual_report_2024.pdf")

    assert result.metadata.total_pages == 3
    assert result.metadata.filename == "annual_report_2024.pdf"
    assert len(result.pages) == 3
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert result.pages[2].page_number == 3

    assert "Revenue: $100,000,000" in result.pages[0].text
    assert "Supply chain" in result.pages[1].text
    assert "FY2025" in result.pages[2].text
    assert result.total_characters > 0
    assert result.pages_with_text == 3
    assert result.metadata.meaningful_pages == 3


def test_extracted_text_statistics():
    """Verify statistics: total extracted characters and meaningful pages."""
    page_text = "Standard disclosures paragraph with sufficient density to qualify as meaningful text content."
    pdf_bytes = create_synthetic_pdf([page_text, page_text])
    result = load_pdf(pdf_bytes, "report.pdf", min_page_chars=30)

    assert result.metadata.total_pages == 2
    assert result.pages_with_text == 2
    assert result.metadata.meaningful_pages == 2
    assert result.total_characters == len(result.pages[0].text) + len(result.pages[1].text)


def test_empty_input_bytes():
    """Verify rejection of empty 0-byte input."""
    with pytest.raises(DocumentValidationError) as exc_info:
        load_pdf(b"", "empty_report.pdf")
    assert "0 bytes" in str(exc_info.value)


def test_invalid_bytes_with_pdf_extension():
    """Verify rejection of random non-PDF bytes disguised with .pdf extension."""
    fake_bytes = b"This is plain text pretending to be a PDF file."
    with pytest.raises(DocumentValidationError) as exc_info:
        load_pdf(fake_bytes, "fraudulent.pdf")
    assert "%PDF-" in str(exc_info.value)


def test_unsupported_file_extension():
    """Verify rejection of non-PDF file extensions."""
    valid_pdf = create_synthetic_pdf(["Sample text"])
    with pytest.raises(DocumentValidationError) as exc_info:
        load_pdf(valid_pdf, "statement.xlsx")
    assert "Invalid file extension" in str(exc_info.value)

    with pytest.raises(DocumentValidationError):
        load_pdf(valid_pdf, "statement.txt")


def test_oversized_upload_limit():
    """Verify enforcement of upload size limit."""
    small_pdf = create_synthetic_pdf(["Sample text"])
    # Create config with 1 KB upload limit
    strict_config = AppConfig(
        gemini_api_key="test",
        gemini_generation_model="gemini-2.5-flash",
        gemini_embedding_model="gemini-embedding-2",
        max_upload_mb=0,  # 0 MB implies 0 bytes
        max_pdf_pages=50,
        chunk_size_tokens=850,
        chunk_overlap_tokens=120,
        retrieval_top_k=5,
        min_evidence_threshold=0.35,
        log_level="INFO",
    )
    with pytest.raises(DocumentValidationError) as exc_info:
        load_pdf(small_pdf, "large.pdf", config=strict_config)
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_page_count_limit_exceeded():
    """Verify enforcement of maximum page count limit."""
    # 5-page PDF
    pdf_bytes = create_synthetic_pdf(["P1", "P2", "P3", "P4", "P5"])
    strict_config = AppConfig(
        gemini_api_key="test",
        gemini_generation_model="gemini-2.5-flash",
        gemini_embedding_model="gemini-embedding-2",
        max_upload_mb=25,
        max_pdf_pages=3,  # Limit to 3 pages
        chunk_size_tokens=850,
        chunk_overlap_tokens=120,
        retrieval_top_k=5,
        min_evidence_threshold=0.35,
        log_level="INFO",
    )
    with pytest.raises(DocumentValidationError) as exc_info:
        load_pdf(pdf_bytes, "exceeds_pages.pdf", config=strict_config)
    assert "exceeds the limit of 3 pages" in str(exc_info.value)


def test_malformed_corrupt_pdf():
    """Verify rejection of corrupt PDF bytes starting with %PDF- but having invalid syntax."""
    corrupted = b"%PDF-1.4\n%corrupt bytes that do not contain valid xref table or objects\x00\xff\xfe"
    with pytest.raises((PDFParsingError, DocumentValidationError)):
        load_pdf(corrupted, "corrupted.pdf")


def test_valid_pdf_with_blank_page():
    """Verify handling of a report that contains a blank page."""
    pdf_bytes = create_synthetic_pdf(["Cover page with text", "", "Page 3 with financial data"])
    result = load_pdf(pdf_bytes, "with_blank.pdf")

    assert result.metadata.total_pages == 3
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert result.pages[1].char_count == 0
    assert result.pages[1].has_meaningful_text is False
    assert result.pages[2].page_number == 3

    # Check that a warning was recorded specifically for page 2
    blank_warnings = [w for w in result.metadata.warnings if w.page_number == 2]
    assert len(blank_warnings) == 1
    assert blank_warnings[0].warning_type == "blank_or_unextractable_page"


def test_readable_pages_with_sparse_page():
    """Verify handling of sparse page warning without failing overall ingestion."""
    dense_text = (
        "Operating Results for Consolidated Statements of Income. "
        "Total revenue increased by 7.8% year-over-year driven by cloud growth. "
        "Operating expenses remained controlled across sales and research divisions."
    )
    sparse_text = "Chart 1"  # Very short text

    pdf_bytes = create_synthetic_pdf([dense_text, sparse_text, dense_text])
    result = load_pdf(pdf_bytes, "report.pdf", sparse_page_chars=50)

    assert result.metadata.total_pages == 3
    assert result.pages[1].page_number == 2
    assert result.pages[1].char_count == len("Chart 1")

    # Sparse page warning should be attached to page 2
    sparse_warnings = [w for w in result.metadata.warnings if w.page_number == 2]
    assert len(sparse_warnings) == 1
    assert sparse_warnings[0].warning_type == "sparse_page"


def test_pdf_with_no_extractable_text():
    """Verify behavior on image-only or completely unextractable PDF."""
    pdf_bytes = create_synthetic_pdf(["", "", ""])
    result = load_pdf(pdf_bytes, "scanned_doc.pdf")

    assert result.metadata.total_pages == 3
    assert result.pages_with_text == 0
    assert result.total_characters == 0
    assert result.is_sparse is True

    # Document-level warning about scanned / OCR requirement
    no_text_warnings = [w for w in result.metadata.warnings if w.warning_type == "no_extractable_text"]
    assert len(no_text_warnings) == 1
    assert "OCR" in no_text_warnings[0].message

    # Also verify require_extractable_text raises DocumentValidationError
    with pytest.raises(DocumentValidationError) as exc_info:
        load_pdf(pdf_bytes, "scanned_doc.pdf", require_extractable_text=True)
    assert "no extractable text" in str(exc_info.value)


def test_unicode_and_financial_values_preservation():
    """Verify exact preservation of currency symbols, parentheses, minus signs, and decimals."""
    financial_content = (
        "Consolidated Balance Sheets:\n"
        "Revenue: $383,285.50\n"
        "Net Income: €95,000.25\n"
        "Tokyo Segment: ¥1,250,000\n"
        "UK Division: £45,200.10\n"
        "India Division: ₹85,000.00\n"
        "Operating Loss: ($12,450.75)\n"
        "Margin Decline: -4.5%\n"
        "En-dash range: $100M – $120M\n"
    )

    pdf_bytes = create_synthetic_pdf([financial_content])
    result = load_pdf(pdf_bytes, "financials.pdf")

    extracted = result.pages[0].text
    assert "$383,285.50" in extracted
    assert "€95,000.25" in extracted
    assert "¥1,250,000" in extracted
    assert "£45,200.10" in extracted
    assert "₹85,000.00" in extracted
    assert "($12,450.75)" in extracted
    assert "-4.5%" in extracted


def test_password_protected_pdf_rejection():
    """Verify rejection of encrypted / password-protected PDF."""
    pdf_bytes = create_synthetic_pdf(["Classified financial forecast"], password="secret_pass_123")
    with pytest.raises(DocumentValidationError) as exc_info:
        load_pdf(pdf_bytes, "encrypted_report.pdf")
    assert "password-protected" in str(exc_info.value).lower()


def test_normalize_financial_text_preserves_tables_and_spaces():
    """Verify normalize_financial_text strips excessive blanks without mangling columns."""
    raw = (
        "Item                        2024        2023\n"
        "Net Sales                 $400,000    $380,000\n"
        "Cost of Sales             ($240,000)  ($230,000)\n"
        "\n\n\n\n"
        "Notes to Statements:\n"
        "Depreciation:  -2.1%\n"
    )
    cleaned = normalize_financial_text(raw)
    assert "($240,000)" in cleaned
    assert "-2.1%" in cleaned
    assert "\n\n\n" not in cleaned  # Quadruple newlines collapsed to double
    assert "Net Sales" in cleaned


def test_generate_document_id():
    """Verify deterministic and safe document ID generation."""
    b1 = b"%PDF-1.4 sample content 1"
    b2 = b"%PDF-1.4 sample content 2"
    id1_a = generate_document_id(b1)
    id1_b = generate_document_id(b1)
    id2 = generate_document_id(b2)

    assert id1_a == id1_b
    assert id1_a != id2
    assert id1_a.startswith("doc_")
    assert len(id1_a) == 16  # "doc_" (4) + 12 hex chars
