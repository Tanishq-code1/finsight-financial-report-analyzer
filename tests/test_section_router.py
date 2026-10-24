"""Unit tests for section-aware page routing."""

import pytest
from financial_report_analyzer.schemas import PageRecord
from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.section_router import (
    route_pages_for_extraction,
    detect_report_sections,
    partition_page_sequence,
)


def make_page(page_number: int, text: str, doc_id: str = "doc_test") -> PageRecord:
    return PageRecord(
        document_id=doc_id,
        source_filename="test.pdf",
        page_number=page_number,
        text=text,
        char_count=len(text),
        has_meaningful_text=len(text.strip()) > 0,
    )


def test_small_document_routes_all_pages():
    """Verify documents with <= 10 pages do not filter out any pages."""
    pages = [
        make_page(1, "Executive Summary: Strong financial performance in FY2024."),
        make_page(2, "Income Statement: Revenue $1,250M, Net Income $185M."),
        make_page(3, "Balance Sheet: Total Assets $2,150M."),
        make_page(4, "Risk Factors: Cybersecurity threats and operational risks."),
    ]
    cfg = AppConfig(enable_section_routing=True)
    batches = route_pages_for_extraction(pages, config=cfg)

    assert len(batches) == 1
    assert len(batches[0]) == 4
    assert [p.page_number for p in batches[0]] == [1, 2, 3, 4]


def test_10k_section_detection_and_routing():
    """Verify realistic 10-K document structure routes statements, MD&A, and risks."""
    pages = []
    # Overview (pages 1-5)
    for p in range(1, 6):
        pages.append(make_page(p, f"Page {p} Business Overview and Table of Contents"))

    # Item 1A Risk Factors (pages 11-15)
    pages.append(make_page(11, "ITEM 1A. RISK FACTORS\nOur business faces competition and cyber risks."))
    for p in range(12, 16):
        pages.append(make_page(p, f"Page {p} More disclosed risk factors and supply chain delays."))

    # Routine disclosures (pages 16-25, non-material)
    for p in range(16, 26):
        pages.append(make_page(p, f"Page {p} Mine safety disclosures and properties."))

    # Item 7 MD&A (pages 26-30)
    pages.append(make_page(26, "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION\nRevenues increased 15%."))
    for p in range(27, 31):
        pages.append(make_page(p, f"Page {p} MD&A operational trends, segment guidance, and outlook."))

    # Item 8 Financial Statements (pages 35-40)
    pages.append(make_page(35, "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA\nConsolidated Statements of Income"))
    pages.append(make_page(36, "Consolidated Statements of Income\nTotal revenues were $350,000 million."))
    pages.append(make_page(37, "Consolidated Balance Sheets\nTotal assets were $450,000 million."))
    pages.append(make_page(38, "Consolidated Statements of Cash Flows\nOperating cash flow was $125,000 million."))
    pages.append(make_page(39, "Note 1 - Summary of Significant Accounting Policies."))
    pages.append(make_page(40, "Note 2 - Cash, Cash Equivalents, and Marketable Securities."))

    # Item 9 and Signatures / Exhibits (pages 41-50)
    pages.append(make_page(41, "ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING"))
    for p in range(42, 51):
        pages.append(make_page(p, f"Page {p} Exhibit index and executive signature blocks."))

    overview, stmts, mda, risks = detect_report_sections(pages)
    assert 1 in overview and 5 in overview
    assert 11 in risks
    assert 26 in mda
    assert 35 in stmts and 36 in stmts and 37 in stmts and 38 in stmts

    cfg = AppConfig(enable_section_routing=True, extraction_batch_page_size=10)
    routed_batches = route_pages_for_extraction(pages, config=cfg)

    routed_pnums = set()
    for b in routed_batches:
        for page in b:
            routed_pnums.add(page.page_number)

    # Core statements MUST be included
    assert 35 in routed_pnums
    assert 36 in routed_pnums
    assert 37 in routed_pnums
    assert 38 in routed_pnums

    # MD&A and risks included
    assert 26 in routed_pnums
    assert 11 in routed_pnums

    # Routine boilerplate non-material pages excluded from extraction
    assert 45 not in routed_pnums
    assert 48 not in routed_pnums


def test_fallback_when_routing_disabled():
    """Verify disabling section routing returns all pages partitioned sequentially."""
    pages = [make_page(i, f"Page {i} content text") for i in range(1, 21)]
    cfg = AppConfig(enable_section_routing=False, extraction_batch_page_size=5)
    batches = route_pages_for_extraction(pages, config=cfg)

    assert len(batches) == 4
    total_routed = sum(len(b) for b in batches)
    assert total_routed == 20
