"""Section-aware page router for targeted financial extraction in 10-K, 10-Q, and annual reports.

Enables targeted batch extraction for long financial filings while preserving 100% ChromaDB
indexing for comprehensive semantic search and citation auditability.
"""

import re
import logging
from typing import List, Sequence, Optional, Set, Tuple
from financial_report_analyzer.schemas import PageRecord
from financial_report_analyzer.config import AppConfig

logger = logging.getLogger(__name__)

# Core section header patterns (SEC Form 10-K and 10-Q standard numbering)
ITEM_1A_RE = re.compile(r"(?i)\bitem\s+1a[\.:\s\-]+risk\s+factors\b")
ITEM_7_RE = re.compile(r"(?i)\bitem\s+7[\.:\s\-]+management.*discussion", re.DOTALL)
ITEM_7A_RE = re.compile(r"(?i)\bitem\s+7a[\.:\s\-]+quantitative", re.DOTALL)
ITEM_8_RE = re.compile(r"(?i)\bitem\s+8[\.:\s\-]+financial\s+statements", re.DOTALL)
ITEM_9_RE = re.compile(r"(?i)\bitem\s+9[\.:\s\-]+changes", re.DOTALL)

# Core statement headline patterns
STMT_INCOME_RE = re.compile(r"(?i)consolidated\s+statements\s+of\s+(?:income|operations|earnings)")
STMT_BALANCE_RE = re.compile(r"(?i)consolidated\s+balance\s+sheets")
STMT_CASH_RE = re.compile(r"(?i)consolidated\s+statements\s+of\s+cash\s+flows")

# Maximum pages and character caps per batch
DEFAULT_MAX_BATCH_PAGES = 16
DEFAULT_MAX_BATCH_CHARS = 45000


def partition_page_sequence(
    pages: Sequence[PageRecord],
    max_pages: int = DEFAULT_MAX_BATCH_PAGES,
    max_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> List[List[PageRecord]]:
    """Partition a list of PageRecords into bounded batches based on page count and character limits."""
    if not pages:
        return []

    batches: List[List[PageRecord]] = []
    current_batch: List[PageRecord] = []
    current_chars = 0

    for page in pages:
        page_len = len(page.text)
        if current_batch and (len(current_batch) >= max_pages or (current_chars + page_len > max_chars)):
            batches.append(current_batch)
            current_batch = [page]
            current_chars = page_len
        else:
            current_batch.append(page)
            current_chars += page_len

    if current_batch:
        batches.append(current_batch)

    return batches


def detect_report_sections(pages: Sequence[PageRecord]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Identify page numbers belonging to key filing sections.

    Returns:
        Tuple of (overview_pages, statement_pages, mda_pages, risk_pages)
    """
    total_pages = len(pages)
    page_by_num = {p.page_number: p for p in pages}
    sorted_pnums = sorted(page_by_num.keys())

    # 1. Overview pages (First 1-5 pages)
    overview_pnums = [p for p in sorted_pnums if p <= min(5, total_pages)]

    # Scan for section marker page numbers (exclude table of contents in first 8 pages)
    risk_start = None
    mda_start = None
    mda_end = None
    stmt_start = None
    stmt_end = None

    for p in sorted_pnums:
        if p <= 8:
            continue

        header_sample = page_by_num[p].text[:400]

        if risk_start is None and ITEM_1A_RE.search(header_sample):
            risk_start = p

        if mda_start is None and ITEM_7_RE.search(header_sample):
            mda_start = p

        if mda_end is None and (ITEM_7A_RE.search(header_sample) or ITEM_8_RE.search(header_sample)):
            if mda_start is not None and p > mda_start:
                mda_end = p

        if stmt_start is None and (ITEM_8_RE.search(header_sample) or STMT_INCOME_RE.search(header_sample) or STMT_BALANCE_RE.search(header_sample)):
            stmt_start = p

        if stmt_end is None and ITEM_9_RE.search(header_sample):
            if stmt_start is not None and p > stmt_start:
                stmt_end = p

    # Build statement pages (Statements are highest priority for financial metrics)
    statement_pnums: List[int] = []
    if stmt_start is not None:
        # Target primary statements + first notes (typically ~18 pages)
        end_p = min(stmt_end or (stmt_start + 18), stmt_start + 18, max(sorted_pnums))
        statement_pnums = [p for p in sorted_pnums if stmt_start <= p <= end_p]
    else:
        # Check individual statement occurrences
        for p in sorted_pnums:
            txt = page_by_num[p].text
            if STMT_INCOME_RE.search(txt) or STMT_BALANCE_RE.search(txt) or STMT_CASH_RE.search(txt):
                if p not in statement_pnums:
                    statement_pnums.append(p)
                    if p + 1 in page_by_num and (p + 1) not in statement_pnums:
                        statement_pnums.append(p + 1)
        statement_pnums.sort()

    # Build MD&A pages (Forward outlook, business context, capex discussion)
    mda_pnums: List[int] = []
    if mda_start is not None:
        end_p = min(mda_end or (mda_start + 16), mda_start + 16, max(sorted_pnums))
        mda_pnums = [p for p in sorted_pnums if mda_start <= p <= end_p]

    # Build Risk Factor pages (Disclosed risks and mitigations)
    risk_pnums: List[int] = []
    if risk_start is not None:
        end_p = min(mda_start or (risk_start + 14), risk_start + 14, max(sorted_pnums))
        risk_pnums = [p for p in sorted_pnums if risk_start <= p <= end_p]

    return overview_pnums, statement_pnums, mda_pnums, risk_pnums


def route_pages_for_extraction(
    pages: Sequence[PageRecord],
    config: Optional[AppConfig] = None,
) -> List[List[PageRecord]]:
    """Route report pages into semantically targeted batches for financial extraction.

    For documents <= 10 pages or when section routing is disabled:
        Partitions all readable pages into consecutive batches.
    For documents > 10 pages (e.g. 100-page 10-K filings):
        Identifies Executive Overview, Financial Statements (Item 8), MD&A & Outlook (Item 7),
        and Risk Factors (Item 1A), creating prioritized, bounded batches.

    Guarantees:
        - 1-based page numbers are perfectly preserved.
        - Core financial statements are ALWAYS prioritized.
        - If section routing yields no pages, falls back safely to all pages.
    """
    if not pages:
        return []

    batch_page_size = getattr(config, "extraction_batch_page_size", DEFAULT_MAX_BATCH_PAGES)
    max_batch_chars = getattr(config, "extraction_max_batch_chars", DEFAULT_MAX_BATCH_CHARS)
    enable_routing = getattr(config, "enable_section_routing", True)

    total_pages = len(pages)

    # Small documents (e.g., 4-page synthetic reports) do not need section routing
    if total_pages <= 10 or not enable_routing:
        logger.info("Using standard sequential partitioning for %d pages (routing not required)", total_pages)
        return partition_page_sequence(pages, max_pages=batch_page_size, max_chars=max_batch_chars)

    # Large filings: detect sections
    overview_pnums, statement_pnums, mda_pnums, risk_pnums = detect_report_sections(pages)
    page_by_num = {p.page_number: p for p in pages}

    # If neither statements nor MD&A were identified, fall back safely
    if not statement_pnums and not mda_pnums:
        logger.info("No SEC section markers detected in %d pages; falling back to sequential partitioning", total_pages)
        return partition_page_sequence(pages, max_pages=batch_page_size, max_chars=max_batch_chars)

    routed_batches: List[List[PageRecord]] = []
    seen_page_numbers: Set[int] = set()

    def add_section_batches(pnums: List[int], section_name: str):
        selected_pages = [page_by_num[p] for p in pnums if p in page_by_num and p not in seen_page_numbers]
        if not selected_pages:
            return
        for p in selected_pages:
            seen_page_numbers.add(p.page_number)
        sec_batches = partition_page_sequence(selected_pages, max_pages=batch_page_size, max_chars=max_batch_chars)
        logger.info("Section '%s': %d pages partitioned into %d batches", section_name, len(selected_pages), len(sec_batches))
        routed_batches.extend(sec_batches)

    # 1. Executive Overview (pages 1-5): Sets company name, period, and high-level context
    add_section_batches(overview_pnums, "Overview")

    # 2. Financial Statements (Item 8): HIGHEST PRIORITY for metrics (Revenue, Net Income, Cash Flows, Balance Sheet)
    add_section_batches(statement_pnums, "Financial Statements")

    # 3. MD&A & Management Outlook (Item 7): Forward-looking statements, Capex guidance, segment discussion
    add_section_batches(mda_pnums, "MD&A and Outlook")

    # 4. Risk Factors (Item 1A): Disclosed business, regulatory, and cyber risks
    add_section_batches(risk_pnums, "Risk Factors")

    if not routed_batches:
        logger.warning("Section routing produced 0 batches; falling back to sequential partitioning")
        return partition_page_sequence(pages, max_pages=batch_page_size, max_chars=max_batch_chars)

    total_routed_pages = sum(len(b) for b in routed_batches)
    logger.info(
        "Section routing complete: selected %d of %d pages into %d batches (saving %d non-material pages from LLM extraction)",
        total_routed_pages,
        total_pages,
        len(routed_batches),
        total_pages - total_routed_pages,
    )
    return routed_batches
