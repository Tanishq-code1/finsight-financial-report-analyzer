"""Orchestration pipeline for full-report financial extraction, citation verification, and deterministic analysis."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Callable, Sequence, Dict, Tuple

from financial_report_analyzer.config import AppConfig, load_config
from financial_report_analyzer.errors import FinancialAnalyzerError, ConfigurationError
from financial_report_analyzer.schemas import (
    PageRecord,
    DocumentMetadata,
    FinancialMetric,
    RiskItem,
    OutlookItem,
    ReportAnalysis,
    BatchExtractionResult,
)
from financial_report_analyzer.financial_extractor import FinancialExtractor
from financial_report_analyzer.financial_calculations import (
    calculate_yoy_comparison,
    calculate_free_cash_flow,
    deduplicate_and_resolve_metrics,
    parse_financial_number,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_PAGE_SIZE = 12
DEFAULT_MAX_BATCH_CHARS = 30000


def partition_pages_into_batches(
    pages: Sequence[PageRecord],
    max_pages: int = DEFAULT_BATCH_PAGE_SIZE,
    max_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> List[List[PageRecord]]:
    """Partition readable pages into bounded, page-aware batches based on page count and character budget."""
    if not pages:
        return []

    batches: List[List[PageRecord]] = []
    current_batch: List[PageRecord] = []
    current_chars = 0

    for page in pages:
        page_len = len(page.text)
        # If adding this page exceeds either page cap or character budget (unless batch is empty)
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


def synthesize_executive_summary(
    company_name: Optional[str],
    fiscal_period: Optional[str],
    metrics: List[FinancialMetric],
    risks: List[RiskItem],
    outlook: List[OutlookItem],
    summary_snippets: List[str],
) -> Tuple[str, List[str]]:
    """Synthesize a concise, factual executive summary and key strategic takeaways from extracted findings."""
    company = company_name or "The company"
    period = fiscal_period or "the latest reporting period"

    # Find key figures
    rev = next((m for m in metrics if m.normalized_name == "revenue" or "revenue" in m.name.lower()), None)
    ni = next((m for m in metrics if m.normalized_name == "net_income" or "net income" in m.name.lower()), None)
    ocf = next((m for m in metrics if m.normalized_name == "operating_cash_flow" or "cash flow" in m.name.lower()), None)

    summary_parts = []
    summary_parts.append(f"Financial Report Analysis for {company} covering {period}.")

    if rev and rev.value is not None:
        unit_str = f" {rev.unit_scale}" if rev.unit_scale else ""
        curr_str = f"{rev.currency} " if rev.currency else ""
        yoy_str = f" ({rev.yoy_calculation_note})" if rev.yoy_calculation_note and "YoY" in rev.yoy_calculation_note else ""
        summary_parts.append(f"Reported total revenue was {curr_str}{rev.value}{unit_str}{yoy_str}.")

    if ni and ni.value is not None:
        unit_str = f" {ni.unit_scale}" if ni.unit_scale else ""
        curr_str = f"{ni.currency} " if ni.currency else ""
        summary_parts.append(f"Net income was {curr_str}{ni.value}{unit_str}.")

    if ocf and ocf.value is not None:
        unit_str = f" {ocf.unit_scale}" if ocf.unit_scale else ""
        curr_str = f"{ocf.currency} " if ocf.currency else ""
        summary_parts.append(f"Operating cash flow was {curr_str}{ocf.value}{unit_str}.")

    if summary_snippets:
        summary_parts.append(" ".join(summary_snippets[:3]))

    executive_summary = " ".join(summary_parts)

    takeaways = []
    if rev and rev.value is not None:
        takeaways.append(f"Revenue reached {rev.value} {rev.unit_scale or ''} for {rev.period_label or period}.")
    if ni and ni.value is not None:
        takeaways.append(f"Net income stood at {ni.value} {ni.unit_scale or ''}.")
    if risks:
        takeaways.append(f"Identified {len(risks)} material disclosed risk factors, led by: {risks[0].risk_title}.")
    if outlook:
        takeaways.append(f"Management outlook notes: {outlook[0].statement[:100]}...")

    if not takeaways:
        takeaways.append("Detailed performance metrics and disclosures extracted into structured tables.")

    return executive_summary, takeaways


def deduplicate_risks(risks: List[RiskItem]) -> List[RiskItem]:
    """Deduplicate risk factors by normalized title while merging unique supporting pages and quotes."""
    seen: Dict[str, RiskItem] = {}
    for r in risks:
        key = r.risk_title.lower().strip()
        if key not in seen:
            seen[key] = r
        else:
            existing = seen[key]
            combined_pages = list(set(existing.source_pages + r.source_pages))
            combined_quotes = list(set(existing.source_quotes + r.source_quotes))
            seen[key] = existing.model_copy(
                update={
                    "source_pages": combined_pages,
                    "source_quotes": combined_quotes,
                    "company_stated_mitigation": existing.company_stated_mitigation or r.company_stated_mitigation,
                }
            )
    return list(seen.values())


def deduplicate_outlook(outlook: List[OutlookItem]) -> List[OutlookItem]:
    """Deduplicate forward-looking statements by statement excerpt."""
    seen: Dict[str, OutlookItem] = {}
    for o in outlook:
        key = o.statement[:80].lower().strip()
        if key not in seen:
            seen[key] = o
        else:
            existing = seen[key]
            combined_pages = list(set(existing.source_pages + o.source_pages))
            combined_quotes = list(set(existing.source_quotes + o.source_quotes))
            seen[key] = existing.model_copy(
                update={
                    "source_pages": combined_pages,
                    "source_quotes": combined_quotes,
                    "period": existing.period or o.period,
                }
            )
    return list(seen.values())


def run_financial_analysis_pipeline(
    pages: Sequence[PageRecord],
    metadata: DocumentMetadata,
    config: Optional[AppConfig] = None,
    extractor: Optional[FinancialExtractor] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    batch_page_size: int = DEFAULT_BATCH_PAGE_SIZE,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> ReportAnalysis:
    """Orchestrate end-to-end full document financial analysis across bounded batches.

    Returns:
        Validated ReportAnalysis container.
    """
    app_config = config or load_config()

    if progress_callback:
        progress_callback(0.05, "Inspecting document pages for readable text...")

    # 1. Filter readable pages vs sparse/empty pages
    readable_pages = [p for p in pages if p.has_meaningful_text]
    total_pages_count = len(pages)
    readable_pages_count = len(readable_pages)
    skipped_pages_count = total_pages_count - readable_pages_count

    limitations: List[str] = []
    if skipped_pages_count > 0:
        limitations.append(
            f"{skipped_pages_count} page(s) contained sparse or unselectable text (OCR required) and were excluded from extraction."
        )

    if not readable_pages:
        logger.warning("No readable pages found in document %s", metadata.document_id)
        return ReportAnalysis(
            report_title=metadata.filename,
            company_name=None,
            report_type="Financial Report",
            fiscal_period=None,
            executive_summary="The uploaded document does not contain selectable text pages suitable for financial analysis.",
            key_takeaways=[],
            metrics=[],
            risks=[],
            outlook=[],
            limitations=limitations + ["Document text extraction produced zero readable pages."],
            generated_at=datetime.now(timezone.utc).isoformat(),
            analysis_status="failed",
            pages_analyzed=0,
            total_pages=total_pages_count,
            batches_succeeded=0,
            batches_failed=0,
            data_conflicts=[],
            verification_rate=0.0,
        )

    # 2. Partition into bounded batches
    batches = partition_pages_into_batches(readable_pages, max_pages=batch_page_size, max_chars=max_batch_chars)
    total_batches = len(batches)
    logger.info("Partitioned %d readable pages into %d extraction batches", readable_pages_count, total_batches)

    # 3. Initialize or use supplied extractor
    ext = extractor or FinancialExtractor(config=app_config)

    raw_metrics: List[FinancialMetric] = []
    raw_risks: List[RiskItem] = []
    raw_outlook: List[OutlookItem] = []
    summary_snippets: List[str] = []
    detected_company: Optional[str] = None
    detected_period: Optional[str] = None
    detected_title: Optional[str] = None

    batches_succeeded = 0
    batches_failed = 0

    for idx, batch in enumerate(batches, start=1):
        page_range_label = f"pages {batch[0].page_number}–{batch[-1].page_number}"
        progress_fraction = 0.10 + (0.75 * ((idx - 1) / total_batches))

        if progress_callback:
            progress_callback(
                progress_fraction,
                f"Analyzing batch {idx}/{total_batches} ({page_range_label})...",
            )

        try:
            batch_result = ext.extract_batch(batch, document_id=metadata.document_id)
            batches_succeeded += 1

            if batch_result.company_name and not detected_company:
                detected_company = batch_result.company_name
            if batch_result.fiscal_period and not detected_period:
                detected_period = batch_result.fiscal_period
            if batch_result.report_title and not detected_title:
                detected_title = batch_result.report_title

            raw_metrics.extend(batch_result.metrics)
            raw_risks.extend(batch_result.risks)
            raw_outlook.extend(batch_result.outlook)
            summary_snippets.extend(batch_result.executive_summary_excerpts)

        except Exception as exc:
            batches_failed += 1
            err_msg = f"Extraction batch {idx} ({page_range_label}) failed: {str(exc)}"
            logger.error(err_msg)
            limitations.append(err_msg)

    if progress_callback:
        progress_callback(0.88, "Deduplicating metrics and computing financial YoY changes...")

    # 4. Deduplicate and detect conflicts in metrics
    resolved_metrics, metric_conflicts = deduplicate_and_resolve_metrics(raw_metrics)

    # 5. Deduplicate risks and outlook
    resolved_risks = deduplicate_risks(raw_risks)
    resolved_outlook = deduplicate_outlook(raw_outlook)

    # 6. Deterministic Python YoY Calculations
    calculated_metrics: List[FinancialMetric] = []
    for m in resolved_metrics:
        # Normalize name for clean key lookup
        norm_key = m.normalized_name or m.name.lower().replace(" ", "_")
        parsed_curr = parse_financial_number(m.value)

        # If metric has reported prior_period_value, calculate YoY in Python
        if m.prior_period_value is not None and m.value is not None:
            yoy_res = calculate_yoy_comparison(
                current_val=m.value,
                prior_val=m.prior_period_value,
                current_currency=m.currency,
                prior_currency=m.currency,
                current_scale=m.unit_scale,
                prior_scale=m.unit_scale,
                current_period=m.period_label,
                prior_period=m.prior_period_label,
            )
            updated_m = m.model_copy(
                update={
                    "normalized_name": norm_key,
                    "numeric_value": float(parsed_curr) if parsed_curr is not None else None,
                    "yoy_absolute_change": float(yoy_res.absolute_change) if yoy_res.absolute_change is not None else None,
                    "yoy_percentage_change": float(yoy_res.percentage_change) if yoy_res.percentage_change is not None else None,
                    "yoy_calculation_note": yoy_res.note,
                }
            )
            calculated_metrics.append(updated_m)
        else:
            updated_m = m.model_copy(
                update={
                    "normalized_name": norm_key,
                    "numeric_value": float(parsed_curr) if parsed_curr is not None else None,
                }
            )
            calculated_metrics.append(updated_m)

    # 7. Deterministically calculate Free Cash Flow if OCF and Capex exist
    ocf = next((m for m in calculated_metrics if m.normalized_name in ("operating_cash_flow", "cash_from_operations") or "operating cash" in m.name.lower()), None)
    capex = next((m for m in calculated_metrics if m.normalized_name in ("capital_expenditures", "capex") or "capital expenditures" in m.name.lower()), None)

    if ocf and capex:
        fcf = calculate_free_cash_flow(ocf, capex)
        if fcf:
            calculated_metrics.append(fcf)

    # 8. Compute citation verification rate
    total_items = len(calculated_metrics) + len(resolved_risks) + len(resolved_outlook)
    verified_items = (
        sum(1 for m in calculated_metrics if m.is_verified)
        + sum(1 for r in resolved_risks if r.is_verified)
        + sum(1 for o in resolved_outlook if o.is_verified)
    )
    verification_rate = (verified_items / total_items) if total_items > 0 else 1.0

    # 9. Synthesize Executive Summary
    summary_text, takeaways = synthesize_executive_summary(
        company_name=detected_company,
        fiscal_period=detected_period,
        metrics=calculated_metrics,
        risks=resolved_risks,
        outlook=resolved_outlook,
        summary_snippets=summary_snippets,
    )

    # 10. Determine overall status
    if batches_failed == 0 and batches_succeeded > 0:
        analysis_status = "complete"
    elif batches_succeeded > 0 and batches_failed > 0:
        analysis_status = "partial"
    else:
        analysis_status = "failed"

    if progress_callback:
        progress_callback(1.0, "Analysis complete.")

    return ReportAnalysis(
        report_title=detected_title or metadata.filename,
        company_name=detected_company,
        report_type="10-K / Annual Report",
        fiscal_period=detected_period,
        executive_summary=summary_text,
        key_takeaways=takeaways,
        metrics=calculated_metrics,
        risks=resolved_risks,
        outlook=resolved_outlook,
        limitations=limitations,
        generated_at=datetime.now(timezone.utc).isoformat(),
        analysis_status=analysis_status,
        pages_analyzed=readable_pages_count,
        total_pages=total_pages_count,
        batches_succeeded=batches_succeeded,
        batches_failed=batches_failed,
        data_conflicts=metric_conflicts,
        verification_rate=round(verification_rate, 4),
    )
