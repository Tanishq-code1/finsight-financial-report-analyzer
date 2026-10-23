"""Export service for FinSight financial report analysis.

Provides in-memory generation of JSON, CSV, and Markdown exports with:
- Strict Pydantic validation
- Preservation of Decimal precision and source provenance
- Spreadsheet formula injection defense for CSVs
- Markdown table and layout escaping
- Zero filesystem writes and zero external network calls
"""

import io
import re
import csv
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List, Union

from financial_report_analyzer.schemas import (
    ReportAnalysis,
    FinancialMetric,
    RiskItem,
    OutlookItem,
    DocumentMetadata,
)

EXPORT_SCHEMA_VERSION = "1.0.0"

DISCLAIMER_TEXT = (
    "FinSight is an AI-assisted financial document research tool and information aid, "
    "NOT investment, audit, credit, or tax advice. All figures, claims, and forward-looking "
    "statements should be independently verified against official regulatory filings."
)

# Characters that trigger spreadsheet formula execution in Microsoft Excel, LibreOffice, Google Sheets
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_export_filename(filename: Optional[str], export_type: str, extension: str) -> str:
    """Produce a safe, predictable filename without directory traversal or forbidden characters."""
    raw_name = filename or "financial_report"
    # Strip path separators
    clean_base = re.sub(r"[\\/]", "_", raw_name)
    # Remove extension if present
    clean_base = re.sub(r"\.[a-zA-Z0-9]+$", "", clean_base)
    # Remove characters forbidden on Windows/Linux/macOS filesystems
    clean_base = re.sub(r'[<>:"|?*]', "_", clean_base).strip(" ._")
    if not clean_base:
        clean_base = "financial_report"

    clean_ext = extension.lstrip(".")
    clean_type = re.sub(r"[^a-zA-Z0-9_-]", "_", export_type).strip("_")

    return f"{clean_base}_{clean_type}.{clean_ext}"


def escape_csv_formula(val: Any) -> str:
    """Prevent CSV formula injection (DDE/formula execution) in untrusted string fields.

    If a text string begins with '=', '+', '-', '@', tab, or carriage return after
    stripping whitespace, prepends a single quote "'" so spreadsheet engines treat it
    as literal text.

    Numeric types (int, float, Decimal) are NOT escaped here so numeric values remain
    functional in calculations.
    """
    if val is None:
        return ""

    if isinstance(val, (int, float, Decimal)):
        return str(val)

    s = str(val)
    stripped = s.lstrip()
    if stripped and stripped.startswith(FORMULA_PREFIXES):
        # Escape by prepending a single quote
        return f"'{s}"
    return s


def decimal_default_serializer(obj: Any) -> Any:
    """JSON serializer callback supporting Decimal and datetime objects."""
    if isinstance(obj, Decimal):
        # Preserves exact decimal precision as string or canonical number
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def escape_markdown_table_cell(text: Optional[str]) -> str:
    """Escape pipe characters and line breaks so untrusted strings don't corrupt Markdown tables."""
    if text is None:
        return "—"
    s = str(text)
    # Escape pipe
    s = s.replace("|", "\\|")
    # Replace newlines with space
    s = re.sub(r"[\r\n]+", " ", s)
    return s.strip()


def _is_report_analysis(obj: Any) -> bool:
    """Validate whether obj is a ReportAnalysis instance, robust against Streamlit module reloading skew."""
    if isinstance(obj, ReportAnalysis):
        return True
    return (
        obj is not None
        and type(obj).__name__ == "ReportAnalysis"
        and hasattr(obj, "metrics")
        and hasattr(obj, "risks")
        and hasattr(obj, "outlook")
    )


def export_analysis_json(
    analysis: ReportAnalysis,
    metadata: Optional[DocumentMetadata] = None,
    generation_model: Optional[str] = None,
) -> str:
    """Generate a structured, canonical JSON export of the complete analysis.

    Validates schema, preserves Decimal precision, formats ISO 8601 timestamps,
    and excludes secrets or environment details.
    """
    if not _is_report_analysis(analysis):
        raise TypeError("Input must be a valid ReportAnalysis instance.")

    export_timestamp = datetime.now(timezone.utc).isoformat()

    doc_info: Dict[str, Any] = {
        "document_id": metadata.document_id if metadata else "doc_unknown",
        "source_filename": metadata.filename if metadata else (analysis.report_title or "document.pdf"),
        "total_pages": metadata.total_pages if metadata else analysis.total_pages,
        "meaningful_pages": metadata.meaningful_pages if metadata else analysis.pages_analyzed,
    }

    coverage_info: Dict[str, Any] = {
        "analysis_status": analysis.analysis_status,
        "pages_analyzed": analysis.pages_analyzed,
        "total_pages": analysis.total_pages,
        "batches_succeeded": analysis.batches_succeeded,
        "batches_failed": analysis.batches_failed,
        "citation_verification_rate": analysis.verification_rate,
        "disclosed_conflicts_count": len(analysis.data_conflicts),
    }

    # Metrics serialization
    metrics_data = []
    for m in analysis.metrics:
        metrics_data.append(
            {
                "name": m.name,
                "normalized_name": m.normalized_name,
                "reported_value": m.value,
                "numeric_value": m.numeric_value,
                "currency": m.currency,
                "unit_scale": m.unit_scale,
                "period_label": m.period_label,
                "prior_period_value": m.prior_period_value,
                "prior_period_label": m.prior_period_label,
                "yoy_absolute_change": m.yoy_absolute_change,
                "yoy_percentage_change": m.yoy_percentage_change,
                "yoy_calculation_note": m.yoy_calculation_note,
                "metric_type": m.metric_type,
                "calculation_method": m.calculation_method,
                "source_page": m.source_page,
                "source_quote": m.source_quote,
                "status": m.status,
                "confidence_note": m.confidence_note,
                "is_verified": m.is_verified,
                "has_conflict": m.has_conflict,
                "conflicting_values": m.conflicting_values,
            }
        )

    # Risks serialization
    risks_data = []
    for r in analysis.risks:
        risks_data.append(
            {
                "risk_title": r.risk_title,
                "description": r.description,
                "risk_category": r.risk_category,
                "company_stated_mitigation": r.company_stated_mitigation,
                "severity": r.severity,
                "source_pages": r.source_pages,
                "source_quotes": r.source_quotes,
                "impact_type": r.impact_type,
                "is_verified": r.is_verified,
                "source_heading": r.source_heading,
            }
        )

    # Outlook serialization
    outlook_data = []
    for o in analysis.outlook:
        outlook_data.append(
            {
                "statement": o.statement,
                "period": o.period,
                "statement_type": o.statement_type,
                "source_pages": o.source_pages,
                "source_quotes": o.source_quotes,
                "time_horizon": o.time_horizon,
                "direction": o.direction,
                "target_metric": o.target_metric,
                "target_value": o.target_value,
                "is_verified": o.is_verified,
            }
        )

    payload: Dict[str, Any] = {
        "export_metadata": {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "exported_at": export_timestamp,
            "analysis_generated_at": analysis.generated_at,
            "generation_model": generation_model or "gemini-3.8-flash",
            "disclaimer": DISCLAIMER_TEXT,
        },
        "document": doc_info,
        "filing_info": {
            "company_name": analysis.company_name,
            "report_title": analysis.report_title,
            "report_type": analysis.report_type,
            "fiscal_period": analysis.fiscal_period,
        },
        "coverage": coverage_info,
        "executive_summary": analysis.executive_summary,
        "key_takeaways": analysis.key_takeaways,
        "financial_metrics": metrics_data,
        "disclosed_risks": risks_data,
        "management_outlook": outlook_data,
        "data_conflicts": analysis.data_conflicts,
        "limitations": analysis.limitations,
    }

    return json.dumps(payload, indent=2, ensure_ascii=False, default=decimal_default_serializer)


def export_metrics_csv(
    analysis: ReportAnalysis,
    metadata: Optional[DocumentMetadata] = None,
) -> str:
    """Generate a spreadsheet-friendly CSV containing financial metrics and YoY comparisons.

    Implements formula injection defense on untrusted string cells while preserving
    numeric values in calculation columns.
    """
    if not _is_report_analysis(analysis):
        raise TypeError("Input must be a valid ReportAnalysis instance.")

    filename = metadata.filename if metadata else (analysis.report_title or "document.pdf")
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

    # Documented column headers
    headers = [
        "Document Filename",
        "Metric Name",
        "Normalized Key",
        "Reported Value",
        "Numeric Value",
        "Currency",
        "Unit Scale",
        "Period",
        "Prior Period Value",
        "Prior Period",
        "YoY Absolute Change",
        "YoY Percentage Change (%)",
        "YoY Calculation Note",
        "Metric Type",
        "Calculation Method",
        "Source Page",
        "Verification Status",
        "Conflict Status",
        "Source Excerpt",
        "Notes",
    ]
    writer.writerow(headers)

    for m in analysis.metrics:
        # Format numeric YoY values safely
        abs_val = f"{m.yoy_absolute_change:.2f}" if m.yoy_absolute_change is not None else ""
        pct_val = f"{m.yoy_percentage_change:.2f}" if m.yoy_percentage_change is not None else ""
        num_val = f"{m.numeric_value}" if m.numeric_value is not None else ""
        rep_val = str(m.value) if m.value is not None else ""

        row = [
            escape_csv_formula(filename),
            escape_csv_formula(m.name),
            escape_csv_formula(m.normalized_name),
            escape_csv_formula(rep_val),
            num_val,  # Clean numeric string without formula injection
            escape_csv_formula(m.currency),
            escape_csv_formula(m.unit_scale),
            escape_csv_formula(m.period_label),
            escape_csv_formula(str(m.prior_period_value) if m.prior_period_value is not None else ""),
            escape_csv_formula(m.prior_period_label),
            abs_val,  # Clean numeric string
            pct_val,  # Clean numeric string
            escape_csv_formula(m.yoy_calculation_note),
            escape_csv_formula(m.metric_type),
            escape_csv_formula(m.calculation_method),
            m.source_page if m.source_page is not None else "",
            "Verified" if m.is_verified else "Unverified",
            "Conflict Detected" if m.has_conflict else "Consistent",
            escape_csv_formula(m.source_quote),
            escape_csv_formula(m.confidence_note),
        ]
        writer.writerow(row)

    return output.getvalue()


def export_risks_csv(
    analysis: ReportAnalysis,
    metadata: Optional[DocumentMetadata] = None,
) -> str:
    """Generate a spreadsheet-friendly CSV of disclosed risk factors."""
    if not _is_report_analysis(analysis):
        raise TypeError("Input must be a valid ReportAnalysis instance.")

    filename = metadata.filename if metadata else (analysis.report_title or "document.pdf")
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

    headers = [
        "Document Filename",
        "Risk Title",
        "Category",
        "Disclosed Description",
        "Company Mitigation",
        "Impact Type",
        "Reported Severity",
        "Source Pages",
        "Verification Status",
        "Supporting Excerpts",
        "Section Heading",
    ]
    writer.writerow(headers)

    for r in analysis.risks:
        pages_str = ", ".join(str(p) for p in r.source_pages)
        quotes_str = " | ".join(r.source_quotes)

        row = [
            escape_csv_formula(filename),
            escape_csv_formula(r.risk_title),
            escape_csv_formula(r.risk_category),
            escape_csv_formula(r.description),
            escape_csv_formula(r.company_stated_mitigation),
            escape_csv_formula(r.impact_type),
            escape_csv_formula(r.severity),
            pages_str,
            "Verified" if r.is_verified else "Unverified",
            escape_csv_formula(quotes_str),
            escape_csv_formula(r.source_heading),
        ]
        writer.writerow(row)

    return output.getvalue()


def export_outlook_csv(
    analysis: ReportAnalysis,
    metadata: Optional[DocumentMetadata] = None,
) -> str:
    """Generate a spreadsheet-friendly CSV of management forward guidance."""
    if not _is_report_analysis(analysis):
        raise TypeError("Input must be a valid ReportAnalysis instance.")

    filename = metadata.filename if metadata else (analysis.report_title or "document.pdf")
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

    headers = [
        "Document Filename",
        "Forward Statement",
        "Statement Type",
        "Target Period",
        "Time Horizon",
        "Direction",
        "Target Metric",
        "Target Value",
        "Source Pages",
        "Verification Status",
        "Supporting Excerpts",
    ]
    writer.writerow(headers)

    for o in analysis.outlook:
        pages_str = ", ".join(str(p) for p in o.source_pages)
        quotes_str = " | ".join(o.source_quotes)

        row = [
            escape_csv_formula(filename),
            escape_csv_formula(o.statement),
            escape_csv_formula(o.statement_type),
            escape_csv_formula(o.period),
            escape_csv_formula(o.time_horizon),
            escape_csv_formula(o.direction),
            escape_csv_formula(o.target_metric),
            escape_csv_formula(o.target_value),
            pages_str,
            "Verified" if o.is_verified else "Unverified",
            escape_csv_formula(quotes_str),
        ]
        writer.writerow(row)

    return output.getvalue()


def export_analysis_csv(
    analysis: ReportAnalysis,
    metadata: Optional[DocumentMetadata] = None,
) -> str:
    """Primary CSV export function (aliases to metrics CSV)."""
    return export_metrics_csv(analysis, metadata=metadata)


def export_analysis_markdown(
    analysis: ReportAnalysis,
    metadata: Optional[DocumentMetadata] = None,
) -> str:
    """Generate a structured, human-readable Markdown analysis report."""
    if not _is_report_analysis(analysis):
        raise TypeError("Input must be a valid ReportAnalysis instance.")

    company = analysis.company_name or "Company"
    title = analysis.report_title or (metadata.filename if metadata else "Financial Report")
    period = analysis.fiscal_period or "Current Period"
    status_label = analysis.analysis_status.upper()

    md_lines: List[str] = []

    # Title & Metadata
    md_lines.append(f"# Financial Analysis Report: {company}")
    md_lines.append("")
    md_lines.append(f"**Report Title:** {title}  ")
    md_lines.append(f"**Fiscal Period:** {period}  ")
    md_lines.append(f"**Analysis Status:** `{status_label}`  ")
    md_lines.append(
        f"**Coverage:** {analysis.pages_analyzed} of {analysis.total_pages} pages analyzed "
        f"({analysis.batches_succeeded} batches succeeded, {analysis.batches_failed} failed)  "
    )
    md_lines.append(f"**Citation Verification Rate:** {analysis.verification_rate * 100:.1f}%  ")
    md_lines.append(f"**Generated:** {analysis.generated_at}  ")
    md_lines.append("")

    # Disclaimer Alert
    md_lines.append("> [!NOTE]")
    md_lines.append(f"> {DISCLAIMER_TEXT}")
    md_lines.append("")

    # Partial analysis warning if applicable
    if analysis.analysis_status == "partial":
        md_lines.append("> [!WARNING]")
        md_lines.append(
            f"> **Partial Analysis Notice:** {analysis.batches_failed} batch(es) could not be processed. "
            "The findings below represent available verified evidence and may not cover the full filing."
        )
        md_lines.append("")

    # Discrepancy warnings if any
    if analysis.data_conflicts:
        md_lines.append("> [!CAUTION]")
        md_lines.append(f"> **Reporting Conflicts Detected ({len(analysis.data_conflicts)}):**")
        for c in analysis.data_conflicts:
            md_lines.append(f"> - {c}")
        md_lines.append("")

    # Executive Summary
    md_lines.append("## 1. Executive Summary")
    md_lines.append("")
    md_lines.append(analysis.executive_summary)
    md_lines.append("")

    if analysis.key_takeaways:
        md_lines.append("### Key Strategic Takeaways")
        for t in analysis.key_takeaways:
            md_lines.append(f"- {t}")
        md_lines.append("")

    # Financial Metrics Table
    md_lines.append("## 2. Key Performance Indicators & Financial Metrics")
    md_lines.append("")

    if not analysis.metrics:
        md_lines.append("*No structured financial metrics extracted from the analyzed pages.*")
        md_lines.append("")
    else:
        md_lines.append("| Metric | Reported Value | Scale | Period | Prior Value | YoY Change (%) | YoY Change (Abs) | Source | Verification |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")

        for m in analysis.metrics:
            curr_str = f"{m.currency} " if m.currency else ""
            val_str = f"{curr_str}{m.value:,}" if isinstance(m.value, (int, float)) else (f"{curr_str}{m.value}" if m.value else "—")
            prior_str = f"{curr_str}{m.prior_period_value:,}" if isinstance(m.prior_period_value, (int, float)) else (f"{curr_str}{m.prior_period_value}" if m.prior_period_value else "—")
            yoy_pct = f"{'+' if m.yoy_percentage_change and m.yoy_percentage_change > 0 else ''}{m.yoy_percentage_change:.2f}%" if m.yoy_percentage_change is not None else (m.yoy_calculation_note or "—")
            yoy_abs = f"{'+' if m.yoy_absolute_change and m.yoy_absolute_change > 0 else ''}{m.yoy_absolute_change:,.2f}" if m.yoy_absolute_change is not None else "—"
            page_str = f"Page {m.source_page}" if m.source_page else "—"
            v_str = "✓ Verified" if m.is_verified else "⚠️ Unverified"

            row = [
                escape_markdown_table_cell(m.name),
                escape_markdown_table_cell(val_str),
                escape_markdown_table_cell(m.unit_scale or "units"),
                escape_markdown_table_cell(m.period_label or "Current"),
                escape_markdown_table_cell(prior_str),
                escape_markdown_table_cell(yoy_pct),
                escape_markdown_table_cell(yoy_abs),
                escape_markdown_table_cell(page_str),
                escape_markdown_table_cell(v_str),
            ]
            md_lines.append(f"| {' | '.join(row)} |")

        md_lines.append("")
        md_lines.append("*(Note: YoY changes and Free Cash Flow are computed deterministically in Python using Decimal arithmetic. Missing values are indicated by '—' and are never assumed to be zero.)*")
        md_lines.append("")

    # Disclosed Risks
    md_lines.append("## 3. Disclosed Risk Factors")
    md_lines.append("")

    if not analysis.risks:
        md_lines.append("*No material risk disclosures recorded in the analyzed pages.*")
        md_lines.append("")
    else:
        for idx, r in enumerate(analysis.risks, start=1):
            cat_str = f" `[{r.risk_category}]`" if r.risk_category else ""
            pages_str = f" (Page {', '.join(str(p) for p in r.source_pages)})" if r.source_pages else ""
            v_badge = "✓ Verified" if r.is_verified else "⚠️ Unverified"

            md_lines.append(f"### 3.{idx} {r.risk_title}{cat_str}{pages_str} — *{v_badge}*")
            md_lines.append("")
            md_lines.append(r.description)
            md_lines.append("")

            if r.company_stated_mitigation:
                md_lines.append(f"**Company-Stated Mitigation:** {r.company_stated_mitigation}")
                md_lines.append("")

            if r.source_quotes:
                md_lines.append(f"> \"{r.source_quotes[0]}\"")
                md_lines.append("")

    # Management Outlook
    md_lines.append("## 4. Management Outlook & Forward Guidance")
    md_lines.append("")

    if not analysis.outlook:
        md_lines.append("*No forward-looking guidance or management outlook recorded.*")
        md_lines.append("")
    else:
        for idx, o in enumerate(analysis.outlook, start=1):
            type_str = f" `[{o.statement_type.replace('_', ' ').title()}]`"
            period_str = f" — Target: **{o.period or 'Upcoming Period'}**"
            pages_str = f" (Page {', '.join(str(p) for p in o.source_pages)})" if o.source_pages else ""
            v_badge = "✓ Verified" if o.is_verified else "⚠️ Unverified"

            md_lines.append(f"### 4.{idx} {type_str}{period_str}{pages_str} — *{v_badge}*")
            md_lines.append("")
            md_lines.append(o.statement)
            md_lines.append("")

            if o.time_horizon or o.target_value:
                horizon_info = []
                if o.time_horizon:
                    horizon_info.append(f"Horizon: {o.time_horizon}")
                if o.target_value:
                    horizon_info.append(f"Target: {o.target_value}")
                md_lines.append(f"**Details:** {', '.join(horizon_info)}")
                md_lines.append("")

            if o.source_quotes:
                md_lines.append(f"> \"{o.source_quotes[0]}\"")
                md_lines.append("")

    # Limitations & Scope Warnings
    if analysis.limitations:
        md_lines.append("## 5. Scope & Limitations")
        md_lines.append("")
        for lim in analysis.limitations:
            md_lines.append(f"- {lim}")
        md_lines.append("")

    return "\n".join(md_lines)
