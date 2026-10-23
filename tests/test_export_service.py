"""Unit tests for FinSight export service (JSON, CSV, and Markdown generation)."""

import json
import csv
import io
from decimal import Decimal
import pytest

from financial_report_analyzer.schemas import (
    ReportAnalysis,
    FinancialMetric,
    RiskItem,
    OutlookItem,
    DocumentMetadata,
)
from financial_report_analyzer.export_service import (
    export_analysis_json,
    export_metrics_csv,
    export_risks_csv,
    export_outlook_csv,
    export_analysis_csv,
    export_analysis_markdown,
    sanitize_export_filename,
    escape_csv_formula,
    EXPORT_SCHEMA_VERSION,
)


@pytest.fixture
def sample_analysis():
    """Predictable ReportAnalysis fixture with complete financial metrics, risks, and outlook."""
    return ReportAnalysis(
        report_title="Acme Corp Form 10-K",
        company_name="Acme Corporation",
        report_type="Form 10-K",
        fiscal_period="FY2024",
        executive_summary="Acme Corporation delivered record performance in FY2024 driven by cloud services.",
        key_takeaways=[
            "Total revenue grew 15.00% YoY to $96,770 million.",
            "Operating cash flow reached $30,000 million with strong cash conversion.",
        ],
        metrics=[
            FinancialMetric(
                name="Total Net Sales",
                normalized_name="revenue",
                value=96770.0,
                numeric_value=96770.0,
                currency="USD",
                unit_scale="millions",
                period_label="FY2024",
                prior_period_value=84150.0,
                prior_period_label="FY2023",
                yoy_absolute_change=12620.0,
                yoy_percentage_change=15.0,
                yoy_calculation_note="+15.00% YoY",
                metric_type="reported",
                source_page=14,
                source_quote="Total net sales was $96,770 million for FY2024",
                status="found",
                is_verified=True,
            ),
            FinancialMetric(
                name="Net Income",
                normalized_name="net_income",
                value=22100.0,
                numeric_value=22100.0,
                currency="USD",
                unit_scale="millions",
                period_label="FY2024",
                prior_period_value=None,  # Missing prior value
                source_page=15,
                source_quote="Net income was $22,100 million",
                status="found",
                is_verified=True,
            ),
            FinancialMetric(
                name="Free Cash Flow",
                normalized_name="free_cash_flow",
                value=20000.0,
                numeric_value=20000.0,
                currency="USD",
                unit_scale="millions",
                period_label="FY2024",
                metric_type="calculated",
                calculation_method="Operating Cash Flow - Capital Expenditures",
                source_page=45,
                source_quote="OCF: $30,000 million; Capex: $10,000 million",
                status="found",
                is_verified=True,
            ),
        ],
        risks=[
            RiskItem(
                risk_title="Semiconductor Supply Chain Disruption",
                description="Shortages in global semiconductor manufacturing may constrain hardware gross margin.",
                risk_category="Supply Chain",
                company_stated_mitigation="We maintain multi-source vendor agreements to mitigate single-source failure.",
                severity="High",
                source_pages=[28],
                source_quotes=["Shortages in global semiconductor manufacturing may constrain hardware gross margin."],
                impact_type="qualitative",
                is_verified=True,
                source_heading="Item 1A. Risk Factors",
            )
        ],
        outlook=[
            OutlookItem(
                statement="Management anticipates 10% operating margin expansion in FY2025.",
                period="FY2025",
                statement_type="guidance",
                source_pages=[32],
                source_quotes=["Management anticipates 10% operating margin expansion in FY2025."],
                time_horizon="FY2025",
                direction="positive",
                target_metric="Operating Margin",
                target_value="10%",
                is_verified=True,
            )
        ],
        limitations=["Scan quality warning on page 4"],
        generated_at="2026-10-23T12:00:00Z",
        analysis_status="complete",
        pages_analyzed=50,
        total_pages=50,
        batches_succeeded=5,
        batches_failed=0,
        data_conflicts=[],
        verification_rate=1.0,
    )


@pytest.fixture
def sample_metadata():
    """DocumentMetadata fixture."""
    return DocumentMetadata(
        document_id="doc_acme_2024",
        filename="acme_corp_10k.pdf",
        file_size_bytes=1024 * 750,
        total_pages=50,
        meaningful_pages=50,
        indexed=True,
    )


# --- JSON Export Tests ---

def test_json_export_successful(sample_analysis, sample_metadata):
    """Verify JSON export contains canonical structure, versioning, and parses cleanly."""
    json_str = export_analysis_json(sample_analysis, metadata=sample_metadata)
    data = json.loads(json_str)

    assert data["export_metadata"]["schema_version"] == EXPORT_SCHEMA_VERSION
    assert "disclaimer" in data["export_metadata"]
    assert data["document"]["document_id"] == "doc_acme_2024"
    assert data["document"]["source_filename"] == "acme_corp_10k.pdf"
    assert data["filing_info"]["company_name"] == "Acme Corporation"
    assert data["filing_info"]["fiscal_period"] == "FY2024"
    assert data["coverage"]["analysis_status"] == "complete"
    assert data["coverage"]["pages_analyzed"] == 50

    # Metrics
    metrics = data["financial_metrics"]
    assert len(metrics) == 3
    rev = next(m for m in metrics if m["normalized_name"] == "revenue")
    assert rev["reported_value"] == 96770.0
    assert rev["yoy_percentage_change"] == 15.0
    assert rev["is_verified"] is True

    # Missing value is null, not 0
    ni = next(m for m in metrics if m["normalized_name"] == "net_income")
    assert ni["prior_period_value"] is None

    # Calculated metric
    fcf = next(m for m in metrics if m["normalized_name"] == "free_cash_flow")
    assert fcf["metric_type"] == "calculated"

    # Risks and Outlook
    assert len(data["disclosed_risks"]) == 1
    assert data["disclosed_risks"][0]["risk_category"] == "Supply Chain"
    assert len(data["management_outlook"]) == 1
    assert data["management_outlook"][0]["direction"] == "positive"


def test_json_export_partial_analysis():
    """Verify JSON export accurately reports partial status and limitations."""
    partial_analysis = ReportAnalysis(
        report_title="Beta Inc. 10-Q",
        company_name="Beta Inc.",
        fiscal_period="Q3 2024",
        executive_summary="Partial results.",
        metrics=[],
        risks=[],
        outlook=[],
        limitations=["Batch 2 failed due to rate limit"],
        generated_at="2026-10-23T12:00:00Z",
        analysis_status="partial",
        pages_analyzed=10,
        total_pages=20,
        batches_succeeded=1,
        batches_failed=1,
        data_conflicts=["Conflict detected between p. 12 and p. 18"],
        verification_rate=0.85,
    )

    json_str = export_analysis_json(partial_analysis)
    data = json.loads(json_str)

    assert data["coverage"]["analysis_status"] == "partial"
    assert data["coverage"]["batches_failed"] == 1
    assert len(data["data_conflicts"]) == 1
    assert "Batch 2 failed" in data["limitations"][0]


# --- CSV Formula Injection & Export Tests ---

def test_csv_formula_injection_defense():
    """Verify formula injection triggers are prepended with a quote in text fields."""
    # Malicious spreadsheet formula triggers
    assert escape_csv_formula("=cmd|' /C calc'!A0") == "'=cmd|' /C calc'!A0"
    assert escape_csv_formula("+12345_dangerous_macro") == "'+12345_dangerous_macro"
    assert escape_csv_formula("-cmd|' /C calc'!A0") == "'-cmd|' /C calc'!A0"
    assert escape_csv_formula("@SUM(A1:A10)") == "'@SUM(A1:A10)"
    assert escape_csv_formula("\t=cmd") == "'\t=cmd"

    # Standard benign text remains unescaped
    assert escape_csv_formula("Revenue") == "Revenue"
    assert escape_csv_formula("Normal risk description.") == "Normal risk description."

    # Pure numeric values in numeric columns are NOT broken
    assert escape_csv_formula(Decimal("-15.00")) == "-15.00"
    assert escape_csv_formula(-25000.0) == "-25000.0"


def test_metrics_csv_export(sample_analysis, sample_metadata):
    """Verify metrics CSV generation, headers, row count, and parseability."""
    csv_str = export_metrics_csv(sample_analysis, metadata=sample_metadata)
    reader = list(csv.reader(io.StringIO(csv_str)))

    # Header row
    headers = reader[0]
    assert "Metric Name" in headers
    assert "YoY Percentage Change (%)" in headers
    assert "Verification Status" in headers

    # Rows (3 metrics = 3 data rows)
    assert len(reader) == 4

    # Check row 1 (Revenue)
    row_dict = dict(zip(headers, reader[1]))
    assert row_dict["Metric Name"] == "Total Net Sales"
    assert row_dict["Reported Value"] == "96770.0"
    assert row_dict["YoY Percentage Change (%)"] == "15.00"
    assert row_dict["Verification Status"] == "Verified"
    assert row_dict["Metric Type"] == "reported"

    # Check row 3 (Free Cash Flow)
    row_dict3 = dict(zip(headers, reader[3]))
    assert row_dict3["Metric Name"] == "Free Cash Flow"
    assert row_dict3["Metric Type"] == "calculated"
    assert "Operating Cash Flow" in row_dict3["Calculation Method"]


def test_risks_and_outlook_csv_export(sample_analysis, sample_metadata):
    """Verify risks and outlook CSV exports."""
    risks_csv = export_risks_csv(sample_analysis, metadata=sample_metadata)
    risks_rows = list(csv.reader(io.StringIO(risks_csv)))
    assert len(risks_rows) == 2  # header + 1 risk
    assert risks_rows[0][1] == "Risk Title"
    assert risks_rows[1][1] == "Semiconductor Supply Chain Disruption"
    assert risks_rows[1][4] == "We maintain multi-source vendor agreements to mitigate single-source failure."

    outlook_csv = export_outlook_csv(sample_analysis, metadata=sample_metadata)
    outlook_rows = list(csv.reader(io.StringIO(outlook_csv)))
    assert len(outlook_rows) == 2  # header + 1 outlook
    assert outlook_rows[0][1] == "Forward Statement"
    assert "operating margin expansion" in outlook_rows[1][1]


def test_empty_collections_csv(sample_metadata):
    """Verify empty collections still return valid CSV with correct headers."""
    empty_analysis = ReportAnalysis(
        report_title="Empty Report",
        executive_summary="Empty.",
        metrics=[],
        risks=[],
        outlook=[],
        generated_at="2026-10-23T12:00:00Z",
    )
    m_csv = export_metrics_csv(empty_analysis, metadata=sample_metadata)
    m_rows = list(csv.reader(io.StringIO(m_csv)))
    assert len(m_rows) == 1  # Header only
    assert "Metric Name" in m_rows[0]

    r_csv = export_risks_csv(empty_analysis, metadata=sample_metadata)
    r_rows = list(csv.reader(io.StringIO(r_csv)))
    assert len(r_rows) == 1
    assert "Risk Title" in r_rows[0]


# --- Markdown Export Tests ---

def test_markdown_export(sample_analysis, sample_metadata):
    """Verify Markdown report formatting, sections, tables, and alerts."""
    md = export_analysis_markdown(sample_analysis, metadata=sample_metadata)

    assert "# Financial Analysis Report: Acme Corporation" in md
    assert "**Report Title:** Acme Corp Form 10-K" in md
    assert "**Analysis Status:** `COMPLETE`" in md
    assert "> [!NOTE]" in md  # GitHub style alert for disclaimer
    assert "## 1. Executive Summary" in md
    assert "Acme Corporation delivered record performance" in md
    assert "## 2. Key Performance Indicators & Financial Metrics" in md
    assert "| Total Net Sales |" in md
    assert "| Free Cash Flow |" in md
    assert "## 3. Disclosed Risk Factors" in md
    assert "Semiconductor Supply Chain Disruption" in md
    assert "## 4. Management Outlook & Forward Guidance" in md
    assert "Management anticipates 10% operating margin expansion" in md


def test_markdown_export_pipe_escaping():
    """Verify pipe characters in user text do not break Markdown table columns."""
    analysis = ReportAnalysis(
        report_title="Test Filing",
        executive_summary="Summary",
        metrics=[
            FinancialMetric(
                name="Segment | Europe & Americas",
                value=5000.0,
                source_page=12,
                source_quote="Segment | Europe generated $5,000",
            )
        ],
        risks=[],
        outlook=[],
        generated_at="2026-10-23T12:00:00Z",
    )
    md = export_analysis_markdown(analysis)
    # The pipe should be escaped as \| in table rows
    assert "Segment \\| Europe & Americas" in md


# --- Filename Sanitization Tests ---

def test_sanitize_export_filename():
    """Verify path traversal and forbidden characters are stripped from filenames."""
    assert (
        sanitize_export_filename("apple_10k.pdf", "analysis", "json")
        == "apple_10k_analysis.json"
    )
    assert (
        sanitize_export_filename("../../etc/passwd", "metrics", "csv")
        == "etc_passwd_metrics.csv"
    )
    assert (
        sanitize_export_filename("report<with>:bad*chars.pdf", "summary", "md")
        == "report_with__bad_chars_summary.md"
    )
    assert (
        sanitize_export_filename(None, "data", "json")
        == "financial_report_data.json"
    )
