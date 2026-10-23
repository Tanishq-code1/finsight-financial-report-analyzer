"""Independent validation and security tests for FinSight exports."""

import json
import csv
import io
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
    export_analysis_markdown,
)


@pytest.fixture
def independent_fixture():
    """Independently constructed fixture with diverse financial values, signs, and unicode."""
    return ReportAnalysis(
        report_title="Global Tech Corp 2024 Annual Report",
        company_name="Global Tech Corp",
        report_type="Annual Report",
        fiscal_period="FY2024",
        executive_summary="Global Tech demonstrated revenue acceleration and operating margin expansion.",
        key_takeaways=["Services revenue increased 22%."],
        metrics=[
            FinancialMetric(
                name="Gross Margin",
                value="44.5%",
                currency="USD",
                unit_scale="percent",
                period_label="FY2024",
                metric_type="reported",
                source_page=12,
                source_quote="Gross margin was 44.5%",
                status="found",
                is_verified=True,
            ),
            FinancialMetric(
                name="Capital Expenditures",
                value="(12500.50)",  # Parenthetical negative
                numeric_value=-12500.50,
                currency="EUR",
                unit_scale="millions",
                period_label="FY2024",
                metric_type="reported",
                source_page=45,
                source_quote="Additions to property were €(12,500.50) million",
                status="found",
                is_verified=True,
            ),
        ],
        risks=[
            RiskItem(
                risk_title="Foreign Currency Exchange Risk",
                description="Fluctuations between USD, EUR, and JPY may impact reported revenues.",
                risk_category="Market Risk",
                source_pages=[18],
                source_quotes=["Fluctuations in foreign currencies may adversely impact revenue."],
                is_verified=True,
            )
        ],
        outlook=[
            OutlookItem(
                statement="Management forecasts 8% to 10% revenue growth in FY2025.",
                period="FY2025",
                statement_type="guidance",
                source_pages=[22],
                source_quotes=["Management forecasts 8% to 10% revenue growth in FY2025."],
                is_verified=True,
            )
        ],
        limitations=["Preliminary unaudited figures"],
        generated_at="2026-10-23T15:30:00Z",
        analysis_status="complete",
        pages_analyzed=100,
        total_pages=100,
        batches_succeeded=10,
        batches_failed=0,
        data_conflicts=[],
        verification_rate=1.0,
    )


def test_type_validation_rejection():
    """Verify that passing invalid non-ReportAnalysis objects raises TypeError."""
    with pytest.raises(TypeError):
        export_analysis_json({"invalid": "dict"})  # type: ignore

    with pytest.raises(TypeError):
        export_metrics_csv(None)  # type: ignore

    with pytest.raises(TypeError):
        export_analysis_markdown("string_instead_of_analysis")  # type: ignore


def test_roundtrip_json_fidelity(independent_fixture):
    """Verify all fields and provenance survive JSON serialization and round-trip parsing."""
    json_str = export_analysis_json(independent_fixture)
    data = json.loads(json_str)

    assert data["filing_info"]["company_name"] == "Global Tech Corp"
    assert len(data["financial_metrics"]) == 2

    # Check parenthetical negative value was preserved
    capex = next(m for m in data["financial_metrics"] if m["name"] == "Capital Expenditures")
    assert capex["reported_value"] == "(12500.50)"
    assert capex["numeric_value"] == -12500.50
    assert capex["currency"] == "EUR"
    assert capex["source_page"] == 45
    assert capex["is_verified"] is True


def test_roundtrip_csv_fidelity(independent_fixture):
    """Verify that CSV output can be read back with csv.DictReader and values match source."""
    csv_str = export_metrics_csv(independent_fixture)
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)

    assert len(rows) == 2
    row0 = rows[0]
    assert row0["Metric Name"] == "Gross Margin"
    assert row0["Reported Value"] == "44.5%"
    assert row0["Source Page"] == "12"

    row1 = rows[1]
    assert row1["Metric Name"] == "Capital Expenditures"
    assert row1["Reported Value"] == "(12500.50)"
    assert row1["Numeric Value"] == "-12500.5"
    assert row1["Currency"] == "EUR"


def test_no_secrets_or_private_paths_leaked(independent_fixture):
    """Verify exports do not leak environment variables, API keys, or absolute system paths."""
    mock_meta = DocumentMetadata(
        document_id="doc_safe_123",
        filename="safe_file.pdf",
        file_size_bytes=1000,
        total_pages=10,
        meaningful_pages=10,
    )

    json_str = export_analysis_json(independent_fixture, metadata=mock_meta)
    csv_str = export_metrics_csv(independent_fixture, metadata=mock_meta)
    md_str = export_analysis_markdown(independent_fixture, metadata=mock_meta)

    for content in (json_str, csv_str, md_str):
        # Must not contain API key patterns or local filesystem roots
        assert "AIzaSy" not in content
        assert "sk-" not in content
        assert "C:\\Users\\" not in content
        assert "/home/" not in content
        assert "GEMINI_API_KEY" not in content


def test_markdown_preserves_structure_with_quotes_and_negatives(independent_fixture):
    """Verify Markdown export contains proper tables, alerts, and negative values."""
    md = export_analysis_markdown(independent_fixture)

    assert "# Financial Analysis Report: Global Tech Corp" in md
    assert "(12500.50)" in md
    assert "EUR" in md
    assert "✓ Verified" in md
    assert "## 3. Disclosed Risk Factors" in md
    assert "Foreign Currency Exchange Risk" in md
