"""Unit tests for deterministic financial calculations and YoY comparison engine."""

from decimal import Decimal
import pytest

from financial_report_analyzer.schemas import FinancialMetric
from financial_report_analyzer.financial_calculations import (
    parse_financial_number,
    are_periods_comparable,
    calculate_yoy_comparison,
    calculate_free_cash_flow,
    deduplicate_and_resolve_metrics,
)


def test_parse_financial_number_formats():
    """Verify numeric parsing across currencies, commas, parentheses, and decimals."""
    assert parse_financial_number(100) == Decimal("100")
    assert parse_financial_number(12450.75) == Decimal("12450.75")
    assert parse_financial_number("$12,450.75") == Decimal("12450.75")
    assert parse_financial_number("($12,450.75)") == Decimal("-12450.75")
    assert parse_financial_number("(500)") == Decimal("-500")
    assert parse_financial_number("-120.50") == Decimal("-120.50")
    assert parse_financial_number("–450") == Decimal("-450")  # en-dash
    assert parse_financial_number("₹4,500") == Decimal("4500")
    assert parse_financial_number("14.5%") == Decimal("14.5")
    assert parse_financial_number(None) is None
    assert parse_financial_number("") is None
    assert parse_financial_number("N/A") is None


def test_are_periods_comparable():
    """Verify period comparability detection."""
    # Annual to annual
    comp, _ = are_periods_comparable("FY2024", "FY2023")
    assert comp is True

    comp, _ = are_periods_comparable("2024", "2023")
    assert comp is True

    # Same quarters
    comp, _ = are_periods_comparable("Q3 2024", "Q3 2023")
    assert comp is True

    # Mismatched quarters
    comp, reason = are_periods_comparable("Q3 2024", "Q2 2024")
    assert comp is False
    assert "Incomparable quarters" in reason

    # Quarter vs Annual
    comp, reason = are_periods_comparable("FY2024", "Q3 2024")
    assert comp is False

    # Identical periods
    comp, reason = are_periods_comparable("FY2024", "FY2024")
    assert comp is False
    assert "identical" in reason


def test_yoy_positive_growth():
    """Verify standard positive-to-positive revenue increase."""
    res = calculate_yoy_comparison(
        current_val="$96,770",
        prior_val="$84,150",
        current_currency="USD",
        prior_currency="USD",
        current_scale="millions",
        prior_scale="millions",
        current_period="FY2024",
        prior_period="FY2023",
    )
    assert res.is_valid is True
    assert res.absolute_change == Decimal("12620")
    assert res.percentage_change == Decimal("15.00")
    assert "+15.00%" in res.note
    assert res.display_pct_change == "+15.00%"


def test_yoy_revenue_decrease():
    """Verify YoY calculation when revenue decreases."""
    res = calculate_yoy_comparison(
        current_val=80000,
        prior_val=100000,
        current_period="FY2024",
        prior_period="FY2023",
    )
    assert res.is_valid is True
    assert res.absolute_change == Decimal("-20000")
    assert res.percentage_change == Decimal("-20.00")
    assert res.display_pct_change == "-20.00%"


def test_yoy_zero_prior_value():
    """Verify percentage change is withheld when prior value is zero."""
    res = calculate_yoy_comparison(
        current_val=5000,
        prior_val=0,
        current_period="FY2024",
        prior_period="FY2023",
    )
    assert res.is_valid is True
    assert res.absolute_change == Decimal("5000")
    assert res.percentage_change is None
    assert "prior period value is zero" in res.note
    assert res.display_pct_change == "N/A"


def test_yoy_loss_to_profit_transition():
    """Verify loss to profit transition withholds deceptive percentage."""
    res = calculate_yoy_comparison(
        current_val=1500,  # +$1,500 profit
        prior_val="(2000)",  # -$2,000 loss
        current_period="FY2024",
        prior_period="FY2023",
    )
    assert res.is_valid is True
    assert res.absolute_change == Decimal("3500")
    assert res.percentage_change is None
    assert "Transition from net loss" in res.note


def test_yoy_loss_reduction():
    """Verify ongoing negative values withhold misleading percentage."""
    res = calculate_yoy_comparison(
        current_val="-800",
        prior_val="-1500",
        current_period="FY2024",
        prior_period="FY2023",
    )
    assert res.is_valid is True
    assert res.absolute_change == Decimal("700")
    assert res.percentage_change is None
    assert "Net loss reduced" in res.note


def test_yoy_mismatched_currencies():
    """Verify currency mismatch is rejected safely."""
    res = calculate_yoy_comparison(
        current_val=1000,
        prior_val=900,
        current_currency="USD",
        prior_currency="EUR",
    )
    assert res.is_valid is False
    assert res.absolute_change is None
    assert "currency mismatch" in res.note


def test_yoy_scale_conversion():
    """Verify deterministic conversion between thousands and millions."""
    # Current: $50 million. Prior: $40,000 thousand ($40 million)
    res = calculate_yoy_comparison(
        current_val=50,
        prior_val=40000,
        current_scale="millions",
        prior_scale="thousands",
        current_period="FY2024",
        prior_period="FY2023",
    )
    assert res.is_valid is True
    assert res.current_value == Decimal("50")
    assert res.prior_value == Decimal("40")
    assert res.absolute_change == Decimal("10")
    assert res.percentage_change == Decimal("25.00")


def test_yoy_unknown_scale():
    """Verify unknown unit scale is rejected."""
    res = calculate_yoy_comparison(
        current_val=50,
        prior_val=40,
        current_scale="unknown_units",
        prior_scale="millions",
        current_period="FY2024",
        prior_period="FY2023",
    )
    assert res.is_valid is False
    assert "unknown unit scale" in res.note


def test_yoy_missing_values():
    """Verify missing values are handled gracefully."""
    res = calculate_yoy_comparison(
        current_val=None,
        prior_val=100,
    )
    assert res.is_valid is False
    assert "missing" in res.note


def test_calculate_free_cash_flow():
    """Verify Free Cash Flow derivation from OCF and Capex."""
    ocf = FinancialMetric(
        name="Operating Cash Flow",
        value=35000.0,
        currency="USD",
        unit_scale="millions",
        period_label="FY2024",
        source_page=45,
        source_quote="Cash generated by operating activities was $35,000 million",
    )
    capex = FinancialMetric(
        name="Capital Expenditures",
        value="(10000.0)",  # Outflow
        currency="USD",
        unit_scale="millions",
        period_label="FY2024",
        source_page=46,
        source_quote="Payments for property, plant and equipment were $10,000 million",
    )

    fcf = calculate_free_cash_flow(ocf, capex)
    assert fcf is not None
    assert fcf.name == "Free Cash Flow"
    assert fcf.value == 25000.0
    assert fcf.metric_type == "calculated"
    assert "Operating Cash Flow - Capital Expenditures" in fcf.calculation_method
    assert fcf.source_page == 45
    assert "OCF:" in fcf.source_quote
    assert "Capex:" in fcf.source_quote


def test_deduplicate_identical_metrics():
    """Verify duplicate identical metrics across batches are consolidated."""
    m1 = FinancialMetric(
        name="Revenue",
        normalized_name="revenue",
        value=100000.0,
        period_label="FY2024",
        currency="USD",
        source_page=12,
    )
    m2 = FinancialMetric(
        name="Total Net Sales",
        normalized_name="revenue",
        value=100000.0,
        period_label="FY2024",
        currency="USD",
        source_page=35,
    )

    resolved, conflicts = deduplicate_and_resolve_metrics([m1, m2])
    assert len(resolved) == 1
    assert len(conflicts) == 0
    assert resolved[0].value == 100000.0


def test_deduplicate_conflicting_metrics():
    """Verify contradictory figures for the same metric/period are flagged."""
    m1 = FinancialMetric(
        name="Net Income",
        normalized_name="net_income",
        value=25000.0,
        period_label="FY2024",
        currency="USD",
        source_page=14,
    )
    m2 = FinancialMetric(
        name="Net Income",
        normalized_name="net_income",
        value=28000.0,
        period_label="FY2024",
        currency="USD",
        source_page=48,
    )

    resolved, conflicts = deduplicate_and_resolve_metrics([m1, m2])
    assert len(resolved) == 1
    assert len(conflicts) == 1
    assert resolved[0].has_conflict is True
    assert resolved[0].status == "needs_review"
    assert "Conflicting reported values" in conflicts[0]
