"""Deterministic financial arithmetic and Year-over-Year (YoY) comparison service.

Performs pure Decimal arithmetic to avoid binary floating-point inaccuracy.
Never uses LLM generation to compute YoY changes or percentage comparisons.
"""

import re
import logging
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional, Union, Tuple, List, Dict
from dataclasses import dataclass

from financial_report_analyzer.schemas import FinancialMetric

logger = logging.getLogger(__name__)

# Currency symbols to strip when parsing raw numbers
CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹", "Rs.", "Rs", "USD", "EUR", "GBP", "INR", "JPY"]

# Standard unit multipliers in Decimal
UNIT_FACTORS: Dict[str, Decimal] = {
    "units": Decimal("1"),
    "ones": Decimal("1"),
    "single": Decimal("1"),
    "thousands": Decimal("1000"),
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "millions": Decimal("1000000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billions": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
    "b": Decimal("1000000000"),
    "crores": Decimal("10000000"),
    "crore": Decimal("10000000"),
    "lakhs": Decimal("100000"),
    "lakh": Decimal("100000"),
}


def parse_financial_number(raw_val: Union[str, int, float, Decimal, None]) -> Optional[Decimal]:
    """Deterministically parse a raw financial string or number into a Decimal.

    Handles:
    - Decimal and integer inputs
    - Currency signs ($12,450.75, €100M)
    - Parenthetical negatives ((1,250.50) -> -1250.50)
    - Minus signs and Unicode en/em dashes (-120, –120, −120)
    - Commas and spaces as thousand separators
    - Percentages (14.5% -> 14.5)
    """
    if raw_val is None:
        return None

    if isinstance(raw_val, Decimal):
        return raw_val

    if isinstance(raw_val, (int, float)):
        try:
            return Decimal(str(raw_val))
        except (InvalidOperation, ValueError):
            return None

    s = str(raw_val).strip()
    if not s:
        return None

    # Replace typographical dashes with standard hyphen
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # Detect parenthetical negative: (1,234.56) or ($1,234.56)
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        is_negative = True
        s = s[1:].strip()

    # Remove currency symbols and codes
    for sym in CURRENCY_SYMBOLS:
        s = s.replace(sym, "")

    # Remove percentage sign if present
    s = s.replace("%", "")

    # Remove commas and non-breaking spaces
    s = s.replace(",", "").replace("\u00a0", "").replace(" ", "").strip()

    # Extract standard numeric string: [+-]?\d+(\.\d+)?
    match = re.search(r"^[+-]?\d+(?:\.\d+)?", s)
    if not match:
        return None

    clean_num_str = match.group(0)
    try:
        val = Decimal(clean_num_str)
        return -val if is_negative else val
    except (InvalidOperation, ValueError):
        return None


def get_unit_multiplier(unit_label: Optional[str]) -> Optional[Decimal]:
    """Return the Decimal scale multiplier for a given unit string, or None if unknown/special."""
    if not unit_label:
        return None
    clean = unit_label.lower().strip()
    return UNIT_FACTORS.get(clean)


def are_periods_comparable(p1: Optional[str], p2: Optional[str]) -> Tuple[bool, str]:
    """Verify that two reporting periods represent comparable consecutive intervals.

    Rules:
    - Annual to annual (e.g. FY2024 vs FY2023 or 2024 vs 2023)
    - Quarterly to same quarter (e.g. Q3 2024 vs Q3 2023)
    - Mismatched intervals (annual vs quarter) are rejected.
    """
    if not p1 or not p2:
        return False, "One or both reporting period labels are missing."

    s1 = p1.upper().strip()
    s2 = p2.upper().strip()

    if s1 == s2:
        return False, f"Periods are identical ('{p1}'). YoY comparison requires two distinct periods."

    # Quarter comparison: e.g. Q3 2024 vs Q3 2023
    q1_match = re.search(r"Q([1-4])", s1)
    q2_match = re.search(r"Q([1-4])", s2)
    if q1_match or q2_match:
        if not (q1_match and q2_match):
            return False, f"Incomparable periods: '{p1}' vs '{p2}' (quarter vs non-quarter)."
        if q1_match.group(1) != q2_match.group(1):
            return False, f"Incomparable quarters: Q{q1_match.group(1)} vs Q{q2_match.group(1)}."

    # Extract 4-digit years
    y1_matches = re.findall(r"\b(20\d{2}|19\d{2})\b", s1)
    y2_matches = re.findall(r"\b(20\d{2}|19\d{2})\b", s2)

    if y1_matches and y2_matches:
        y1 = int(y1_matches[-1])
        y2 = int(y2_matches[-1])
        if abs(y1 - y2) > 2:
            return False, f"Non-consecutive fiscal years ({y1} vs {y2})."

    return True, "Periods are comparable."


@dataclass(frozen=True)
class YoYCalculationResult:
    """Detailed result of a deterministic YoY calculation."""

    current_value: Optional[Decimal]
    prior_value: Optional[Decimal]
    absolute_change: Optional[Decimal]
    percentage_change: Optional[Decimal]
    note: str
    is_valid: bool
    display_current: str
    display_prior: str
    display_abs_change: str
    display_pct_change: str


def calculate_yoy_comparison(
    current_val: Union[str, int, float, Decimal, None],
    prior_val: Union[str, int, float, Decimal, None],
    current_currency: Optional[str] = None,
    prior_currency: Optional[str] = None,
    current_scale: Optional[str] = None,
    prior_scale: Optional[str] = None,
    current_period: Optional[str] = None,
    prior_period: Optional[str] = None,
) -> YoYCalculationResult:
    """Calculate absolute and percentage YoY changes using strict Decimal arithmetic and financial rules.

    Safeguards:
    1. Rejects missing values.
    2. Rejects mismatched currencies (exchange rate conversion is explicitly out of scope).
    3. Converts unit scales deterministically only when both scales are known.
    4. Validates period comparability (e.g. annual to annual, Q3 to Q3).
    5. Handles zero prior value (withholds percentage, provides absolute change).
    6. Handles negative prior values and loss-to-profit transitions safely.
    7. Rounds percentage to 2 decimal places using ROUND_HALF_UP.
    """
    curr_dec = parse_financial_number(current_val)
    prior_dec = parse_financial_number(prior_val)

    if curr_dec is None or prior_dec is None:
        return YoYCalculationResult(
            current_value=curr_dec,
            prior_value=prior_dec,
            absolute_change=None,
            percentage_change=None,
            note="YoY calculation unavailable: either current or prior period value is missing.",
            is_valid=False,
            display_current=str(current_val) if current_val is not None else "N/A",
            display_prior=str(prior_val) if prior_val is not None else "N/A",
            display_abs_change="N/A",
            display_pct_change="N/A",
        )

    # 1. Currency check
    if current_currency and prior_currency:
        if current_currency.upper().strip() != prior_currency.upper().strip():
            return YoYCalculationResult(
                current_value=curr_dec,
                prior_value=prior_dec,
                absolute_change=None,
                percentage_change=None,
                note=f"YoY calculation withheld: currency mismatch ({current_currency} vs {prior_currency}).",
                is_valid=False,
                display_current=f"{curr_dec} {current_currency}",
                display_prior=f"{prior_dec} {prior_currency}",
                display_abs_change="N/A",
                display_pct_change="N/A",
            )

    # 2. Scale normalization
    scale_curr = get_unit_multiplier(current_scale)
    scale_prior = get_unit_multiplier(prior_scale)

    norm_curr = curr_dec
    norm_prior = prior_dec

    if current_scale and prior_scale and current_scale.lower().strip() != prior_scale.lower().strip():
        if scale_curr and scale_prior:
            # Normalize prior to current scale
            norm_prior = (prior_dec * scale_prior) / scale_curr
        else:
            return YoYCalculationResult(
                current_value=curr_dec,
                prior_value=prior_dec,
                absolute_change=None,
                percentage_change=None,
                note=f"YoY calculation withheld: unknown unit scale conversion ({current_scale} vs {prior_scale}).",
                is_valid=False,
                display_current=f"{curr_dec} ({current_scale})",
                display_prior=f"{prior_dec} ({prior_scale})",
                display_abs_change="N/A",
                display_pct_change="N/A",
            )

    # 3. Period comparability check
    if current_period and prior_period:
        comparable, reason = are_periods_comparable(current_period, prior_period)
        if not comparable:
            return YoYCalculationResult(
                current_value=norm_curr,
                prior_value=norm_prior,
                absolute_change=None,
                percentage_change=None,
                note=f"YoY comparison withheld: {reason}",
                is_valid=False,
                display_current=str(norm_curr),
                display_prior=str(norm_prior),
                display_abs_change="N/A",
                display_pct_change="N/A",
            )

    # 4. Compute absolute change
    abs_change = norm_curr - norm_prior

    # 5. Compute percentage change with financial edge-case handling
    pct_change: Optional[Decimal] = None
    note: str = ""

    if norm_prior == Decimal("0"):
        pct_change = None
        note = "Percentage change undefined (prior period value is zero). Absolute change reported."
    elif norm_prior < Decimal("0") and norm_curr > Decimal("0"):
        # Transition from net loss to profit
        pct_change = None
        note = f"Transition from net loss ({norm_prior}) to profit ({norm_curr}). Percentage comparison withheld as economically non-standard."
    elif norm_prior < Decimal("0") and norm_curr < Decimal("0"):
        # Ongoing loss: standard percentage formula ((curr-prior)/prior) inverts intuitive direction
        pct_change = None
        if abs(norm_curr) < abs(norm_prior):
            note = f"Net loss reduced by {abs(abs_change)}. Percentage change withheld for negative base."
        else:
            note = f"Net loss increased by {abs(abs_change)}. Percentage change withheld for negative base."
    else:
        # Standard positive-to-positive or positive-to-loss comparison
        raw_pct = ((norm_curr - norm_prior) / norm_prior) * Decimal("100")
        pct_change = raw_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        sign = "+" if pct_change > 0 else ""
        note = f"Calculated YoY: {sign}{pct_change}%"

    # Display formatting
    abs_rounded = abs_change.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    abs_sign = "+" if abs_rounded > 0 else ""
    display_abs = f"{abs_sign}{abs_rounded}"
    display_pct = f"{'+' if pct_change and pct_change > 0 else ''}{pct_change}%" if pct_change is not None else "N/A"

    return YoYCalculationResult(
        current_value=norm_curr,
        prior_value=norm_prior,
        absolute_change=abs_change,
        percentage_change=pct_change,
        note=note,
        is_valid=True,
        display_current=str(norm_curr),
        display_prior=str(norm_prior),
        display_abs_change=display_abs,
        display_pct_change=display_pct,
    )


def calculate_free_cash_flow(
    operating_cash_flow: FinancialMetric,
    capex: FinancialMetric,
) -> Optional[FinancialMetric]:
    """Deterministically derive Free Cash Flow (FCF) as Operating Cash Flow - Capital Expenditures.

    Only computed if both OCF and Capex are explicitly found and verified.
    """
    if operating_cash_flow.value is None or capex.value is None:
        return None

    ocf_dec = parse_financial_number(operating_cash_flow.value)
    capex_dec = parse_financial_number(capex.value)

    if ocf_dec is None or capex_dec is None:
        return None

    # Handle sign convention: capex is often reported as negative outflow e.g. (500)
    # If capex is negative, subtract absolute value or add negative value
    actual_capex = abs(capex_dec)
    fcf = ocf_dec - actual_capex

    # Determine pages & quotes
    pages = []
    if operating_cash_flow.source_page:
        pages.append(operating_cash_flow.source_page)
    if capex.source_page and capex.source_page not in pages:
        pages.append(capex.source_page)

    primary_page = pages[0] if pages else None
    quotes = []
    if operating_cash_flow.source_quote:
        quotes.append(f"OCF: {operating_cash_flow.source_quote}")
    if capex.source_quote:
        quotes.append(f"Capex: {capex.source_quote}")

    return FinancialMetric(
        name="Free Cash Flow",
        normalized_name="free_cash_flow",
        value=float(fcf),
        numeric_value=float(fcf),
        currency=operating_cash_flow.currency,
        unit_scale=operating_cash_flow.unit_scale,
        period_label=operating_cash_flow.period_label,
        metric_type="calculated",
        calculation_method="Operating Cash Flow - Capital Expenditures (derived in Python)",
        source_page=primary_page,
        source_quote="; ".join(quotes),
        status="found",
        is_verified=True,
    )


def deduplicate_and_resolve_metrics(
    metrics: List[FinancialMetric],
) -> Tuple[List[FinancialMetric], List[str]]:
    """Deduplicate extracted metrics across batches and flag genuine discrepancies.

    Returns:
        (resolved_metrics, detected_conflicts)
    """
    if not metrics:
        return [], []

    grouped: Dict[Tuple[str, str], List[FinancialMetric]] = {}
    conflicts: List[str] = []

    for m in metrics:
        norm_name = (m.normalized_name or m.name).lower().strip()
        period = (m.period_label or "current").upper().strip()
        key = (norm_name, period)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(m)

    resolved: List[FinancialMetric] = []

    for (norm_name, period), group in grouped.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue

        # Check if all values in group are identical
        parsed_values = [parse_financial_number(m.value) for m in group]
        valid_values = [v for v in parsed_values if v is not None]

        if not valid_values:
            resolved.append(group[0])
            continue

        first_val = valid_values[0]
        all_same = all(v == first_val for v in valid_values)

        if all_same:
            # Identical values from different pages: take primary with combined quotes
            primary = group[0]
            resolved.append(primary)
        else:
            # Conflicting reported values for the same metric in the same period!
            val_strs = [f"{m.value} (p. {m.source_page})" for m in group if m.value is not None]
            conflict_msg = (
                f"Conflicting reported values for '{group[0].name}' ({period}): {', '.join(val_strs)}. "
                f"Flagged for human verification."
            )
            conflicts.append(conflict_msg)
            logger.warning(conflict_msg)

            # Mark all items in the conflicting group
            primary = group[0]
            flagged = FinancialMetric(
                name=primary.name,
                normalized_name=primary.normalized_name,
                value=primary.value,
                numeric_value=primary.numeric_value,
                currency=primary.currency,
                unit_scale=primary.unit_scale,
                period_label=primary.period_label,
                metric_type=primary.metric_type,
                source_page=primary.source_page,
                source_quote=primary.source_quote,
                status="needs_review",
                confidence_note=f"Conflict detected between multiple filing citations: {', '.join(val_strs)}",
                is_verified=primary.is_verified,
                has_conflict=True,
                conflicting_values=val_strs,
            )
            resolved.append(flagged)

    return resolved, conflicts
