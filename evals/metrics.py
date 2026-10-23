"""Mathematical evaluation metric implementations for FinSight benchmark scorecard."""

import math
from decimal import Decimal
from typing import Sequence, List, Set, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from financial_report_analyzer.schemas import (
    Citation,
    ChunkRecord,
    FinancialMetric,
    ReportAnalysis,
    DocumentMetadata,
)
from financial_report_analyzer.citation_validator import (
    quote_matches_chunk_text,
    extract_key_numbers,
    normalize_text_for_matching,
)


class MetricResult(BaseModel):
    """Container for an evaluation score with explicit numerator, denominator, and value."""

    metric_name: str
    numerator: float
    denominator: float
    value: float
    is_defined: bool = True
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def calculate_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int = 5,
) -> MetricResult:
    """Calculate Recall@k: fraction of relevant items retrieved in top k.

    Policy:
    - If relevant_chunk_ids is empty, recall is considered undefined (or 1.0 if no items retrieved).
    - If k <= 0, recall is 0.0.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return MetricResult(
            metric_name=f"recall_at_{k}",
            numerator=0.0,
            denominator=0.0,
            value=1.0 if not retrieved_chunk_ids else 0.0,
            is_defined=False,
            note="Zero relevant items defined in ground truth",
        )

    top_k = retrieved_chunk_ids[: max(0, k)]
    hits = len(set(top_k) & rel_set)
    total_rel = len(rel_set)
    score = hits / float(total_rel)

    return MetricResult(
        metric_name=f"recall_at_{k}",
        numerator=float(hits),
        denominator=float(total_rel),
        value=score,
    )


def calculate_precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int = 5,
) -> MetricResult:
    """Calculate Precision@k: fraction of top k retrieved items that are relevant.

    Policy:
    - If k <= 0, precision is undefined/0.0.
    - Denominator is min(k, len(retrieved_chunk_ids)) if considering returned set, or fixed k.
      Here standard precision uses k as the denominator.
    """
    if k <= 0:
        return MetricResult(
            metric_name=f"precision_at_{k}",
            numerator=0.0,
            denominator=0.0,
            value=0.0,
            is_defined=False,
            note="k must be positive",
        )

    rel_set = set(relevant_chunk_ids)
    top_k = retrieved_chunk_ids[:k]
    hits = len(set(top_k) & rel_set)
    score = hits / float(k)

    return MetricResult(
        metric_name=f"precision_at_{k}",
        numerator=float(hits),
        denominator=float(k),
        value=score,
    )


def calculate_mrr(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> MetricResult:
    """Calculate Mean Reciprocal Rank (MRR): 1 / rank of first relevant item retrieved (1-based).

    Returns 0.0 if no relevant item is found in the retrieved list.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return MetricResult(
            metric_name="mrr",
            numerator=0.0,
            denominator=0.0,
            value=0.0,
            is_defined=False,
            note="Zero relevant items defined",
        )

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in rel_set:
            return MetricResult(
                metric_name="mrr",
                numerator=1.0,
                denominator=float(rank),
                value=1.0 / float(rank),
            )

    return MetricResult(
        metric_name="mrr",
        numerator=0.0,
        denominator=float(len(retrieved_chunk_ids) or 1),
        value=0.0,
    )


# ---------------------------------------------------------------------------
# Citation Quality Metrics
# ---------------------------------------------------------------------------

def evaluate_citation_precision(
    citations: Sequence[Citation],
    available_chunks: Dict[str, ChunkRecord],
) -> MetricResult:
    """Calculate Citation Precision: proportion of citations that reference real chunks with verified quotes."""
    if not citations:
        return MetricResult(
            metric_name="citation_precision",
            numerator=0.0,
            denominator=0.0,
            value=1.0,
            note="No citations evaluated (empty list)",
        )

    valid_count = 0
    for cit in citations:
        chunk = available_chunks.get(cit.chunk_id)
        if not chunk:
            continue
        # Verify page
        if cit.page_number not in chunk.page_numbers:
            continue
        # Verify quote
        is_match, _ = quote_matches_chunk_text(cit.quote, chunk.text)
        if is_match:
            valid_count += 1

    return MetricResult(
        metric_name="citation_precision",
        numerator=float(valid_count),
        denominator=float(len(citations)),
        value=valid_count / float(len(citations)),
    )


def evaluate_citation_page_accuracy(
    citations: Sequence[Citation],
    available_chunks: Dict[str, ChunkRecord],
) -> MetricResult:
    """Calculate Citation Page Accuracy: proportion of citations whose 1-based page matches chunk provenance."""
    if not citations:
        return MetricResult(
            metric_name="citation_page_accuracy",
            numerator=0.0,
            denominator=0.0,
            value=1.0,
            note="No citations evaluated",
        )

    correct_page_count = 0
    for cit in citations:
        chunk = available_chunks.get(cit.chunk_id)
        if chunk and cit.page_number in chunk.page_numbers:
            correct_page_count += 1

    return MetricResult(
        metric_name="citation_page_accuracy",
        numerator=float(correct_page_count),
        denominator=float(len(citations)),
        value=correct_page_count / float(len(citations)),
    )


def evaluate_citation_quote_accuracy(
    citations: Sequence[Citation],
    available_chunks: Dict[str, ChunkRecord],
) -> MetricResult:
    """Calculate Citation Quote Accuracy: proportion of citations with verbatim or normalized quotes in source chunk."""
    if not citations:
        return MetricResult(
            metric_name="citation_quote_accuracy",
            numerator=0.0,
            denominator=0.0,
            value=1.0,
            note="No citations evaluated",
        )

    valid_quote_count = 0
    for cit in citations:
        chunk = available_chunks.get(cit.chunk_id)
        if chunk:
            is_match, _ = quote_matches_chunk_text(cit.quote, chunk.text)
            if is_match:
                valid_quote_count += 1

    return MetricResult(
        metric_name="citation_quote_accuracy",
        numerator=float(valid_quote_count),
        denominator=float(len(citations)),
        value=valid_quote_count / float(len(citations)),
    )


def evaluate_citation_numeric_accuracy(
    citations: Sequence[Citation],
    available_chunks: Dict[str, ChunkRecord],
) -> MetricResult:
    """Calculate Citation Numeric Accuracy: proportion of citations whose numbers are supported in source chunk."""
    if not citations:
        return MetricResult(
            metric_name="citation_numeric_accuracy",
            numerator=0.0,
            denominator=0.0,
            value=1.0,
            note="No citations evaluated",
        )

    verified_count = 0
    for cit in citations:
        chunk = available_chunks.get(cit.chunk_id)
        if not chunk:
            continue
        quote_nums = extract_key_numbers(cit.quote)
        if not quote_nums:
            # No numbers in quote, deemed neutral / valid
            verified_count += 1
            continue

        chunk_nums = extract_key_numbers(chunk.text)
        if set(quote_nums).issubset(set(chunk_nums)):
            verified_count += 1

    return MetricResult(
        metric_name="citation_numeric_accuracy",
        numerator=float(verified_count),
        denominator=float(len(citations)),
        value=verified_count / float(len(citations)),
    )


# ---------------------------------------------------------------------------
# Abstention Quality Metrics
# ---------------------------------------------------------------------------

class AbstentionMetrics(BaseModel):
    """Four-way contingency metrics for evidence sufficiency and abstention."""

    correct_abstention_rate: MetricResult
    false_answer_rate: MetricResult
    answer_rate: MetricResult
    unnecessary_abstention_rate: MetricResult


def calculate_abstention_contingency(
    unanswerable_abstained: int,
    unanswerable_answered: int,
    answerable_answered: int,
    answerable_abstained: int,
) -> AbstentionMetrics:
    """Calculate explicit contingency table metrics for abstention behavior.

    Definitions:
    - Correct Abstention Rate: unanswerable_abstained / (unanswerable_abstained + unanswerable_answered)
    - False Answer Rate: unanswerable_answered / (unanswerable_abstained + unanswerable_answered)
    - Answer Rate: answerable_answered / (answerable_answered + answerable_abstained)
    - Unnecessary Abstention Rate: answerable_abstained / (answerable_answered + answerable_abstained)
    """
    total_unanswerable = unanswerable_abstained + unanswerable_answered
    total_answerable = answerable_answered + answerable_abstained

    corr_abs = (
        unanswerable_abstained / float(total_unanswerable)
        if total_unanswerable > 0
        else 1.0
    )
    false_ans = (
        unanswerable_answered / float(total_unanswerable)
        if total_unanswerable > 0
        else 0.0
    )
    ans_rate = (
        answerable_answered / float(total_answerable)
        if total_answerable > 0
        else 1.0
    )
    unnec_abs = (
        answerable_abstained / float(total_answerable)
        if total_answerable > 0
        else 0.0
    )

    return AbstentionMetrics(
        correct_abstention_rate=MetricResult(
            metric_name="correct_abstention_rate",
            numerator=float(unanswerable_abstained),
            denominator=float(total_unanswerable),
            value=corr_abs,
        ),
        false_answer_rate=MetricResult(
            metric_name="false_answer_rate",
            numerator=float(unanswerable_answered),
            denominator=float(total_unanswerable),
            value=false_ans,
        ),
        answer_rate=MetricResult(
            metric_name="answer_rate",
            numerator=float(answerable_answered),
            denominator=float(total_answerable),
            value=ans_rate,
        ),
        unnecessary_abstention_rate=MetricResult(
            metric_name="unnecessary_abstention_rate",
            numerator=float(answerable_abstained),
            denominator=float(total_answerable),
            value=unnec_abs,
        ),
    )


# ---------------------------------------------------------------------------
# Financial Calculation Metrics
# ---------------------------------------------------------------------------

def evaluate_calculation_accuracy(
    eligible_count: int,
    correct_count: int,
    withheld_eligible_count: int,
    withheld_correct_count: int,
) -> Tuple[MetricResult, MetricResult]:
    """Calculate financial calculation accuracy on eligible deterministic cases and properly withheld cases."""
    calc_acc = (
        correct_count / float(eligible_count)
        if eligible_count > 0
        else 1.0
    )
    withheld_acc = (
        withheld_correct_count / float(withheld_eligible_count)
        if withheld_eligible_count > 0
        else 1.0
    )

    return (
        MetricResult(
            metric_name="numeric_calculation_accuracy",
            numerator=float(correct_count),
            denominator=float(eligible_count),
            value=calc_acc,
        ),
        MetricResult(
            metric_name="withheld_calculation_accuracy",
            numerator=float(withheld_correct_count),
            denominator=float(withheld_eligible_count),
            value=withheld_acc,
        ),
    )


# ---------------------------------------------------------------------------
# Financial Extraction Metrics
# ---------------------------------------------------------------------------

def evaluate_extraction_match_rates(
    total_gold_assertions: int,
    matched_metric_and_period: int,
    matched_numeric_values: int,
    matched_currencies: int,
    matched_unit_scales: int,
    matched_source_pages: int,
) -> Dict[str, MetricResult]:
    """Calculate match rates for structured financial metrics against gold assertions."""
    denom = float(total_gold_assertions) if total_gold_assertions > 0 else 1.0

    return {
        "metric_exact_match_rate": MetricResult(
            metric_name="metric_exact_match_rate",
            numerator=float(matched_metric_and_period),
            denominator=denom,
            value=matched_metric_and_period / denom,
        ),
        "numeric_value_accuracy": MetricResult(
            metric_name="numeric_value_accuracy",
            numerator=float(matched_numeric_values),
            denominator=denom,
            value=matched_numeric_values / denom,
        ),
        "currency_accuracy": MetricResult(
            metric_name="currency_accuracy",
            numerator=float(matched_currencies),
            denominator=denom,
            value=matched_currencies / denom,
        ),
        "unit_scale_accuracy": MetricResult(
            metric_name="unit_scale_accuracy",
            numerator=float(matched_unit_scales),
            denominator=denom,
            value=matched_unit_scales / denom,
        ),
        "source_page_accuracy": MetricResult(
            metric_name="source_page_accuracy",
            numerator=float(matched_source_pages),
            denominator=denom,
            value=matched_source_pages / denom,
        ),
    }


def calculate_precision_recall_f1(
    tp: int,
    fp: int,
    fn: int,
    label: str = "item",
) -> Dict[str, MetricResult]:
    """Calculate standard precision, recall, and F1 with documented zero handling."""
    precision = tp / float(tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / float(tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        f"{label}_precision": MetricResult(
            metric_name=f"{label}_precision",
            numerator=float(tp),
            denominator=float(tp + fp),
            value=precision,
        ),
        f"{label}_recall": MetricResult(
            metric_name=f"{label}_recall",
            numerator=float(tp),
            denominator=float(tp + fn),
            value=recall,
        ),
        f"{label}_f1": MetricResult(
            metric_name=f"{label}_f1",
            numerator=float(tp),
            denominator=float(tp + 0.5 * (fp + fn)),
            value=f1,
        ),
    }
