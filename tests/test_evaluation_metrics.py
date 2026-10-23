"""Unit tests for evaluation metric formulas and edge case handling."""

import pytest
from decimal import Decimal
from financial_report_analyzer.schemas import Citation, ChunkRecord

from evals.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    evaluate_citation_precision,
    evaluate_citation_page_accuracy,
    evaluate_citation_quote_accuracy,
    evaluate_citation_numeric_accuracy,
    calculate_abstention_contingency,
    evaluate_calculation_accuracy,
    evaluate_extraction_match_rates,
    calculate_precision_recall_f1,
)


def test_recall_at_k_normal_and_edge_cases():
    """Verify Recall@k formula and empty set handling."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c2", "c4", "c6"]

    # In top 3: c1, c2, c3 -> 1 hit out of 3 relevant
    res_k3 = calculate_recall_at_k(retrieved, relevant, k=3)
    assert res_k3.numerator == 1.0
    assert res_k3.denominator == 3.0
    assert pytest.approx(res_k3.value) == 1.0 / 3.0

    # In top 5: c1, c2, c3, c4, c5 -> 2 hits out of 3 relevant
    res_k5 = calculate_recall_at_k(retrieved, relevant, k=5)
    assert res_k5.numerator == 2.0
    assert res_k5.denominator == 3.0
    assert pytest.approx(res_k5.value) == 2.0 / 3.0

    # Empty relevant items handling
    res_empty = calculate_recall_at_k(retrieved, [], k=5)
    assert res_empty.is_defined is False
    assert res_empty.value == 0.0

    # Negative or zero k
    res_zero_k = calculate_recall_at_k(retrieved, relevant, k=0)
    assert res_zero_k.value == 0.0


def test_precision_at_k_normal_and_edge_cases():
    """Verify Precision@k formula."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c2", "c4"]

    # Top 2: c1, c2 -> 1 hit out of 2
    res_k2 = calculate_precision_at_k(retrieved, relevant, k=2)
    assert res_k2.numerator == 1.0
    assert res_k2.denominator == 2.0
    assert res_k2.value == 0.5

    # Top 5: c1..c5 -> 2 hits out of 5
    res_k5 = calculate_precision_at_k(retrieved, relevant, k=5)
    assert res_k5.numerator == 2.0
    assert res_k5.denominator == 5.0
    assert res_k5.value == 0.4

    # k <= 0
    res_zero = calculate_precision_at_k(retrieved, relevant, k=0)
    assert res_zero.value == 0.0
    assert res_zero.is_defined is False


def test_mrr_calculation():
    """Verify Mean Reciprocal Rank (MRR) formula."""
    # First item relevant -> rank 1 -> MRR = 1.0
    assert calculate_mrr(["c1", "c2"], ["c1"]).value == 1.0

    # Second item relevant -> rank 2 -> MRR = 0.5
    assert calculate_mrr(["c1", "c2", "c3"], ["c2"]).value == 0.5

    # Third item relevant -> rank 3 -> MRR = 1/3
    res_r3 = calculate_mrr(["c1", "c2", "c3"], ["c3"])
    assert pytest.approx(res_r3.value) == 1.0 / 3.0

    # No items relevant -> MRR = 0.0
    assert calculate_mrr(["c1", "c2"], ["c99"]).value == 0.0

    # Empty retrieved list
    assert calculate_mrr([], ["c1"]).value == 0.0


def test_citation_evaluation_metrics():
    """Verify citation precision, page accuracy, quote accuracy, and numeric support."""
    sample_chunk = ChunkRecord(
        chunk_id="chunk_1",
        document_id="doc_1",
        source_filename="doc.pdf",
        page_numbers=[2],
        chunk_index=0,
        text="Revenue was $1,250.0 million in FY2024.",
    )
    chunks_map = {"chunk_1": sample_chunk}

    valid_cit = Citation(chunk_id="chunk_1", page_number=2, quote="Revenue was $1,250.0 million")
    wrong_page_cit = Citation(chunk_id="chunk_1", page_number=3, quote="Revenue was $1,250.0 million")
    hallucinated_quote_cit = Citation(chunk_id="chunk_1", page_number=2, quote="EBITDA was $400.0 million")
    fabricated_num_cit = Citation(chunk_id="chunk_1", page_number=2, quote="Revenue was $9,999.0 million")
    unknown_chunk_cit = Citation(chunk_id="chunk_99", page_number=2, quote="Some text")

    # 1. Precision: only valid_cit is valid out of 5
    prec = evaluate_citation_precision(
        [valid_cit, wrong_page_cit, hallucinated_quote_cit, fabricated_num_cit, unknown_chunk_cit],
        chunks_map,
    )
    assert prec.numerator == 1.0
    assert prec.denominator == 5.0
    assert prec.value == 0.2

    # 2. Page accuracy: valid_cit, hallucinated_quote_cit, fabricated_num_cit reference page 2 correctly
    p_acc = evaluate_citation_page_accuracy(
        [valid_cit, wrong_page_cit, hallucinated_quote_cit],
        chunks_map,
    )
    assert p_acc.numerator == 2.0
    assert p_acc.denominator == 3.0

    # 3. Numeric accuracy: valid_cit (1250 in chunk), fabricated_num_cit (9999 not in chunk)
    num_acc = evaluate_citation_numeric_accuracy(
        [valid_cit, fabricated_num_cit],
        chunks_map,
    )
    assert num_acc.numerator == 1.0
    assert num_acc.denominator == 2.0
    assert num_acc.value == 0.5


def test_abstention_contingency_metrics():
    """Verify abstention contingency table formulas."""
    metrics = calculate_abstention_contingency(
        unanswerable_abstained=9,
        unanswerable_answered=1,
        answerable_answered=18,
        answerable_abstained=2,
    )
    # Correct abstention: 9 / 10 = 0.90
    assert metrics.correct_abstention_rate.value == 0.90
    # False answer: 1 / 10 = 0.10
    assert metrics.false_answer_rate.value == 0.10
    # Answer rate: 18 / 20 = 0.90
    assert metrics.answer_rate.value == 0.90
    # Unnecessary abstention: 2 / 20 = 0.10
    assert metrics.unnecessary_abstention_rate.value == 0.10


def test_precision_recall_f1():
    """Verify precision, recall, and F1 calculations with documented zero handling."""
    res = calculate_precision_recall_f1(tp=8, fp=2, fn=2, label="test")
    assert res["test_precision"].value == 8.0 / 10.0  # 0.8
    assert res["test_recall"].value == 8.0 / 10.0     # 0.8
    assert pytest.approx(res["test_f1"].value) == 0.8

    # Zero hits
    res_zero = calculate_precision_recall_f1(tp=0, fp=5, fn=5, label="test")
    assert res_zero["test_precision"].value == 0.0
    assert res_zero["test_recall"].value == 0.0
    assert res_zero["test_f1"].value == 0.0
