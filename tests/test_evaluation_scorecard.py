"""Tests verifying evaluation scorecard formatting, JSON/Markdown output, and baseline comparison."""

import pytest
from pathlib import Path
from evals.report import (
    EvaluationScorecard,
    format_terminal_summary,
    format_markdown_scorecard,
    save_scorecard_artifacts,
)
from evals.compare import compare_scorecards


@pytest.fixture
def sample_scorecard() -> EvaluationScorecard:
    return EvaluationScorecard(
        run_id="run_test_001",
        started_at_utc="2026-10-23T12:00:00Z",
        completed_at_utc="2026-10-23T12:00:05Z",
        dataset_version="1.0.0",
        git_commit="abcdef123456",
        app_version="0.1.0",
        mode="offline",
        configuration={
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
            "generation_model": "gemini-3.8-flash",
            "retrieval_top_k": 5,
        },
        case_summary={"total": 110, "executed": 110, "passed": 110, "failed": 0, "skipped": 0},
        category_counts={"qa_cases": 30, "retrieval_cases": 20},
        metrics={
            "retrieval_recall_at_5": {"numerator": 20.0, "denominator": 20.0, "value": 1.0},
            "citation_precision": {"numerator": 30.0, "denominator": 30.0, "value": 1.0},
        },
        quality_gates={
            "overall_status": "PASSED",
            "is_provisional": True,
            "results": {
                "retrieval_recall_at_5": {"target": 0.85, "actual": 1.0, "status": "PASS"},
                "citation_precision": {"target": 0.95, "actual": 1.0, "status": "PASS"},
            },
        },
        failures=[],
        latency_summary={"total_elapsed_seconds": 1.25},
        warnings=["Provisional test warning."],
    )


def test_scorecard_markdown_and_terminal_formatting(sample_scorecard):
    """Verify Markdown and Terminal summary generators include critical metadata."""
    term_out = format_terminal_summary(sample_scorecard)
    assert "FINLIGHT (FinSight) EVALUATION BENCHMARK SCORECARD" in term_out
    assert "run_test_001" in term_out
    assert "OVERALL QUALITY GATE RESULT: PASSED" in term_out

    md_out = format_markdown_scorecard(sample_scorecard)
    assert "# FinSight Evaluation Benchmark Scorecard — run_test_001" in md_out
    assert "gemini-embedding-2" in md_out
    assert "gemini-3.8-flash" in md_out
    assert "Quality Gates Passed" in md_out


def test_scorecard_file_saving(tmp_path, sample_scorecard):
    """Verify saving JSON and Markdown scorecard artifacts to filesystem."""
    json_path, md_path = save_scorecard_artifacts(sample_scorecard, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    assert "run_test_001" in json_path.read_text(encoding="utf-8")
    assert "run_test_001" in md_path.read_text(encoding="utf-8")


def test_compare_scorecards_parity_and_diffs(sample_scorecard):
    """Verify baseline comparison detects parity, improvements, regressions, and config mismatch."""
    base_dict = sample_scorecard.model_dump()
    target_dict = sample_scorecard.model_dump()

    # 1. Exact parity
    rep = compare_scorecards(base_dict, target_dict)
    assert rep.is_configuration_compatible is True
    assert "PARITY" in rep.summary

    # 2. Regression detection
    target_dict_regressed = sample_scorecard.model_dump()
    target_dict_regressed["metrics"]["retrieval_recall_at_5"]["value"] = 0.80
    target_dict_regressed["failures"] = [{"case_id": "ret_001", "category": "retrieval", "reason": "Missed"}]
    rep_reg = compare_scorecards(base_dict, target_dict_regressed)
    assert "REGRESSION DETECTED" in rep_reg.summary
    assert "retrieval_recall_at_5" in rep_reg.regressed_metrics
    assert len(rep_reg.new_failures) == 1

    # 3. Incompatible configuration warning
    target_dict_diff_model = sample_scorecard.model_dump()
    target_dict_diff_model["configuration"]["embedding_model"] = "other-embedding-v1"
    rep_incompat = compare_scorecards(base_dict, target_dict_diff_model)
    assert rep_incompat.is_configuration_compatible is False
    assert any("Embedding model mismatch" in w for w in rep_incompat.compatibility_warnings)
