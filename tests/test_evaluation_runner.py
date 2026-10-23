"""Tests verifying evaluation runner execution, isolation, and quality gates."""

import pytest
from pathlib import Path
from evals.runner import BenchmarkRunner

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "evals" / "dataset"
GATES_FILE = REPO_ROOT / "evals" / "gates.json"


def test_runner_offline_execution(tmp_path):
    """Verify that BenchmarkRunner runs completely offline with 100% pass rate."""
    runner = BenchmarkRunner(
        dataset_dir=DATASET_DIR,
        gates_file=GATES_FILE,
        output_dir=tmp_path,
        mode="offline",
        top_k=5,
    )

    scorecard = runner.run_benchmark()

    assert scorecard.run_id.startswith("run_")
    assert scorecard.mode == "offline"
    assert scorecard.dataset_version == "1.0.0"
    assert scorecard.case_summary["total"] == 110
    assert scorecard.case_summary["failed"] == 0
    assert scorecard.case_summary["passed"] == 110

    # Verify gates
    qg = scorecard.quality_gates
    assert qg["overall_status"] == "PASSED"
    assert qg["results"]["retrieval_recall_at_5"]["status"] == "PASS"
    assert qg["results"]["citation_precision"]["status"] == "PASS"
    assert qg["results"]["abstention_accuracy"]["status"] == "PASS"
    assert qg["results"]["numeric_calculation_accuracy"]["status"] == "PASS"
    assert qg["results"]["export_data_fidelity"]["status"] == "PASS"


def test_runner_missing_dataset_directory(tmp_path):
    """Verify runner fails cleanly when dataset path does not exist."""
    empty_dir = tmp_path / "non_existent"
    runner = BenchmarkRunner(
        dataset_dir=empty_dir,
        gates_file=GATES_FILE,
        output_dir=tmp_path,
        mode="offline",
    )
    with pytest.raises(FileNotFoundError):
        runner.load_dataset()
