"""Tests verifying gold evaluation dataset integrity, schema validation, and error rejection."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from evals.dataset_schemas import (
    DatasetManifest,
    QAGoldCase,
    RetrievalGoldCase,
    MetricExtractionGoldCase,
    RiskOutlookGoldCase,
    CalculationGoldCase,
    AbstentionGoldCase,
)

DATASET_DIR = Path(__file__).parent.parent / "evals" / "dataset"


def test_manifest_schema_and_documents():
    """Verify manifest.json adheres to schema and references valid synthetic documents."""
    manifest_path = DATASET_DIR / "manifest.json"
    assert manifest_path.exists(), "Manifest file missing"

    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.dataset_version == "1.0.0"
    assert len(manifest.documents) >= 3

    doc_ids = [d.doc_id for d in manifest.documents]
    assert "doc_synth_apex_10k" in doc_ids
    assert "doc_synth_nexawave_10q" in doc_ids
    assert "doc_synth_solaria_annual" in doc_ids

    # Target counts check
    targets = manifest.target_case_counts
    assert targets.get("qa_cases", 0) >= 30
    assert targets.get("retrieval_cases", 0) >= 20
    assert targets.get("financial_extraction_cases", 0) >= 20
    assert targets.get("risk_outlook_cases", 0) >= 10
    assert targets.get("calculation_cases", 0) >= 15
    assert targets.get("abstention_cases", 0) >= 10


def test_qa_cases_schema_and_uniqueness():
    """Verify all QA gold cases pass schema validation and have unique IDs."""
    qa_path = DATASET_DIR / "qa_cases.jsonl"
    assert qa_path.exists()

    seen_ids = set()
    cases = []
    for line in qa_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = QAGoldCase.model_validate_json(line)
        assert case.case_id not in seen_ids, f"Duplicate case ID: {case.case_id}"
        seen_ids.add(case.case_id)
        assert case.expected_pages and all(p >= 1 for p in case.expected_pages)
        cases.append(case)

    assert len(cases) >= 30


def test_retrieval_cases_schema_and_markers():
    """Verify all retrieval cases have valid evidence markers and valid page numbers."""
    ret_path = DATASET_DIR / "retrieval_cases.jsonl"
    assert ret_path.exists()

    seen_ids = set()
    cases = []
    for line in ret_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = RetrievalGoldCase.model_validate_json(line)
        assert case.case_id not in seen_ids, f"Duplicate case ID: {case.case_id}"
        seen_ids.add(case.case_id)
        assert len(case.expected_evidence_markers) >= 1
        assert all(p >= 1 for p in case.expected_pages)
        cases.append(case)

    assert len(cases) >= 20


def test_financial_extraction_cases_schema():
    """Verify financial metric extraction cases pass schema validation."""
    ext_path = DATASET_DIR / "financial_extraction_cases.jsonl"
    assert ext_path.exists()

    seen_ids = set()
    cases = []
    for line in ext_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = MetricExtractionGoldCase.model_validate_json(line)
        assert case.case_id not in seen_ids, f"Duplicate case ID: {case.case_id}"
        seen_ids.add(case.case_id)
        assert case.expected_page >= 1
        cases.append(case)

    assert len(cases) >= 20


def test_risk_outlook_cases_schema():
    """Verify risk and outlook extraction cases."""
    ro_path = DATASET_DIR / "risk_outlook_cases.jsonl"
    assert ro_path.exists()

    seen_ids = set()
    cases = []
    for line in ro_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = RiskOutlookGoldCase.model_validate_json(line)
        assert case.case_id not in seen_ids
        seen_ids.add(case.case_id)
        assert all(p >= 1 for p in case.expected_pages)
        cases.append(case)

    assert len(cases) >= 10


def test_calculation_cases_schema():
    """Verify deterministic calculation cases."""
    calc_path = DATASET_DIR / "calculation_cases.jsonl"
    assert calc_path.exists()

    seen_ids = set()
    cases = []
    for line in calc_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = CalculationGoldCase.model_validate_json(line)
        assert case.case_id not in seen_ids
        seen_ids.add(case.case_id)
        cases.append(case)

    assert len(cases) >= 15


def test_abstention_cases_schema():
    """Verify abstention cases."""
    abs_path = DATASET_DIR / "abstention_cases.jsonl"
    assert abs_path.exists()

    seen_ids = set()
    cases = []
    for line in abs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = AbstentionGoldCase.model_validate_json(line)
        assert case.case_id not in seen_ids
        seen_ids.add(case.case_id)
        cases.append(case)

    assert len(cases) >= 10


def test_invalid_page_number_rejected():
    """Verify schema rejects invalid page numbers (< 1)."""
    with pytest.raises(ValidationError):
        QAGoldCase(
            case_id="qa_invalid_page",
            dataset_version="1.0.0",
            test_category="test",
            doc_id="doc_test",
            question="What is the revenue?",
            expected_answer_summary="Revenue is 100",
            expected_pages=[0],  # Invalid page 0
        )
