"""Baseline comparison and regression detection between FinSight evaluation scorecards."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class ComparisonReport(BaseModel):
    """Structured diff and regression report between two benchmark scorecards."""

    baseline_run_id: str
    target_run_id: str
    is_configuration_compatible: bool
    compatibility_warnings: List[str] = Field(default_factory=list)
    improved_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    regressed_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    unchanged_metrics: List[str] = Field(default_factory=list)
    new_failures: List[Dict[str, str]] = Field(default_factory=list)
    fixed_failures: List[Dict[str, str]] = Field(default_factory=list)
    gate_status_changes: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    summary: str = ""


def compare_scorecards(
    baseline_data: Dict[str, Any],
    target_data: Dict[str, Any],
) -> ComparisonReport:
    """Compare a target scorecard against a baseline scorecard and identify regressions."""
    baseline_run_id = baseline_data.get("run_id", "unknown_baseline")
    target_run_id = target_data.get("run_id", "unknown_target")

    warnings: List[str] = []
    is_compatible = True

    # Check configuration compatibility
    base_cfg = baseline_data.get("configuration", {})
    targ_cfg = target_data.get("configuration", {})

    base_dver = baseline_data.get("dataset_version")
    targ_dver = target_data.get("dataset_version")
    if base_dver != targ_dver:
        is_compatible = False
        warnings.append(f"Dataset version mismatch: baseline '{base_dver}' vs target '{targ_dver}'.")

    if base_cfg.get("embedding_model") != targ_cfg.get("embedding_model"):
        is_compatible = False
        warnings.append(
            f"Embedding model mismatch: '{base_cfg.get('embedding_model')}' vs '{targ_cfg.get('embedding_model')}'.",
        )

    if base_cfg.get("generation_model") != targ_cfg.get("generation_model"):
        warnings.append(
            f"Generation model changed: '{base_cfg.get('generation_model')}' vs '{targ_cfg.get('generation_model')}'.",
        )

    if base_cfg.get("retrieval_top_k") != targ_cfg.get("retrieval_top_k"):
        warnings.append(
            f"Retrieval top-k changed: '{base_cfg.get('retrieval_top_k')}' vs '{targ_cfg.get('retrieval_top_k')}'.",
        )

    # Compare metrics
    base_metrics = baseline_data.get("metrics", {})
    targ_metrics = target_data.get("metrics", {})

    improved: Dict[str, Dict[str, float]] = {}
    regressed: Dict[str, Dict[str, float]] = {}
    unchanged: List[str] = []

    all_metric_keys = set(base_metrics.keys()) | set(targ_metrics.keys())
    for k in sorted(all_metric_keys):
        b_val = base_metrics.get(k, {}).get("value")
        t_val = targ_metrics.get(k, {}).get("value")

        if b_val is None or t_val is None:
            continue

        diff = round(t_val - b_val, 4)
        if diff > 0.0001:
            improved[k] = {"old": b_val, "new": t_val, "delta": diff}
        elif diff < -0.0001:
            regressed[k] = {"old": b_val, "new": t_val, "delta": diff}
        else:
            unchanged.append(k)

    # Compare failure lists
    base_failures = {f.get("case_id"): f for f in baseline_data.get("failures", []) if "case_id" in f}
    targ_failures = {f.get("case_id"): f for f in target_data.get("failures", []) if "case_id" in f}

    new_failures: List[Dict[str, str]] = []
    for cid, fail_info in targ_failures.items():
        if cid not in base_failures:
            new_failures.append({
                "case_id": cid,
                "category": fail_info.get("category", "unknown"),
                "reason": fail_info.get("reason", "unknown failure"),
            })

    fixed_failures: List[Dict[str, str]] = []
    for cid, fail_info in base_failures.items():
        if cid not in targ_failures:
            fixed_failures.append({
                "case_id": cid,
                "category": fail_info.get("category", "unknown"),
                "prior_reason": fail_info.get("reason", "previously failed"),
            })

    # Compare gates
    base_gates = baseline_data.get("quality_gates", {}).get("results", {})
    targ_gates = target_data.get("quality_gates", {}).get("results", {})
    gate_changes: Dict[str, Dict[str, str]] = {}

    for gname in set(base_gates.keys()) | set(targ_gates.keys()):
        b_status = base_gates.get(gname, {}).get("status", "N/A")
        t_status = targ_gates.get(gname, {}).get("status", "N/A")
        if b_status != t_status:
            gate_changes[gname] = {"baseline": b_status, "target": t_status}

    # Summary
    if regressed or new_failures:
        summary_verdict = f"REGRESSION DETECTED: {len(regressed)} regressed metric(s), {len(new_failures)} new failure(s)."
    elif improved or fixed_failures:
        summary_verdict = f"IMPROVEMENT: {len(improved)} improved metric(s), {len(fixed_failures)} fixed failure(s)."
    else:
        summary_verdict = "PARITY: Performance matched baseline across all metrics."

    report = ComparisonReport(
        baseline_run_id=baseline_run_id,
        target_run_id=target_run_id,
        is_configuration_compatible=is_compatible,
        compatibility_warnings=warnings,
        improved_metrics=improved,
        regressed_metrics=regressed,
        unchanged_metrics=unchanged,
        new_failures=new_failures,
        fixed_failures=fixed_failures,
        gate_status_changes=gate_changes,
        summary=summary_verdict,
    )
    return report


def load_scorecard_file(filepath: Path) -> Dict[str, Any]:
    """Load JSON scorecard from filesystem."""
    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(f"Scorecard file not found: {target_path}")
    return json.loads(target_path.read_text(encoding="utf-8"))
