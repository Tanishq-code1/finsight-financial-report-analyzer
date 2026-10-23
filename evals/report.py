"""Scorecard report generators (Terminal, JSON, Markdown) for FinSight evaluation benchmark."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class EvaluationScorecard(BaseModel):
    """Canonical benchmark results scorecard contract."""

    run_id: str
    started_at_utc: str
    completed_at_utc: str
    dataset_version: str
    git_commit: str
    app_version: str
    mode: str = "offline"  # "offline" or "live"
    configuration: Dict[str, Any] = Field(default_factory=dict)
    case_summary: Dict[str, int] = Field(default_factory=dict)
    category_counts: Dict[str, int] = Field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    quality_gates: Dict[str, Any] = Field(default_factory=dict)
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    latency_summary: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


def format_terminal_summary(scorecard: EvaluationScorecard) -> str:
    """Render a clean, human-readable terminal table summary."""
    lines = []
    lines.append("=" * 78)
    lines.append(" FINLIGHT (FinSight) EVALUATION BENCHMARK SCORECARD")
    lines.append("=" * 78)
    lines.append(f"Run ID:            {scorecard.run_id}")
    lines.append(f"Timestamp (UTC):   {scorecard.started_at_utc} -> {scorecard.completed_at_utc}")
    lines.append(f"Git Commit:        {scorecard.git_commit}")
    lines.append(f"Dataset Version:   {scorecard.dataset_version}")
    lines.append(f"Evaluation Mode:   {scorecard.mode.upper()}")
    lines.append(f"Embedding Model:   {scorecard.configuration.get('embedding_model')} ({scorecard.configuration.get('embedding_dimension')}D)")
    lines.append(f"Generation Model:  {scorecard.configuration.get('generation_model')}")
    lines.append("-" * 78)
    
    # Cases Summary
    cs = scorecard.case_summary
    lines.append(
        f"Cases: Total={cs.get('total', 0)} | Executed={cs.get('executed', 0)} | "
        f"Passed={cs.get('passed', 0)} | Failed={cs.get('failed', 0)} | Skipped={cs.get('skipped', 0)}"
    )
    lines.append("-" * 78)

    # Metrics Table
    lines.append(f"{'METRIC NAME':<34} {'SCORE':<10} {'NUM/DENOM':<14} {'GATE':<8} {'STATUS':<8}")
    lines.append("-" * 78)

    gates_dict = scorecard.quality_gates.get("results", {})
    for mname, mdata in scorecard.metrics.items():
        val = mdata.get("value", 0.0)
        num = mdata.get("numerator", 0.0)
        den = mdata.get("denominator", 0.0)
        
        # Check gate
        gate_info = gates_dict.get(mname)
        if gate_info:
            target_str = f">={gate_info.get('target', 0.0):.2f}"
            status = gate_info.get("status", "N/A")
        else:
            target_str = "—"
            status = "INFO"

        pct_or_score = f"{val * 100:.1f}%" if "accuracy" in mname or "rate" in mname or "precision" in mname or "recall" in mname or "f1" in mname else f"{val:.3f}"
        fraction_str = f"{int(num)}/{int(den)}" if den.is_integer() and num.is_integer() else f"{num:.1f}/{den:.1f}"

        lines.append(f"{mname:<34} {pct_or_score:<10} {fraction_str:<14} {target_str:<8} {status:<8}")

    lines.append("-" * 78)
    overall_status = scorecard.quality_gates.get("overall_status", "UNKNOWN")
    lines.append(f"OVERALL QUALITY GATE RESULT: {overall_status}")
    lines.append("=" * 78)

    if scorecard.failures:
        lines.append(f"FAILURES DETECTED ({len(scorecard.failures)} total):")
        for idx, f in enumerate(scorecard.failures[:10], start=1):
            lines.append(f"  {idx}. [{f.get('case_id')}] ({f.get('category')}): {f.get('reason')}")
        if len(scorecard.failures) > 10:
            lines.append(f"  ... and {len(scorecard.failures) - 10} more failures (see JSON report)")
        lines.append("-" * 78)

    if scorecard.warnings:
        lines.append("NOTICES & WARNINGS:")
        for w in scorecard.warnings:
            lines.append(f"  • {w}")
        lines.append("=" * 78)

    return "\n".join(lines)


def format_markdown_scorecard(scorecard: EvaluationScorecard) -> str:
    """Render a GitHub-flavored Markdown report."""
    md = []
    md.append(f"# FinSight Evaluation Benchmark Scorecard — {scorecard.run_id}")
    md.append("")
    md.append(f"> **Mode**: `{scorecard.mode.upper()}` | **Dataset Version**: `{scorecard.dataset_version}` | **Git Commit**: `{scorecard.git_commit}`")
    md.append(f"> **Date (UTC)**: `{scorecard.started_at_utc}` to `{scorecard.completed_at_utc}`")
    md.append("")

    overall = scorecard.quality_gates.get("overall_status", "UNKNOWN")
    if overall == "PASSED":
        md.append("> [!TIP]\n> **Quality Gates Passed**: All provisional engineering benchmark targets were successfully satisfied.")
    else:
        md.append(f"> [!WARNING]\n> **Quality Gates {overall}**: One or more provisional engineering benchmark targets failed.")
    md.append("")

    # Benchmark Configuration
    md.append("## 1. Benchmark Configuration")
    md.append("")
    md.append("| Setting | Configured Value |")
    md.append("|---|---|")
    md.append(f"| **Embedding Model** | `{scorecard.configuration.get('embedding_model')}` ({scorecard.configuration.get('embedding_dimension')}D) |")
    md.append(f"| **Generation Model** | `{scorecard.configuration.get('generation_model')}` |")
    md.append(f"| **Retrieval Top-K** | `{scorecard.configuration.get('retrieval_top_k')}` |")
    md.append(f"| **App Version** | `{scorecard.app_version}` |")
    md.append("")

    # Execution Summary
    cs = scorecard.case_summary
    md.append("## 2. Test Execution Summary")
    md.append("")
    md.append("| Category | Total | Executed | Passed | Failed | Skipped |")
    md.append("|---|---|---|---|---|---|")
    md.append(
        f"| **All Cases** | {cs.get('total', 0)} | {cs.get('executed', 0)} | "
        f"{cs.get('passed', 0)} | {cs.get('failed', 0)} | {cs.get('skipped', 0)} |"
    )
    for cat, count in scorecard.category_counts.items():
        md.append(f"| `{cat}` | {count} | {count} | — | — | — |")
    md.append("")

    # Quality Metrics & Gates Table
    md.append("## 3. Evaluated Metrics & Quality Gates")
    md.append("")
    md.append("| Metric | Measured Score | Fraction (Num / Den) | Target Gate | Status |")
    md.append("|---|---|---|---|---|")

    gates_dict = scorecard.quality_gates.get("results", {})
    for mname, mdata in scorecard.metrics.items():
        val = mdata.get("value", 0.0)
        num = mdata.get("numerator", 0.0)
        den = mdata.get("denominator", 0.0)

        gate_info = gates_dict.get(mname)
        if gate_info:
            target_str = f"&ge; {gate_info.get('target', 0.0):.2f}"
            status = f"**{gate_info.get('status')}**"
        else:
            target_str = "—"
            status = "INFO"

        pct_or_score = f"{val * 100:.1f}%" if "accuracy" in mname or "rate" in mname or "precision" in mname or "recall" in mname or "f1" in mname else f"{val:.4f}"
        fraction_str = f"{int(num)} / {int(den)}" if den.is_integer() and num.is_integer() else f"{num:.1f} / {den:.1f}"

        md.append(f"| `{mname}` | {pct_or_score} | {fraction_str} | {target_str} | {status} |")
    md.append("")

    # Failures Section
    md.append("## 4. Failure Diagnostics")
    md.append("")
    if not scorecard.failures:
        md.append("No failures detected. All evaluation assertions passed.")
    else:
        md.append(f"**Total Failures:** {len(scorecard.failures)}")
        md.append("")
        md.append("| Case ID | Category | Reason / Expected vs Actual |")
        md.append("|---|---|---|")
        for f in scorecard.failures:
            cid = f.get("case_id", "N/A")
            cat = f.get("category", "N/A")
            reason = str(f.get("reason", "")).replace("|", "\\|")
            md.append(f"| `{cid}` | `{cat}` | {reason} |")
    md.append("")

    # Notices & Disclaimers
    md.append("## 5. Methodology & Disclaimers")
    md.append("")
    for w in scorecard.warnings:
        md.append(f"- {w}")
    md.append("")
    md.append("> [!NOTE]\n> **Non-Investment Advice Notice**: Benchmark results reflect synthetic test datasets and controlled evaluations. They do not constitute investment, financial, or trading recommendations.")
    md.append("")

    return "\n".join(md)


def save_scorecard_artifacts(
    scorecard: EvaluationScorecard,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save both JSON and Markdown scorecard files to output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / f"scorecard_{scorecard.run_id}.json"
    md_path = out_path / f"scorecard_{scorecard.run_id}.md"

    json_path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(format_markdown_scorecard(scorecard), encoding="utf-8")

    return json_path, md_path
