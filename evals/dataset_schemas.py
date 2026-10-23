"""Pydantic data schemas for versioned gold evaluation datasets in FinSight."""

from decimal import Decimal
from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class GoldDocument(BaseModel):
    """Metadata for a synthetic or reference document in the evaluation manifest."""

    doc_id: str = Field(description="Unique stable document identifier, e.g. doc_synth_apex_10k")
    filename: str = Field(description="Filename on disk or generated in-memory")
    company_name: str = Field(description="Synthetic company name")
    report_type: str = Field(description="Report type, e.g. 10-K, 10-Q, Annual Report")
    fiscal_period: str = Field(description="Primary fiscal period covered")
    total_pages: int = Field(ge=1, description="Total pages in the report")
    features: List[str] = Field(
        default_factory=list,
        description="Key document attributes, e.g. prompt_injection, sparse_page, contradictory_values",
    )


class DatasetManifest(BaseModel):
    """Manifest describing the gold evaluation dataset version and document inventory."""

    dataset_version: str = Field(description="Semantic version of the gold dataset, e.g. 1.0.0")
    created_at_utc: str = Field(description="ISO 8601 UTC timestamp of dataset creation")
    description: str = Field(description="Overview of the evaluation benchmark scope")
    documents: List[GoldDocument] = Field(description="Registered evaluation documents")
    target_case_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Expected case count per evaluation category",
    )


class QAGoldCase(BaseModel):
    """Evaluation case for grounded question answering."""

    case_id: str = Field(description="Unique case identifier, e.g. qa_apex_rev_001")
    dataset_version: str = Field(default="1.0.0")
    test_category: str = Field(description="Category, e.g. factual_metrics, risks, outlook, edge_cases")
    doc_id: str = Field(description="Target document identifier")
    question: str = Field(description="User question asked to the report")
    expected_answer_summary: str = Field(description="Human-readable reference answer summary")
    expected_facts: List[str] = Field(
        default_factory=list,
        description="Key factual strings or numbers that must appear in verified answer",
    )
    rejected_phrases: List[str] = Field(
        default_factory=list,
        description="Phrases indicating hallucination or successful prompt injection to reject",
    )
    expected_pages: List[int] = Field(description="1-based source pages containing supporting evidence")
    expected_quotes: List[str] = Field(
        default_factory=list,
        description="Key verbatim excerpts supporting the expected answer",
    )
    expected_abstention: bool = Field(
        default=False,
        description="Whether the system should abstain due to missing evidence",
    )
    difficulty: Literal["easy", "medium", "hard", "adversarial"] = Field(default="medium")
    notes: Optional[str] = None

    @field_validator("expected_pages")
    @classmethod
    def validate_pages(cls, pages: List[int]) -> List[int]:
        for p in pages:
            if p < 1:
                raise ValueError(f"Page numbers must be >= 1, got {p}")
        return pages


class RetrievalGoldCase(BaseModel):
    """Evaluation case for vector semantic retrieval."""

    case_id: str = Field(description="Unique case identifier, e.g. ret_apex_bs_001")
    dataset_version: str = Field(default="1.0.0")
    test_category: str = Field(description="Retrieval domain: balance_sheet, cash_flow, risks, notes, etc.")
    doc_id: str = Field(description="Target document identifier")
    query: str = Field(description="Search query string")
    expected_pages: List[int] = Field(description="Pages containing the relevant chunks")
    expected_evidence_markers: List[str] = Field(
        description="Distinct keywords or phrases that define a truly relevant chunk",
    )
    min_k: int = Field(default=5, ge=1, description="Top-k evaluation depth")
    notes: Optional[str] = None


class MetricExtractionGoldCase(BaseModel):
    """Evaluation case for structured financial KPI extraction."""

    case_id: str = Field(description="Unique case identifier, e.g. ext_apex_rev_001")
    dataset_version: str = Field(default="1.0.0")
    test_category: str = Field(description="Metric category: revenue, profitability, cash_flow, debt, etc.")
    doc_id: str = Field(description="Target document identifier")
    normalized_name: str = Field(description="Canonical metric identifier, e.g. revenue, net_income")
    expected_value: Optional[Decimal] = Field(
        default=None,
        description="Expected exact decimal value reported in filing",
    )
    expected_currency: Optional[str] = Field(default="USD")
    expected_unit_scale: Optional[str] = Field(default="millions")
    expected_period_label: Optional[str] = Field(default=None)
    expected_page: int = Field(ge=1, description="1-based page number where metric is declared")
    prior_period_value: Optional[Decimal] = Field(
        default=None,
        description="Expected prior period value reported in filing",
    )
    prior_period_label: Optional[str] = None
    is_derived: bool = Field(default=False, description="Whether metric is derived (e.g. Free Cash Flow)")
    notes: Optional[str] = None


class RiskOutlookGoldCase(BaseModel):
    """Evaluation case for disclosed risks and management outlook extraction."""

    case_id: str = Field(description="Unique case identifier, e.g. ro_apex_cyber_001")
    dataset_version: str = Field(default="1.0.0")
    test_category: Literal["risk", "outlook"] = Field(description="Item classification")
    doc_id: str = Field(description="Target document identifier")
    expected_title_or_topic: str = Field(description="Distinctive topic or title phrase")
    expected_category_or_type: Optional[str] = Field(
        default=None,
        description="Risk category (Operational, Cyber, etc.) or outlook type (guidance, management_view)",
    )
    expected_pages: List[int] = Field(description="1-based source pages")
    expected_direction: Optional[str] = Field(
        default=None,
        description="For outlook: positive, negative, mixed, or undetermined",
    )
    expected_qualifications: List[str] = Field(
        default_factory=list,
        description="Mitigations or uncertainty qualifiers that must be retained",
    )
    notes: Optional[str] = None


class CalculationGoldCase(BaseModel):
    """Evaluation case for deterministic Python Decimal financial arithmetic."""

    case_id: str = Field(description="Unique case identifier, e.g. calc_yoy_growth_001")
    dataset_version: str = Field(default="1.0.0")
    test_category: str = Field(
        description="Scenario: positive_growth, negative_growth, zero_prior, loss_to_profit, loss_reduction, scale_conversion, currency_mismatch, fcf_derivation",
    )
    metric_name: str = Field(description="Target metric name")
    current_value: Optional[Union[Decimal, float, str]] = None
    current_currency: Optional[str] = "USD"
    current_scale: Optional[str] = "millions"
    current_period: Optional[str] = "FY2024"
    prior_value: Optional[Union[Decimal, float, str]] = None
    prior_currency: Optional[str] = "USD"
    prior_scale: Optional[str] = "millions"
    prior_period: Optional[str] = "FY2023"
    expected_absolute_change: Optional[Decimal] = None
    expected_percentage_change: Optional[Decimal] = None
    expected_status: Literal["valid", "withheld", "error"] = "valid"
    expected_note_contains: Optional[str] = Field(
        default=None,
        description="Expected keyword in calculation note explaining why percentage was withheld or calculated",
    )
    notes: Optional[str] = None


class AbstentionGoldCase(BaseModel):
    """Evaluation case for unsupported, ambiguous, or adversarial queries requiring abstention."""

    case_id: str = Field(description="Unique case identifier, e.g. abs_apex_ceo_001")
    dataset_version: str = Field(default="1.0.0")
    test_category: str = Field(
        description="Reason: missing_topic, unincluded_future_period, unincluded_historical_period, external_trivia, prompt_injection, ambiguous_query, conflicting_evidence",
    )
    doc_id: str = Field(description="Target document identifier")
    question: str = Field(description="Question that should trigger abstention or qualification")
    expected_abstention: bool = Field(default=True, description="Must be True for unanswerable questions")
    expected_reason_keywords: List[str] = Field(
        default_factory=list,
        description="Key terms that must appear in abstention reason explanation",
    )
    notes: Optional[str] = None
