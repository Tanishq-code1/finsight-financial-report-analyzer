"""Data contracts and Pydantic schemas for Financial Report Analyzer."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    """Extracted text and metadata for a single PDF page."""

    document_id: str = Field(description="Unique identifier for the uploaded report session")
    source_filename: str = Field(description="Original filename of the uploaded report")
    page_number: int = Field(ge=1, description="1-based displayed page number")
    text: str = Field(description="Extracted raw text from the page")
    char_count: int = Field(ge=0, description="Number of characters extracted from page")
    has_meaningful_text: bool = Field(description="Whether page has sufficient selectable text")


class ChunkRecord(BaseModel):
    """Processed text segment with provenance metadata for retrieval."""

    chunk_id: str = Field(description="Unique deterministic chunk identifier")
    document_id: str = Field(description="Document ID this chunk belongs to")
    source_filename: str = Field(description="Original document filename")
    page_numbers: List[int] = Field(description="1-based page numbers spanned by this chunk")
    section_title: Optional[str] = Field(default=None, description="Inferred or detected section heading")
    chunk_index: int = Field(ge=0, description="Sequence index of the chunk in document")
    text: str = Field(description="Normalized chunk text content")
    embedding: Optional[List[float]] = Field(default=None, description="Vector representation")


class Citation(BaseModel):
    """Verifiable citation linking generated claims back to retrieved source passages."""

    chunk_id: str = Field(description="ID of the retrieved source chunk")
    page_number: int = Field(ge=1, description="1-based page number where the evidence appears")
    quote: str = Field(description="Direct verbatim or normalized quote from the source chunk")
    source_filename: str = Field(default="", description="Source document name")


class RAGAnswer(BaseModel):
    """Ground-truthed question-answering response with citations and sufficiency flag."""

    answer: str = Field(description="Clear, grounded response to the user question")
    citations: List[Citation] = Field(default_factory=list, description="Validated citations")
    insufficient_evidence: bool = Field(
        default=False,
        description="True if report context lacks sufficient evidence to answer truthfully",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Noted ambiguities, missing periods, or data gaps",
    )
    abstention_reason: Optional[str] = Field(
        default=None,
        description="Explicit explanation when report does not contain sufficient evidence",
    )


class FinancialMetric(BaseModel):
    """Structured financial key performance indicator."""

    name: str = Field(description="Metric name, e.g., Revenue, Operating Income, Net Income")
    value: Optional[float | str] = Field(
        default=None,
        description="Reported or computed value (None if not found)",
    )
    currency: Optional[str] = Field(default=None, description="Currency code (e.g. USD, EUR, INR)")
    unit_scale: Optional[str] = Field(
        default=None,
        description="Scale (units, thousands, millions, billions, percent)",
    )
    period_label: Optional[str] = Field(
        default=None,
        description="Fiscal period label (e.g. FY2024, Q3 FY2025)",
    )
    metric_type: Literal["reported", "calculated"] = Field(
        default="reported",
        description="Whether number was stated directly or calculated",
    )
    source_page: Optional[int] = Field(default=None, ge=1, description="Source page number")
    source_quote: Optional[str] = Field(default=None, description="Exact excerpt supporting figure")
    calculation_method: Optional[str] = Field(
        default=None,
        description="Formula used if metric is calculated",
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Caveats regarding restatements or GAAP/non-GAAP differences",
    )
    status: Literal["found", "not_found", "needs_review"] = Field(
        default="found",
        description="Extraction status indicating whether metric was verified",
    )
    normalized_name: Optional[str] = Field(default=None, description="Standardized metric key, e.g. revenue, net_income")
    numeric_value: Optional[float] = Field(default=None, description="Clean numeric float for calculation")
    prior_period_value: Optional[float | str] = Field(default=None, description="Prior year value reported in filing")
    prior_period_label: Optional[str] = Field(default=None, description="Prior fiscal period label")
    yoy_absolute_change: Optional[float] = Field(default=None, description="Python-calculated absolute YoY change")
    yoy_percentage_change: Optional[float] = Field(default=None, description="Python-calculated percentage YoY change")
    yoy_calculation_note: Optional[str] = Field(default=None, description="Details or constraints of YoY calculation")
    is_verified: bool = Field(default=True, description="Whether citation was verified against source text")
    has_conflict: bool = Field(default=False, description="Whether conflicting values were reported in filing")
    conflicting_values: List[str] = Field(default_factory=list, description="List of conflicting reported values")


class RiskItem(BaseModel):
    """Itemized risk disclosed in the filing."""

    risk_title: str = Field(description="Concise descriptive title of the risk")
    description: str = Field(description="Company disclosed details of the risk")
    risk_category: Optional[str] = Field(
        default=None,
        description="Category, e.g. Operational, Regulatory, Macroeconomic, Cyber",
    )
    company_stated_mitigation: Optional[str] = Field(
        default=None,
        description="Mitigation actions explicitly stated by company, if any",
    )
    severity: Optional[str] = Field(
        default=None,
        description="Reported severity if stated by company; do not invent",
    )
    source_pages: List[int] = Field(default_factory=list, description="Source page numbers")
    source_quotes: List[str] = Field(default_factory=list, description="Exact supporting quotes")
    impact_type: Optional[str] = Field(default=None, description="Quantified, qualitative, or unspecified")
    is_verified: bool = Field(default=True, description="Whether source citation was verified against page text")
    source_heading: Optional[str] = Field(default=None, description="Section heading in filing")


class OutlookItem(BaseModel):
    """Management guidance or forward-looking statement."""

    statement: str = Field(description="Disclosed forward-looking statement or guidance")
    period: Optional[str] = Field(default=None, description="Target period (e.g., FY2026, Q1 2026)")
    statement_type: Literal["guidance", "management_view", "risk_or_uncertainty", "other"] = Field(
        default="management_view",
        description="Classification of forward-looking commentary",
    )
    source_pages: List[int] = Field(default_factory=list, description="Source page numbers")
    source_quotes: List[str] = Field(default_factory=list, description="Supporting excerpts")
    time_horizon: Optional[str] = Field(default=None, description="Time horizon or target timeframe")
    direction: Optional[str] = Field(default=None, description="positive, negative, mixed, or undetermined")
    target_metric: Optional[str] = Field(default=None, description="Targeted financial metric if stated")
    target_value: Optional[str] = Field(default=None, description="Target numerical amount or range if stated")
    is_verified: bool = Field(default=True, description="Whether citation was verified against source page text")


class ReportAnalysis(BaseModel):
    """Complete structured financial report analysis."""

    report_title: Optional[str] = Field(default=None, description="Report title if detected")
    company_name: Optional[str] = Field(default=None, description="Company name if detected")
    report_type: Optional[str] = Field(default=None, description="Filing type (e.g. 10-K, 10-Q, Annual Report)")
    fiscal_period: Optional[str] = Field(default=None, description="Primary fiscal period covered")
    executive_summary: str = Field(description="Concise 150-300 word executive overview")
    key_takeaways: List[str] = Field(default_factory=list, description="Top strategic takeaways")
    metrics: List[FinancialMetric] = Field(default_factory=list, description="Extracted financial KPIs")
    risks: List[RiskItem] = Field(default_factory=list, description="Disclosed risk factors")
    outlook: List[OutlookItem] = Field(default_factory=list, description="Forward-looking statements")
    limitations: List[str] = Field(default_factory=list, description="Document processing or scope limitations")
    generated_at: str = Field(description="ISO timestamp of analysis generation")
    analysis_status: Literal["complete", "partial", "failed"] = Field(
        default="complete",
        description="Coverage status of full report analysis",
    )
    pages_analyzed: int = Field(default=0, ge=0, description="Total readable pages processed")
    total_pages: int = Field(default=0, ge=0, description="Total pages in filing")
    batches_succeeded: int = Field(default=0, ge=0, description="Number of analysis batches completed")
    batches_failed: int = Field(default=0, ge=0, description="Number of analysis batches that failed")
    data_conflicts: List[str] = Field(default_factory=list, description="Unresolved reporting discrepancies")
    verification_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Ratio of verified citations")
    model_used: Optional[str] = Field(default=None, description="Generation model that produced this analysis")
    fallback_used: bool = Field(default=False, description="Whether fallback model was invoked during extraction")


class BatchExtractionResult(BaseModel):
    """Structured extraction result returned by Gemini for a single bounded batch of pages."""

    report_title: Optional[str] = Field(default=None, description="Filing or company title if identified in batch")
    company_name: Optional[str] = Field(default=None, description="Company name if identified in batch")
    fiscal_period: Optional[str] = Field(default=None, description="Fiscal period covered if identified")
    metrics: List[FinancialMetric] = Field(default_factory=list, description="Extracted metrics with page references")
    risks: List[RiskItem] = Field(default_factory=list, description="Extracted risk disclosures")
    outlook: List[OutlookItem] = Field(default_factory=list, description="Extracted forward-looking guidance")
    executive_summary_excerpts: List[str] = Field(default_factory=list, description="Key strategic summary points")
    model_used: Optional[str] = Field(default=None, description="Model that produced this batch")
    fallback_used: bool = Field(default=False, description="Whether fallback model was used for this batch")


class IngestionWarning(BaseModel):
    """Non-fatal warning encountered during document ingestion."""

    page_number: Optional[int] = None
    warning_type: str
    message: str


class DocumentMetadata(BaseModel):
    """Summary metadata of an ingested report."""

    document_id: str
    filename: str
    file_size_bytes: int
    total_pages: int
    meaningful_pages: int
    warnings: List[IngestionWarning] = Field(default_factory=list)
    chunk_count: int = 0
    indexed: bool = False


class RetrievalResult(BaseModel):
    """Result of retrieving a single chunk from the vector store."""

    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Document ID")
    source_filename: str = Field(description="Original filename")
    page_number: int = Field(ge=1, description="1-based page number")
    text: str = Field(description="Text content of the retrieved chunk")
    distance: float = Field(description="Cosine distance from query vector (0.0 = identical)")
    similarity: float = Field(description="Cosine similarity metric (1.0 = identical)")
    chunk_index: int = Field(ge=0, description="Index of chunk in document")
    metadata: dict = Field(default_factory=dict, description="Underlying metadata dictionary")


class IndexingResult(BaseModel):
    """Summary of the vector indexing process for an ingested report."""

    document_id: str = Field(description="Document ID")
    total_pages: int = Field(ge=0, description="Total pages processed")
    total_chunks: int = Field(ge=0, description="Total chunks produced")
    total_embeddings: int = Field(ge=0, description="Total embeddings generated")
    embedding_model: str = Field(description="Name of the embedding model used")
    embedding_dimension: int = Field(ge=1, description="Dimensionality of vectors")
    collection_name: str = Field(description="Name of the isolated Chroma collection")
