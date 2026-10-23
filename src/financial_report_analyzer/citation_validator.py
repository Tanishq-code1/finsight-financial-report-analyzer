"""Citation validation and verification for source-grounded financial Q&A."""

import re
import logging
from typing import List, Sequence, Union, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field

from financial_report_analyzer.schemas import Citation, ChunkRecord, RetrievalResult

logger = logging.getLogger(__name__)

# Regex for numbers including currency symbols, commas, decimals, percentages, and parenthetical negatives
NUMBER_PATTERN = re.compile(r"\(?[\$€£¥]?\d+(?:,\d{3})*(?:\.\d+)?%?\)?")


def normalize_text_for_matching(text: str) -> str:
    """Normalize text by standardizing unicode characters, quotes, dashes, and whitespace."""
    if not text:
        return ""

    # Normalize typographic quotes and apostrophes
    s = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("`", "'").replace("«", '"').replace("»", '"')

    # Normalize en/em dashes and minus signs to standard hyphen
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # Normalize non-breaking spaces and collapse whitespace
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def extract_key_numbers(text: str) -> List[str]:
    """Extract numeric tokens (values, percentages, currency amounts) for precision checking."""
    matches = NUMBER_PATTERN.findall(text)
    # Normalize numbers: strip currency signs, commas, parenthetical negatives for comparison
    normalized_numbers = []
    for m in matches:
        clean = m.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").replace(",", "")
        clean = clean.strip("()")
        if clean:
            normalized_numbers.append(clean)
    return normalized_numbers


def quote_matches_chunk_text(quote: str, chunk_text: str) -> Tuple[bool, Optional[str]]:
    """Determine whether a cited quote accurately reflects the chunk text without hallucinations.

    Returns:
        (is_match, failure_reason)
    """
    clean_quote = quote.strip().strip('"\'`')
    if not clean_quote:
        return False, "Citation quote is empty or whitespace only."

    # 1. Direct exact substring match
    if clean_quote in chunk_text:
        return True, None

    # 2. Case-insensitive substring match
    if clean_quote.lower() in chunk_text.lower():
        return True, None

    # 3. Normalized whitespace and punctuation match
    norm_quote = normalize_text_for_matching(clean_quote)
    norm_chunk = normalize_text_for_matching(chunk_text)

    if norm_quote in norm_chunk or norm_quote.lower() in norm_chunk.lower():
        return True, None

    # 4. Critical financial check: All numbers in the quote must exist in the chunk text
    quote_numbers = extract_key_numbers(clean_quote)
    chunk_numbers = extract_key_numbers(chunk_text)
    chunk_num_set = set(chunk_numbers)

    for num in quote_numbers:
        if num not in chunk_num_set:
            return False, f"Quote contains figure '{num}' not found in the source chunk."

    # 5. Ellipsis-separated phrase match (e.g. "Revenue grew ... to $500M")
    if "..." in clean_quote or "…" in clean_quote:
        parts = [p.strip() for p in re.split(r"\.\.\.|…", clean_quote) if p.strip()]
        if parts:
            search_start = 0
            all_parts_found = True
            norm_chunk_lower = norm_chunk.lower()
            for part in parts:
                norm_part = normalize_text_for_matching(part).lower()
                idx = norm_chunk_lower.find(norm_part, search_start)
                if idx == -1:
                    all_parts_found = False
                    break
                search_start = idx + len(norm_part)
            if all_parts_found:
                return True, None

    # 6. Sliding window token overlap for slight phrase paraphrase (at least 85% word overlap)
    quote_tokens = [w.lower() for w in re.findall(r"\b\w+\b", norm_quote)]
    chunk_tokens = [w.lower() for w in re.findall(r"\b\w+\b", norm_chunk)]

    if len(quote_tokens) >= 5 and chunk_tokens:
        window_size = len(quote_tokens)
        chunk_token_set = set(chunk_tokens)
        # Check if high percentage of unique quote tokens exist in the chunk
        common_tokens = sum(1 for t in quote_tokens if t in chunk_token_set)
        token_ratio = common_tokens / len(quote_tokens)

        # Sliding window check for localized sequence
        max_seq_ratio = 0.0
        for i in range(max(1, len(chunk_tokens) - window_size + 1)):
            window = set(chunk_tokens[i : i + window_size])
            overlap = sum(1 for t in quote_tokens if t in window)
            ratio = overlap / len(quote_tokens)
            if ratio > max_seq_ratio:
                max_seq_ratio = ratio

        if token_ratio >= 0.85 and max_seq_ratio >= 0.80:
            return True, None

    return False, f"Quoted text '{clean_quote[:60]}...' was not found in the source passage."


@dataclass
class CitationValidationItem:
    """Validation verdict for an individual Citation."""

    citation: Citation
    is_valid: bool
    reason: Optional[str] = None
    matched_chunk: Optional[Union[RetrievalResult, ChunkRecord]] = None
    normalized_quote: Optional[str] = None


@dataclass
class CitationValidationReport:
    """Summary of citation validation across all citations in an answer."""

    items: List[CitationValidationItem] = field(default_factory=list)
    valid_citations: List[Citation] = field(default_factory=list)
    invalid_citations: List[Citation] = field(default_factory=list)
    is_fully_valid: bool = True
    total_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    verification_rate: float = 1.0


class CitationValidator:
    """Validates citations against retrieved passages to ensure source grounding."""

    def __init__(self, strict_page_check: bool = True) -> None:
        self.strict_page_check = strict_page_check

    def validate_citations(
        self,
        citations: List[Citation],
        retrieved_chunks: Sequence[Union[RetrievalResult, ChunkRecord]],
    ) -> CitationValidationReport:
        """Validate every citation against the retrieved chunks."""
        if not citations:
            return CitationValidationReport(
                items=[],
                valid_citations=[],
                invalid_citations=[],
                is_fully_valid=True,
                total_count=0,
                valid_count=0,
                invalid_count=0,
                verification_rate=1.0,
            )

        # Build lookup table of chunks by ID
        chunk_map: Dict[str, Union[RetrievalResult, ChunkRecord]] = {}
        for c in retrieved_chunks:
            chunk_map[c.chunk_id] = c

        items: List[CitationValidationItem] = []
        valid_citations: List[Citation] = []
        invalid_citations: List[Citation] = []

        for cit in citations:
            item = self._validate_single_citation(cit, chunk_map)
            items.append(item)
            if item.is_valid:
                valid_citations.append(item.citation)
            else:
                invalid_citations.append(item.citation)

        total_count = len(citations)
        valid_count = len(valid_citations)
        invalid_count = len(invalid_citations)
        verification_rate = (valid_count / total_count) if total_count > 0 else 1.0
        is_fully_valid = invalid_count == 0

        return CitationValidationReport(
            items=items,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            is_fully_valid=is_fully_valid,
            total_count=total_count,
            valid_count=valid_count,
            invalid_count=invalid_count,
            verification_rate=round(verification_rate, 4),
        )

    def _validate_single_citation(
        self,
        citation: Citation,
        chunk_map: Dict[str, Union[RetrievalResult, ChunkRecord]],
    ) -> CitationValidationItem:
        """Verify an individual citation against available chunk records."""
        # 1. Verify Chunk ID
        chunk = chunk_map.get(citation.chunk_id)
        if chunk is None:
            return CitationValidationItem(
                citation=citation,
                is_valid=False,
                reason=f"Cited chunk ID '{citation.chunk_id}' was not in the retrieved passages.",
            )

        # 2. Verify Page Number
        allowed_pages: Set[int] = set()
        if hasattr(chunk, "page_numbers") and chunk.page_numbers:
            allowed_pages.update(chunk.page_numbers)
        if hasattr(chunk, "page_number") and chunk.page_number is not None:
            allowed_pages.add(chunk.page_number)
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            if "page_number" in chunk.metadata:
                try:
                    allowed_pages.add(int(chunk.metadata["page_number"]))
                except (ValueError, TypeError):
                    pass
            if "page_numbers" in chunk.metadata and isinstance(chunk.metadata["page_numbers"], list):
                for p in chunk.metadata["page_numbers"]:
                    try:
                        allowed_pages.add(int(p))
                    except (ValueError, TypeError):
                        pass

        # If strict page check is enabled and allowed_pages is non-empty
        if self.strict_page_check and allowed_pages:
            if citation.page_number not in allowed_pages:
                return CitationValidationItem(
                    citation=citation,
                    is_valid=False,
                    reason=(
                        f"Page number {citation.page_number} does not match chunk '{citation.chunk_id}' "
                        f"pages ({sorted(list(allowed_pages))})."
                    ),
                    matched_chunk=chunk,
                )

        # 3. Verify Quote Substring
        chunk_text = getattr(chunk, "text", "")
        is_match, fail_reason = quote_matches_chunk_text(citation.quote, chunk_text)

        if not is_match:
            return CitationValidationItem(
                citation=citation,
                is_valid=False,
                reason=fail_reason,
                matched_chunk=chunk,
            )

        # 4. Auto-backfill source_filename if missing or blank
        enriched_citation = citation
        if not citation.source_filename and getattr(chunk, "source_filename", None):
            enriched_citation = Citation(
                chunk_id=citation.chunk_id,
                page_number=citation.page_number,
                quote=citation.quote,
                source_filename=chunk.source_filename,
            )

        return CitationValidationItem(
            citation=enriched_citation,
            is_valid=True,
            matched_chunk=chunk,
            normalized_quote=normalize_text_for_matching(citation.quote),
        )

    def filter_and_annotate(
        self,
        citations: List[Citation],
        retrieved_chunks: Sequence[Union[RetrievalResult, ChunkRecord]],
    ) -> Tuple[List[Citation], List[str]]:
        """Filter out invalid citations and return valid citations plus descriptive limitation notes."""
        report = self.validate_citations(citations, retrieved_chunks)
        limitations: List[str] = []

        if report.invalid_count > 0:
            limitations.append(
                f"{report.invalid_count} of {report.total_count} citation(s) could not be verified against source report passages and were excluded."
            )
            for item in report.items:
                if not item.is_valid and item.reason:
                    logger.warning("Excluded citation for chunk '%s': %s", item.citation.chunk_id, item.reason)

        return report.valid_citations, limitations
