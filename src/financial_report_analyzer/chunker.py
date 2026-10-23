"""Sentence-aware, page-scoped chunking for financial reports."""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass

from financial_report_analyzer.schemas import PageRecord, ChunkRecord

logger = logging.getLogger(__name__)

# Default heuristic token targets (1 token ~= 4 characters for English financial text)
DEFAULT_TARGET_TOKENS = 850
DEFAULT_MAX_TOKENS = 1000
DEFAULT_OVERLAP_TOKENS = 120
DEFAULT_MIN_CHUNK_CHARS = 20


@dataclass(frozen=True)
class ChunkerConfig:
    """Configuration container for document chunking."""

    target_tokens: int = DEFAULT_TARGET_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS


def estimate_tokens(text: str) -> int:
    """Deterministically estimate the token count for a text string.

    NOTE: This is a local, deterministic approximation (~4 characters per token
    with a 1-token minimum for non-empty text). It avoids network latency and
    API cost during chunk boundary selection. It is not an exact Gemini tokenizer count.
    """
    if not text or not text.strip():
        return 0
    return max(1, (len(text.strip()) + 3) // 4)


# Known abbreviations that should not trigger sentence splits
COMMON_ABBREVIATIONS = {
    "inc.",
    "corp.",
    "ltd.",
    "co.",
    "vs.",
    "e.g.",
    "i.e.",
    "no.",
    "u.s.",
    "u.k.",
    "approx.",
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "oct.",
    "nov.",
    "dec.",
}


def split_into_sentence_units(paragraph: str) -> List[str]:
    """Split a paragraph into sentence units without breaking decimals, financial numbers, or common abbreviations."""
    if not paragraph or not paragraph.strip():
        return []

    # If the paragraph is formatted as table rows (multiple lines with digits/columns), keep lines as units
    lines = paragraph.split("\n")
    if len(lines) > 1 and any(re.search(r"\d", line) for line in lines):
        # Table or structured item listing: treat individual lines as indivisible units
        return [line.strip() for line in lines if line.strip()]

    # Sentence boundary regex:
    # Requires ending punctuation (. ! ?), followed by space, followed by capital letter, digit, currency, or open paren.
    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$€£¥(–—-])")

    raw_sentences = sentence_pattern.split(paragraph)
    merged: List[str] = []

    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # If the previous segment ended with an abbreviation, merge back
        if merged:
            last_tokens = merged[-1].split()
            last_word = last_tokens[-1].lower() if last_tokens else ""
            if last_word in COMMON_ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {s_clean}"
                continue
        merged.append(s_clean)

    return merged


def split_long_unit(unit: str, max_chars: int) -> List[str]:
    """Safely partition an unusually long unbroken string or sentence into pieces."""
    if len(unit) <= max_chars:
        return [unit]

    # Attempt word boundary partition
    words = unit.split()
    if len(words) > 1:
        pieces: List[str] = []
        current: List[str] = []
        curr_len = 0
        for w in words:
            w_len = len(w) + 1
            if curr_len + w_len > max_chars and current:
                pieces.append(" ".join(current))
                current = [w]
                curr_len = len(w)
            else:
                current.append(w)
                curr_len += w_len
        if current:
            pieces.append(" ".join(current))
        return pieces

    # Fallback for unbroken text without whitespace (e.g. repeated long characters)
    return [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)]


def chunk_single_page(
    page: PageRecord,
    config: ChunkerConfig,
    starting_chunk_index: int,
) -> List[ChunkRecord]:
    """Split a single PageRecord into page-scoped ChunkRecords."""
    raw_text = page.text.strip() if page.text else ""
    if len(raw_text) < config.min_chunk_chars:
        return []

    page_tokens = estimate_tokens(raw_text)
    # If entire page fits within target token size, return as a single clean chunk
    if page_tokens <= config.target_tokens:
        chunk_id = f"{page.document_id}_p{page.page_number}_c0"
        return [
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source_filename=page.source_filename,
                page_numbers=[page.page_number],
                section_title=None,
                chunk_index=starting_chunk_index,
                text=raw_text,
                embedding=None,
            )
        ]

    # Otherwise, split page into hierarchical units (paragraphs -> sentences/lines)
    paragraphs = raw_text.split("\n\n")
    max_unit_chars = config.target_tokens * 4

    units: List[str] = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        p_units = split_into_sentence_units(p_clean)
        for u in p_units:
            if len(u) > max_unit_chars:
                units.extend(split_long_unit(u, max_unit_chars))
            else:
                units.append(u)

    if not units:
        return []

    # Greedily pack units into chunks with target_tokens and overlap_tokens
    chunks: List[ChunkRecord] = []
    unit_count = len(units)
    start_idx = 0
    local_chunk_idx = 0

    while start_idx < unit_count:
        current_units: List[str] = []
        current_tokens = 0
        end_idx = start_idx

        while end_idx < unit_count:
            next_unit = units[end_idx]
            next_tokens = estimate_tokens(next_unit)

            if current_units and (current_tokens + next_tokens > config.target_tokens):
                # Target exceeded; check if adding next unit remains below max_tokens
                if current_tokens + next_tokens <= config.max_tokens:
                    current_units.append(next_unit)
                    current_tokens += next_tokens
                    end_idx += 1
                break

            current_units.append(next_unit)
            current_tokens += next_tokens
            end_idx += 1

        if not current_units:
            # Force advance at least one unit to prevent infinite loop
            current_units.append(units[start_idx])
            end_idx = start_idx + 1

        chunk_text = "\n\n".join(current_units).strip()
        chunk_id = f"{page.document_id}_p{page.page_number}_c{local_chunk_idx}"

        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source_filename=page.source_filename,
                page_numbers=[page.page_number],
                section_title=None,
                chunk_index=starting_chunk_index + local_chunk_idx,
                text=chunk_text,
                embedding=None,
            )
        )
        local_chunk_idx += 1

        if end_idx >= unit_count:
            break

        # Calculate overlap for the next chunk: pick suffix of current units whose tokens <= overlap_tokens
        overlap_accum = 0
        overlap_units_count = 0
        for i in range(end_idx - 1, start_idx, -1):
            u_tokens = estimate_tokens(units[i])
            if overlap_accum + u_tokens > config.overlap_tokens:
                break
            overlap_accum += u_tokens
            overlap_units_count += 1

        if overlap_units_count > 0:
            next_start = end_idx - overlap_units_count
            # Ensure progress: next_start must be strictly greater than start_idx
            start_idx = max(start_idx + 1, next_start)
        else:
            start_idx = end_idx

    return chunks


def chunk_pages(
    pages: List[PageRecord],
    config: Optional[ChunkerConfig] = None,
) -> List[ChunkRecord]:
    """Partition extracted PageRecords into page-scoped, sentence-aware ChunkRecords.

    Args:
        pages: List of PageRecord objects extracted in Phase 2.
        config: Chunker configuration parameters.

    Returns:
        List of ChunkRecord objects with preserved page numbers and deterministic chunk IDs.
    """
    chunker_cfg = config or ChunkerConfig()
    all_chunks: List[ChunkRecord] = []
    global_chunk_idx = 0

    for page in pages:
        page_chunks = chunk_single_page(
            page=page,
            config=chunker_cfg,
            starting_chunk_index=global_chunk_idx,
        )
        all_chunks.extend(page_chunks)
        global_chunk_idx += len(page_chunks)

    logger.info(
        "Chunked %d pages into %d chunks (target=%d, overlap=%d)",
        len(pages),
        len(all_chunks),
        chunker_cfg.target_tokens,
        chunker_cfg.overlap_tokens,
    )
    return all_chunks
