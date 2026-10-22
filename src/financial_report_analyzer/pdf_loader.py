"""In-memory PDF report ingestion and page-level provenance extraction."""

import re
import hashlib
import logging
from typing import List, Optional
from dataclasses import dataclass
import pymupdf

from financial_report_analyzer.config import AppConfig, load_config
from financial_report_analyzer.errors import DocumentValidationError, PDFParsingError
from financial_report_analyzer.schemas import DocumentMetadata, IngestionWarning, PageRecord

logger = logging.getLogger(__name__)

# Constants for heuristic text density detection
DEFAULT_MIN_PAGE_CHARS = 50
DEFAULT_SPARSE_PAGE_CHARS = 100
DEFAULT_SPARSE_DOC_AVG_CHARS = 120


@dataclass(frozen=True)
class PDFIngestionResult:
    """Encapsulates the complete result of parsing and validating a PDF report."""

    metadata: DocumentMetadata
    pages: List[PageRecord]
    total_characters: int
    pages_with_text: int
    is_sparse: bool

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were flagged during extraction."""
        return len(self.metadata.warnings) > 0


def normalize_financial_text(raw_text: str) -> str:
    """Normalize extracted text without damaging financial figures, tables, or signs.

    Preserves:
    - Currency symbols ($, €, £, ¥, ₹, etc.)
    - Negative indicators (leading minus, parentheses around figures, e.g., (1,234.56))
    - Decimal points and percentage signs
    - Tabular line structures and column spacing
    - Paragraph boundaries (double newlines)
    """
    if not raw_text:
        return ""

    # Strip null bytes, form feeds, zero-width artifacts, and normalize soft hyphens
    text = (
        raw_text.replace("\x00", "")
        .replace("\x0c", "\n")
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\xad", "-")
    )

    # Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize non-breaking spaces to standard spaces
    text = text.replace("\u00a0", " ")

    # Process line by line to preserve row structures while pruning excessive whitespace
    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        # Convert horizontal tabs to two spaces to preserve table column alignments
        line = line.replace("\t", "  ")
        # Collapse 3 or more consecutive spaces to 2 spaces (keeps columns apart without huge gaps)
        line = re.sub(r"[ ]{3,}", "  ", line)
        cleaned_lines.append(line.rstrip())

    # Rejoin lines
    joined = "\n".join(cleaned_lines)

    # Collapse 3 or more blank lines into a double newline (standard paragraph break)
    normalized = re.sub(r"\n{3,}", "\n\n", joined)

    return normalized.strip()


def generate_document_id(pdf_bytes: bytes) -> str:
    """Generate a deterministic, secure document ID from content digest."""
    digest = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    return f"doc_{digest}"


def validate_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    max_upload_bytes: int,
) -> None:
    """Perform pre-parsing validation on filename, file size, and PDF signature."""
    if not isinstance(filename, str) or not filename.strip():
        raise DocumentValidationError(
            "Filename must be a non-empty string.",
            user_friendly_message="Invalid filename provided for uploaded document.",
        )

    # Validate file extension
    clean_filename = filename.strip()
    if not clean_filename.lower().endswith(".pdf"):
        raise DocumentValidationError(
            f"Invalid file extension: '{clean_filename}'. Only '.pdf' files are supported.",
            user_friendly_message=f"'{clean_filename}' is not a PDF file. Please upload a file ending with .pdf.",
        )

    # Validate byte payload
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise DocumentValidationError(
            "Uploaded content must be bytes.",
            user_friendly_message="Invalid file content type.",
        )

    byte_len = len(pdf_bytes)
    if byte_len == 0:
        raise DocumentValidationError(
            "Uploaded PDF file is empty (0 bytes).",
            user_friendly_message="The uploaded file is completely empty (0 bytes).",
        )

    # Enforce size limits
    if byte_len > max_upload_bytes:
        file_mb = byte_len / (1024 * 1024)
        max_mb = max_upload_bytes / (1024 * 1024)
        raise DocumentValidationError(
            f"File size ({file_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB.",
            user_friendly_message=f"File is too large ({file_mb:.1f} MB). Maximum upload size is {max_mb:.0f} MB.",
        )

    # Validate standard PDF header signature (%PDF-)
    # Per the PDF specification, %PDF- must appear within the first 1024 bytes
    header_sample = bytes(pdf_bytes[:1024])
    if b"%PDF-" not in header_sample:
        raise DocumentValidationError(
            "File header does not contain standard PDF signature ('%PDF-').",
            user_friendly_message="The uploaded file is not a valid PDF document or its header is corrupt.",
        )


def load_pdf(
    pdf_bytes: bytes,
    filename: str,
    config: Optional[AppConfig] = None,
    min_page_chars: int = DEFAULT_MIN_PAGE_CHARS,
    sparse_page_chars: int = DEFAULT_SPARSE_PAGE_CHARS,
    require_extractable_text: bool = False,
) -> PDFIngestionResult:
    """Parse a PDF document in memory, extracting text page-by-page with 1-based page provenance.

    Args:
        pdf_bytes: Raw binary content of the PDF file.
        filename: Original file name.
        config: Application configuration (defaults to load_config() if None).
        min_page_chars: Character count threshold to classify a page as having meaningful text.
        sparse_page_chars: Character count threshold below which a non-empty page triggers a warning.
        require_extractable_text: If True, raises DocumentValidationError if no text is found across all pages.

    Returns:
        PDFIngestionResult containing DocumentMetadata, PageRecord list, and extraction statistics.

    Raises:
        DocumentValidationError: If file violates format, size, encryption, or page limits.
        PDFParsingError: If PyMuPDF encounters fatal corruption while parsing the document.
    """
    app_cfg = config or load_config()

    # Pre-parse validation
    validate_pdf_bytes(
        pdf_bytes=pdf_bytes,
        filename=filename,
        max_upload_bytes=app_cfg.max_upload_bytes,
    )

    doc_id = generate_document_id(pdf_bytes)
    logger.info("Ingesting document '%s' (ID: %s, Size: %d bytes)", filename, doc_id, len(pdf_bytes))

    # Open PDF stream purely in memory
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("PyMuPDF failed to open document '%s': %s", filename, str(exc))
        raise PDFParsingError(
            f"PyMuPDF failed to parse PDF document: {str(exc)}",
            user_friendly_message="The uploaded file appears corrupted and could not be parsed as a PDF.",
        ) from exc

    try:
        # Check for password protection / encryption
        if doc.is_encrypted:
            if doc.needs_pass or doc.authenticate("") == 0:
                raise DocumentValidationError(
                    "PDF is encrypted or password-protected.",
                    user_friendly_message="The uploaded PDF is password-protected. Please remove password encryption before uploading.",
                )

        # Enforce page limits
        total_pages = doc.page_count
        if total_pages == 0:
            raise DocumentValidationError(
                "PDF document contains 0 pages.",
                user_friendly_message="The uploaded PDF contains no pages.",
            )

        if total_pages > app_cfg.max_pdf_pages:
            raise DocumentValidationError(
                f"Document has {total_pages} pages, which exceeds the limit of {app_cfg.max_pdf_pages} pages.",
                user_friendly_message=f"Document has {total_pages} pages, which exceeds the limit of {app_cfg.max_pdf_pages} pages.",
            )

        pages: List[PageRecord] = []
        warnings: List[IngestionWarning] = []
        total_extracted_characters = 0
        pages_with_text = 0
        meaningful_pages = 0

        # Extract text page by page (1-based page indices)
        for page_idx in range(total_pages):
            page_number = page_idx + 1  # Accurate 1-based page index
            try:
                page = doc[page_idx]
                raw_text = page.get_text("text") or ""
            except Exception as page_exc:
                logger.warning("Error reading page %d of '%s': %s", page_number, filename, str(page_exc))
                raw_text = ""
                warnings.append(
                    IngestionWarning(
                        page_number=page_number,
                        warning_type="page_read_error",
                        message=f"Page {page_number} could not be extracted completely: {str(page_exc)}",
                    )
                )

            cleaned_text = normalize_financial_text(raw_text)
            char_count = len(cleaned_text)
            total_extracted_characters += char_count

            has_meaningful = char_count >= min_page_chars
            if has_meaningful:
                meaningful_pages += 1

            if char_count > 0:
                pages_with_text += 1
            else:
                warnings.append(
                    IngestionWarning(
                        page_number=page_number,
                        warning_type="blank_or_unextractable_page",
                        message=(
                            f"Page {page_number} contains no extractable text. It may be blank or contain "
                            "scanned images requiring OCR (OCR is not supported in this version)."
                        ),
                    )
                )

            if 0 < char_count < sparse_page_chars:
                warnings.append(
                    IngestionWarning(
                        page_number=page_number,
                        warning_type="sparse_page",
                        message=(
                            f"Page {page_number} has unusually low text density ({char_count} characters). "
                            "It may be a cover, diagram, or partial scan."
                        ),
                    )
                )

            pages.append(
                PageRecord(
                    document_id=doc_id,
                    source_filename=filename,
                    page_number=page_number,
                    text=cleaned_text,
                    char_count=char_count,
                    has_meaningful_text=has_meaningful,
                )
            )

        # Check document-level sparsity
        is_sparse = False
        avg_chars_per_page = total_extracted_characters / total_pages if total_pages > 0 else 0

        if pages_with_text == 0:
            is_sparse = True
            if require_extractable_text:
                raise DocumentValidationError(
                    "PDF contains no extractable text across all pages.",
                    user_friendly_message=(
                        "No readable text could be extracted from this document. It appears to be an image-only or "
                        "scanned PDF, which requires OCR (OCR is not supported in this version)."
                    ),
                )
            warnings.insert(
                0,
                IngestionWarning(
                    page_number=None,
                    warning_type="no_extractable_text",
                    message=(
                        "No extractable text was found across all pages. The file appears to be a scanned or "
                        "image-only PDF; OCR is not supported in this version."
                    ),
                ),
            )
        elif avg_chars_per_page < DEFAULT_SPARSE_DOC_AVG_CHARS or (meaningful_pages / total_pages) < 0.25:
            is_sparse = True
            warnings.insert(
                0,
                IngestionWarning(
                    page_number=None,
                    warning_type="sparse_document",
                    message=(
                        f"Document has unusually low overall text density (average {avg_chars_per_page:.0f} chars/page). "
                        "Scanned portions may require OCR (not supported in this version)."
                    ),
                ),
            )

        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            file_size_bytes=len(pdf_bytes),
            total_pages=total_pages,
            meaningful_pages=meaningful_pages,
            warnings=warnings,
            chunk_count=0,
            indexed=False,
        )

        logger.info(
            "Parsed '%s': %d pages (%d meaningful, %d with text), %d characters extracted, %d warnings",
            filename,
            total_pages,
            meaningful_pages,
            pages_with_text,
            total_extracted_characters,
            len(warnings),
        )

        return PDFIngestionResult(
            metadata=metadata,
            pages=pages,
            total_characters=total_extracted_characters,
            pages_with_text=pages_with_text,
            is_sparse=is_sparse,
        )

    finally:
        doc.close()
