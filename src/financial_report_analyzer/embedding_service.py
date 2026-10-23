"""Google GenAI embedding service supporting gemini-embedding-2."""

import time
import math
import logging
from typing import List, Optional, Callable
from google import genai
from google.genai import errors as genai_errors

from financial_report_analyzer.config import AppConfig, load_config
from financial_report_analyzer.errors import ConfigurationError, EmbeddingError
from financial_report_analyzer.schemas import ChunkRecord

logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF_SEC = 1.0


def format_document_text(title: str, text: str) -> str:
    """Format document chunk text using recommended asymmetric retrieval prefix."""
    clean_title = title.strip() if title else "Document"
    clean_text = text.strip() if text else ""
    return f"title: {clean_title} | text: {clean_text}"


def format_query_text(query: str) -> str:
    """Format query text using recommended asymmetric retrieval task prefix."""
    clean_query = query.strip() if query else ""
    return f"task: question answering | query: {clean_query}"


def validate_embedding_vector(vector: List[float], expected_dim: int, context_label: str = "item") -> None:
    """Validate that returned embedding is a finite numeric vector of expected dimensionality."""
    if not isinstance(vector, (list, tuple)):
        raise EmbeddingError(f"Embedding service returned invalid vector type for {context_label}: {type(vector)}")

    if len(vector) != expected_dim:
        raise EmbeddingError(
            f"Embedding vector dimension mismatch for {context_label}: expected {expected_dim}, got {len(vector)}"
        )

    for idx, val in enumerate(vector):
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise EmbeddingError(f"Embedding vector contains non-finite value at index {idx} for {context_label}: {val}")


class EmbeddingService:
    """Service adapter for generating embeddings via official Google GenAI SDK."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        client: Optional[genai.Client] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SEC,
    ) -> None:
        self.config = config or load_config()
        self.model_name = self.config.gemini_embedding_model or DEFAULT_EMBEDDING_MODEL
        self.dimension = getattr(self.config, "embedding_dimension", DEFAULT_EMBEDDING_DIMENSION)
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        # Initialize official GenAI client
        if client is not None:
            self._client = client
        else:
            if not self.config.has_api_key:
                logger.warning("Initializing EmbeddingService without an active GEMINI_API_KEY")
                self._client = None
            else:
                self._client = genai.Client(api_key=self.config.gemini_api_key)

    @property
    def is_configured(self) -> bool:
        """Check if client is ready with credentials."""
        return self._client is not None

    def _require_client(self) -> genai.Client:
        """Return valid client or raise ConfigurationError."""
        if self._client is None:
            raise ConfigurationError(
                "Gemini API key is not configured.",
                user_friendly_message="Gemini API key is required to generate embeddings. Please set GEMINI_API_KEY in .env or settings.",
            )
        return self._client

    def _call_embed_with_retry(self, formatted_content: str, label: str) -> List[float]:
        """Execute a single embedding request with bounded exponential backoff for transient errors."""
        client = self._require_client()
        attempt = 0

        while True:
            try:
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=formatted_content,
                    config={"output_dimensionality": self.dimension},
                )

                if not response or not response.embeddings:
                    raise EmbeddingError(f"Google GenAI returned empty embedding response for {label}")

                vector = response.embeddings[0].values
                validate_embedding_vector(vector, self.dimension, context_label=label)
                return list(vector)

            except (genai_errors.ClientError, genai_errors.APIError) as err:
                code = getattr(err, "code", None)
                # Permanent client errors (400, 401, 403, 404): fail immediately
                if code in {400, 401, 403, 404}:
                    logger.error("Permanent API error %s while embedding %s: %s", code, label, str(err))
                    if code in (401, 403):
                        user_msg = "Authentication failed: Invalid or unauthorized Gemini API key. Please verify GEMINI_API_KEY."
                    else:
                        user_msg = "Configuration or request error with Google Gemini API. Please check your model settings."
                    raise EmbeddingError(
                        f"Gemini API error ({code}): {str(err)}",
                        user_friendly_message=user_msg,
                    ) from err

                # Retriable errors: 429 (Resource Exhausted) or 5xx Server Errors
                is_retriable = (
                    code == 429
                    or isinstance(err, genai_errors.ServerError)
                    or (isinstance(code, int) and 500 <= code < 600)
                )

                if is_retriable and attempt < self.max_retries:
                    attempt += 1
                    sleep_time = self.initial_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Retriable error (%s) on %s (attempt %d/%d). Retrying in %.2fs: %s",
                        code,
                        label,
                        attempt,
                        self.max_retries,
                        sleep_time,
                        str(err),
                    )
                    time.sleep(sleep_time)
                    continue

                logger.error("Exhausted retries or non-retriable API error for %s: %s", label, str(err))
                if code == 429:
                    user_msg = "Gemini embedding API quota or rate limit exceeded. Please check your Google AI Studio quota or try again later."
                elif isinstance(code, int) and 500 <= code < 600:
                    user_msg = "Google Gemini API service outage (500/503). Please try again shortly."
                else:
                    user_msg = "Temporary embedding service failure. Please verify API quota or try again later."
                raise EmbeddingError(
                    f"Gemini embedding API failed after {attempt} retries: {str(err)}",
                    user_friendly_message=user_msg,
                ) from err

            except EmbeddingError:
                raise

            except Exception as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    sleep_time = self.initial_backoff * (2 ** (attempt - 1))
                    logger.warning("Transient error on %s. Retrying in %.2fs: %s", label, sleep_time, str(exc))
                    time.sleep(sleep_time)
                    continue
                logger.error("Unexpected error embedding %s: %s", label, str(exc), exc_info=True)
                raise EmbeddingError(f"Unexpected embedding error for {label}: {str(exc)}") from exc

    def embed_chunk(self, text: str, document_title: str) -> List[float]:
        """Embed an individual document chunk using asymmetric document format."""
        formatted = format_document_text(title=document_title, text=text)
        return self._call_embed_with_retry(formatted, label=f"chunk ({document_title})")

    def embed_query(self, query: str) -> List[float]:
        """Embed a natural-language search query using asymmetric query task format."""
        formatted = format_query_text(query=query)
        return self._call_embed_with_retry(formatted, label="query")

    def embed_chunks(
        self,
        chunks: List[ChunkRecord],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """Embed each document chunk individually, populating chunk.embedding and returning all vectors.

        NOTE: Per specification for gemini-embedding-2, each document chunk is embedded individually
        to prevent input aggregation across chunks.
        """
        if not chunks:
            return []

        total = len(chunks)
        embeddings: List[List[float]] = []

        logger.info("Generating embeddings for %d chunks using '%s' (dim=%d)", total, self.model_name, self.dimension)

        for idx, chunk in enumerate(chunks):
            vector = self.embed_chunk(text=chunk.text, document_title=chunk.source_filename)
            chunk.embedding = vector
            embeddings.append(vector)

            if progress_callback is not None:
                try:
                    progress_callback(idx + 1, total)
                except Exception as cb_err:
                    logger.debug("Progress callback error ignored: %s", str(cb_err))

        return embeddings
