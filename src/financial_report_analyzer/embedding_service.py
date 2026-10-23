"""Google GenAI embedding service supporting gemini-embedding-2 with token-aware pacing and rate-limit hardening."""

import re
import time
import math
import random
import logging
import collections
from typing import List, Optional, Callable, Dict, Any, Tuple, Union
from dataclasses import dataclass
from google import genai
from google.genai import errors as genai_errors

from financial_report_analyzer.config import AppConfig, load_config
from financial_report_analyzer.errors import ConfigurationError, EmbeddingError
from financial_report_analyzer.schemas import ChunkRecord

logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF_SEC = 2.0
DEFAULT_MAX_BACKOFF_SEC = 60.0
DEFAULT_TPM_LIMIT = 30000
DEFAULT_RPM_LIMIT = 60
DEFAULT_SAFETY_MARGIN = 0.80


@dataclass(frozen=True)
class APIErrorDetails:
    """Classified error attributes from Google GenAI provider exceptions."""

    error_type: str  # 'daily_quota_exhausted', 'tpm_limit', 'rpm_limit', 'rate_limit_transient', 'auth_error', 'server_error', 'bad_request', 'unknown'
    is_retriable: bool
    is_daily_quota: bool
    user_friendly_message: str
    suggested_wait_sec: Optional[float] = None

    @property
    def retriable(self) -> bool:
        return self.is_retriable

    @property
    def retry_after_sec(self) -> Optional[float]:
        return self.suggested_wait_sec

    @property
    def is_rate_limit(self) -> bool:
        return self.error_type in ("tpm_limit", "rpm_limit", "rate_limit_transient", "daily_quota_exhausted")

    @property
    def is_tpm(self) -> bool:
        return self.error_type == "tpm_limit"

    @property
    def is_rpm(self) -> bool:
        return self.error_type == "rpm_limit"

    @property
    def is_auth(self) -> bool:
        return self.error_type == "auth_error"

    @property
    def is_server_error(self) -> bool:
        return self.error_type == "server_error"


def classify_genai_error(err: Exception) -> APIErrorDetails:
    """Classify Google GenAI exception into actionable categories and non-leaking user messages."""
    status_code = getattr(err, "code", None)
    err_str = str(err).lower()

    # 1. Authentication & Permissions (Permanent, NOT retriable)
    if status_code in (401, 403) or "api_key_invalid" in err_str or "api key not valid" in err_str:
        return APIErrorDetails(
            error_type="auth_error",
            is_retriable=False,
            is_daily_quota=False,
            user_friendly_message="Authentication failed: Invalid, expired, or unauthorized Gemini API key. Please verify GEMINI_API_KEY.",
        )

    # 2. Daily Quota Exhaustion (Permanent for the day, NOT retriable)
    # Distinguish daily quota from per-minute rate limits:
    is_daily = (
        "per day" in err_str
        or "daily" in err_str
        or "queries per day" in err_str
        or "perday" in err_str
        or "free tier limit" in err_str and "day" in err_str
    )
    if is_daily:
        return APIErrorDetails(
            error_type="daily_quota_exhausted",
            is_retriable=False,
            is_daily_quota=True,
            user_friendly_message=(
                "Google Gemini API daily quota exhausted for this API key. "
                "Daily quota resets at midnight Pacific Time. "
                "To continue immediately, configure a billing account or an API key with higher daily quota in Google AI Studio."
            ),
        )

    # 3. Rate Limits (HTTP 429 or ResourceExhausted)
    if status_code == 429 or "resource_exhausted" in err_str or "quota exceeded" in err_str or "too many requests" in err_str:
        # Check for suggested wait time in error message (e.g., "wait 18s", "retry after 20s", "retry in 5s", "after 15.2 seconds")
        suggested_wait = None
        match = re.search(r"(?:wait|retry(?:[- ]after)?|retry in|after)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?", err_str)
        if match:
            try:
                suggested_wait = float(match.group(1))
            except ValueError:
                pass

        if "token" in err_str or "tpm" in err_str:
            return APIErrorDetails(
                error_type="tpm_limit",
                is_retriable=True,
                is_daily_quota=False,
                suggested_wait_sec=suggested_wait,
                user_friendly_message=(
                    "Gemini embedding Token-Per-Minute (TPM) limit reached. "
                    "The system will automatically pace requests to resume embedding within your quota."
                ),
            )
        elif "request" in err_str or "rpm" in err_str or "queries" in err_str:
            return APIErrorDetails(
                error_type="rpm_limit",
                is_retriable=True,
                is_daily_quota=False,
                suggested_wait_sec=suggested_wait,
                user_friendly_message=(
                    "Gemini embedding Request-Per-Minute (RPM) limit reached. "
                    "The system will automatically pace requests."
                ),
            )
        else:
            return APIErrorDetails(
                error_type="rate_limit_transient",
                is_retriable=True,
                is_daily_quota=False,
                suggested_wait_sec=suggested_wait,
                user_friendly_message=(
                    "Gemini embedding rate limit exceeded. Pacing requests with exponential backoff and jitter."
                ),
            )

    # 4. Server Outages (500, 502, 503, 504)
    if status_code in (500, 502, 503, 504) or isinstance(err, genai_errors.ServerError):
        return APIErrorDetails(
            error_type="server_error",
            is_retriable=True,
            is_daily_quota=False,
            user_friendly_message="Google Gemini API temporary service outage (500/503). Retrying shortly.",
        )

    # 5. Bad Request / Model error (400, 404)
    if status_code in (400, 404):
        return APIErrorDetails(
            error_type="bad_request",
            is_retriable=False,
            is_daily_quota=False,
            user_friendly_message="Gemini API request error. Please verify the configured embedding model and dimension.",
        )

    # 6. Default / Unknown
    return APIErrorDetails(
        error_type="unknown",
        is_retriable=True,
        is_daily_quota=False,
        user_friendly_message=f"Gemini embedding service error: {str(err)}",
    )


def calculate_backoff_with_jitter(
    attempt: int,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SEC,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SEC,
    jitter_factor: float = 0.20,
    suggested_wait: Optional[float] = None,
) -> float:
    """Calculate exponential backoff with random jitter to avoid thundering herds.

    If suggested_wait is provided, it is prioritized and scaled with a safety jitter factor.
    """
    if suggested_wait is not None and suggested_wait > 0:
        base = min(max_backoff, suggested_wait)
        return base * random.uniform(1.0, 1.0 + jitter_factor)

    base = min(max_backoff, initial_backoff * (2 ** attempt))
    low = base * max(0.0, 1.0 - jitter_factor)
    high = base * (1.0 + jitter_factor)
    return random.uniform(low, high)


class TokenPacer:
    """Sliding-window token-aware and request-aware rate limiter.

    Enforces both Tokens-Per-Minute (TPM) and Requests-Per-Minute (RPM) with a configurable
    safety margin to absorb token-estimation errors on dense financial text.
    """

    def __init__(
        self,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        window_seconds: float = 60.0,
        sleeper: Callable[[float], None] = time.sleep,
        sleep_fn: Optional[Callable[[float], None]] = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tpm_limit = max(1, tpm_limit)
        self.rpm_limit = max(1, rpm_limit)
        self.safety_margin = max(0.1, min(1.0, safety_margin))
        self.window_seconds = window_seconds
        self._sleeper = sleep_fn if sleep_fn is not None else sleeper
        self._time_fn = time_fn
        # History queue of (timestamp, token_count)
        self._history: collections.deque[Tuple[float, int]] = collections.deque()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count with safety buffer."""
        return estimate_token_count(text)
        # History queue of (timestamp, token_count)
        self._history: collections.deque[Tuple[float, int]] = collections.deque()

    @property
    def effective_tpm(self) -> int:
        """Effective TPM ceiling incorporating safety margin."""
        return max(1, int(self.tpm_limit * self.safety_margin))

    @property
    def effective_rpm(self) -> int:
        """Effective RPM ceiling incorporating safety margin."""
        return max(1, int(self.rpm_limit * self.safety_margin))

    def _prune(self, now: float) -> None:
        """Prune entries outside the sliding window."""
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] <= cutoff:
            self._history.popleft()

    def current_usage(self) -> Tuple[int, int]:
        """Return (tokens_in_window, requests_in_window)."""
        now = self._time_fn()
        self._prune(now)
        tokens = sum(tok for _, tok in self._history)
        return tokens, len(self._history)

    def calculate_wait_seconds(self, tokens_needed: int) -> float:
        """Calculate wait seconds before tokens_needed can be dispatched without exceeding limits."""
        now = self._time_fn()
        self._prune(now)

        current_tokens = sum(tok for _, tok in self._history)
        current_reqs = len(self._history)
        wait_sec = 0.0

        # Check RPM limit
        if current_reqs + 1 > self.effective_rpm and self._history:
            oldest_req_time = self._history[0][0]
            wait_sec = max(wait_sec, (oldest_req_time + self.window_seconds) - now)

        # Check TPM limit
        # Cap single item requirement to effective_tpm to prevent infinite block on large items
        effective_tokens_needed = min(tokens_needed, self.effective_tpm)
        if current_tokens + effective_tokens_needed > self.effective_tpm:
            needed_reduction = (current_tokens + effective_tokens_needed) - self.effective_tpm
            shed = 0
            for ts, tok in self._history:
                shed += tok
                wait_sec = max(wait_sec, (ts + self.window_seconds) - now)
                if shed >= needed_reduction:
                    break

        return max(0.0, wait_sec)

    def acquire(self, tokens_needed: int = 1, estimated_tokens: Optional[int] = None) -> float:
        """Wait if necessary to stay within budget, then record token usage. Returns seconds waited."""
        effective_tokens = estimated_tokens if estimated_tokens is not None else tokens_needed
        wait_sec = self.calculate_wait_seconds(effective_tokens)
        if wait_sec > 0:
            self._sleeper(wait_sec)

        now = self._time_fn()
        self._prune(now)
        self._history.append((now, effective_tokens))
        return wait_sec


def format_document_text(title: str, text: str) -> str:
    """Format document chunk text using recommended asymmetric retrieval prefix."""
    clean_title = title.strip() if title else "Document"
    clean_text = text.strip() if text else ""
    return f"title: {clean_title} | text: {clean_text}"


def format_query_text(query: str) -> str:
    """Format query text using recommended asymmetric retrieval task prefix."""
    clean_query = query.strip() if query else ""
    return f"task: question answering | query: {clean_query}"


def estimate_token_count(text: str) -> int:
    """Deterministic token estimation heuristic (~4 chars/token + safety buffer)."""
    return max(1, len(text) // 4)


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
    """Service adapter for generating embeddings via official Google GenAI SDK with token pacing and retries."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        client: Optional[genai.Client] = None,
        pacer: Optional[TokenPacer] = None,
        max_retries: Optional[int] = None,
        initial_backoff: Optional[float] = None,
        max_backoff: Optional[float] = None,
    ) -> None:
        self.config = config or load_config()
        self.model_name = self.config.gemini_embedding_model or DEFAULT_EMBEDDING_MODEL
        self.dimension = getattr(self.config, "embedding_dimension", DEFAULT_EMBEDDING_DIMENSION)
        self.max_retries = max_retries if max_retries is not None else getattr(self.config, "embedding_max_retries", DEFAULT_MAX_RETRIES)
        self.initial_backoff = initial_backoff if initial_backoff is not None else getattr(self.config, "embedding_initial_backoff_sec", DEFAULT_INITIAL_BACKOFF_SEC)
        self.max_backoff = max_backoff if max_backoff is not None else getattr(self.config, "embedding_max_backoff_sec", DEFAULT_MAX_BACKOFF_SEC)
        self.batch_size = getattr(self.config, "embedding_batch_size", 1)

        # Initialize rate pacer
        if pacer is not None:
            self.pacer = pacer
        else:
            tpm = getattr(self.config, "embedding_tpm_limit", DEFAULT_TPM_LIMIT)
            rpm = getattr(self.config, "embedding_rpm_limit", DEFAULT_RPM_LIMIT)
            margin = getattr(self.config, "embedding_safety_margin", DEFAULT_SAFETY_MARGIN)
            self.pacer = TokenPacer(tpm_limit=tpm, rpm_limit=rpm, safety_margin=margin)

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

    def _call_embed_with_retry(
        self,
        formatted_contents: Union[str, List[str]],
        label: str,
        tokens_est: int = 100,
    ) -> List[List[float]]:
        """Execute embedding request with token-aware pacing and bounded exponential backoff with jitter."""
        client = self._require_client()
        attempt = 0

        # Pace the request before sending
        waited = self.pacer.acquire(tokens_est)
        if waited > 0:
            logger.info("Paced request for %s (waited %.2fs to observe TPM/RPM ceiling)", label, waited)

        is_batch = isinstance(formatted_contents, list)

        while True:
            try:
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=formatted_contents,
                    config={"output_dimensionality": self.dimension},
                )

                if not response or not response.embeddings:
                    raise EmbeddingError(f"Google GenAI returned empty embedding response for {label}")

                vectors: List[List[float]] = []
                for idx, emb_item in enumerate(response.embeddings):
                    vec = emb_item.values
                    validate_embedding_vector(vec, self.dimension, context_label=f"{label}[{idx}]")
                    vectors.append(list(vec))

                return vectors

            except (genai_errors.ClientError, genai_errors.APIError) as err:
                info = classify_genai_error(err)

                # Daily Quota Exhaustion: FAIL IMMEDIATELY (no retries)
                if info.is_daily_quota:
                    logger.error("Daily Gemini API quota exhausted for %s: %s", label, str(err))
                    raise EmbeddingError(
                        f"Gemini API daily quota exhausted: {str(err)}",
                        user_friendly_message=info.user_friendly_message,
                    ) from err

                # Permanent non-retriable error: FAIL IMMEDIATELY
                if not info.is_retriable:
                    logger.error("Permanent API error (%s) for %s: %s", info.error_type, label, str(err))
                    raise EmbeddingError(
                        f"Gemini API error ({info.error_type}): {str(err)}",
                        user_friendly_message=info.user_friendly_message,
                    ) from err

                # Retriable rate limits or server errors
                if attempt < self.max_retries:
                    attempt += 1
                    backoff = calculate_backoff_with_jitter(
                        attempt=attempt,
                        initial_backoff=self.initial_backoff,
                        max_backoff=self.max_backoff,
                    )
                    if info.suggested_wait_sec is not None:
                        backoff = max(backoff, info.suggested_wait_sec)

                    logger.warning(
                        "Retriable error (%s) on %s (attempt %d/%d). Sleeping %.2fs before retry...",
                        info.error_type,
                        label,
                        attempt,
                        self.max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                logger.error("Exhausted retries (%d) for %s on %s: %s", self.max_retries, label, info.error_type, str(err))
                raise EmbeddingError(
                    f"Gemini embedding API failed after {attempt} retries: {str(err)}",
                    user_friendly_message=info.user_friendly_message,
                ) from err

            except EmbeddingError:
                raise

            except Exception as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    backoff = calculate_backoff_with_jitter(
                        attempt=attempt,
                        initial_backoff=self.initial_backoff,
                        max_backoff=self.max_backoff,
                    )
                    logger.warning("Transient error on %s. Retrying in %.2fs: %s", label, backoff, str(exc))
                    time.sleep(backoff)
                    continue
                logger.error("Unexpected error embedding %s: %s", label, str(exc), exc_info=True)
                raise EmbeddingError(f"Unexpected embedding error for {label}: {str(exc)}") from exc

    def embed_chunk(self, text: str, document_title: str) -> List[float]:
        """Embed an individual document chunk using asymmetric document format."""
        formatted = format_document_text(title=document_title, text=text)
        tokens_est = estimate_token_count(formatted) + 15
        vectors = self._call_embed_with_retry(formatted, label=f"chunk ({document_title})", tokens_est=tokens_est)
        return vectors[0]

    def embed_query(self, query: str) -> List[float]:
        """Embed a natural-language search query using asymmetric query task format."""
        formatted = format_query_text(query=query)
        tokens_est = estimate_token_count(formatted) + 10
        vectors = self._call_embed_with_retry(formatted, label="query", tokens_est=tokens_est)
        return vectors[0]

    def embed_chunks(
        self,
        chunks: List[ChunkRecord],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """Embed document chunks with token pacing, partial progress preservation, and batching.

        If any chunks already have valid in-memory embeddings (from a previous partial run),
        they are preserved without re-calling the API, enabling seamless resumption.
        """
        if not chunks:
            return []

        total = len(chunks)
        already_embedded = sum(1 for c in chunks if c.embedding is not None and len(c.embedding) == self.dimension)

        if already_embedded > 0:
            logger.info("Resuming embedding: %d of %d chunks already embedded in memory", already_embedded, total)

        embeddings: List[List[float]] = []

        logger.info(
            "Generating embeddings for %d chunks using '%s' (dim=%d, batch_size=%d, effective_tpm=%d)",
            total,
            self.model_name,
            self.dimension,
            self.batch_size,
            self.pacer.effective_tpm,
        )

        batch_size = max(1, self.batch_size)
        idx = 0

        while idx < total:
            # Check if current chunk already has a vector
            if chunks[idx].embedding is not None and len(chunks[idx].embedding) == self.dimension:
                embeddings.append(chunks[idx].embedding)
                idx += 1
                if progress_callback:
                    try:
                        progress_callback(idx, total)
                    except Exception:
                        pass
                continue

            # If batch_size == 1 (default), embed single chunk
            if batch_size == 1:
                chunk = chunks[idx]
                tokens_est = getattr(chunk, "token_count_estimate", None) or estimate_token_count(chunk.text)
                tokens_est += 15

                try:
                    vector = self.embed_chunk(text=chunk.text, document_title=chunk.source_filename)
                    chunk.embedding = vector
                    embeddings.append(vector)
                except EmbeddingError as err:
                    # Partial progress is preserved on chunk.embedding in memory
                    logger.error("Embedding interrupted at chunk %d/%d: %s", idx + 1, total, str(err))
                    err.user_friendly_message = (
                        f"{err.user_friendly_message} (Embedded {idx}/{total} chunks before interruption; "
                        "completed chunks are preserved in memory)."
                    )
                    raise

                idx += 1
                if progress_callback:
                    try:
                        progress_callback(idx, total)
                    except Exception:
                        pass
            else:
                # Group up to batch_size consecutive un-embedded chunks
                batch: List[ChunkRecord] = []
                while idx < total and len(batch) < batch_size:
                    if chunks[idx].embedding is not None and len(chunks[idx].embedding) == self.dimension:
                        break
                    batch.append(chunks[idx])
                    idx += 1

                formatted_list = [format_document_text(c.source_filename, c.text) for c in batch]
                batch_tokens = sum(estimate_token_count(f) + 15 for f in formatted_list)

                try:
                    vectors = self._call_embed_with_retry(
                        formatted_list,
                        label=f"batch ({len(batch)} chunks)",
                        tokens_est=batch_tokens,
                    )
                    for b_idx, c in enumerate(batch):
                        c.embedding = vectors[b_idx]
                        embeddings.append(vectors[b_idx])
                except EmbeddingError as err:
                    logger.error("Batch embedding interrupted: %s", str(err))
                    raise

                if progress_callback:
                    try:
                        progress_callback(idx, total)
                    except Exception:
                        pass

        return embeddings
