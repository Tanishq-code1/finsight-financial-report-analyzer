"""Google GenAI embedding service supporting gemini-embedding-2 with token-aware pacing and rate-limit hardening."""

import re
import time
import math
import random
import logging
import collections
import threading
from typing import List, Optional, Callable, Dict, Any, Tuple, Union
from dataclasses import dataclass
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from financial_report_analyzer.config import AppConfig, load_config
from financial_report_analyzer.errors import ConfigurationError, EmbeddingError
from financial_report_analyzer.schemas import ChunkRecord
from financial_report_analyzer.cache import GLOBAL_CACHE

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

    error_type: str  # 'daily_quota_exhausted', 'tpm_limit', 'rpm_limit', 'rate_limit_transient', 'auth_error', 'permission_denied', 'server_error', 'bad_request', 'unknown'
    is_retriable: bool
    is_daily_quota: bool
    user_friendly_message: str
    suggested_wait_sec: Optional[float] = None
    quota_metric: Optional[str] = None

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
        return self.error_type in ("auth_error", "permission_denied")

    @property
    def is_server_error(self) -> bool:
        return self.error_type == "server_error"


def extract_genai_error_payload(err: Exception) -> Tuple[Optional[int], str, Optional[float], Optional[str]]:
    """Safely inspect Google GenAI exception structures and extract status, message, retryDelay, and quota metric.

    Handles:
    - google.genai.errors.APIError / ClientError / ServerError
    - Nested google.rpc.RetryInfo (retryDelay e.g. "25s", "25.123s", {"seconds": 25, "nanos": ...})
    - Nested google.rpc.QuotaFailure (violations with subject & description, e.g. embed_content_free_tier_tokens)
    - Nested google.rpc.ErrorInfo (reason, domain, metadata with quota_metric)
    - Response headers (e.g. Retry-After)
    - Fallback regex parsing on error message and string representation
    """
    status_code = getattr(err, "code", None)
    if status_code is None and hasattr(err, "response") and err.response is not None:
        status_code = getattr(err.response, "status_code", None)

    message = getattr(err, "message", None) or str(err)
    retry_delay: Optional[float] = None
    quota_metric: Optional[str] = None

    # Inspect nested details (Google RPC structure)
    details = getattr(err, "details", None)
    if isinstance(details, dict):
        candidate_lists = []
        if "error" in details and isinstance(details["error"], dict):
            if "details" in details["error"] and isinstance(details["error"]["details"], list):
                candidate_lists.append(details["error"]["details"])
        if "details" in details and isinstance(details["details"], list):
            candidate_lists.append(details["details"])

        for d_list in candidate_lists:
            for item in d_list:
                if not isinstance(item, dict):
                    continue
                type_str = str(item.get("@type", "")).lower()

                # RetryInfo
                if "retryinfo" in type_str or "retrydelay" in item or "retry_delay" in item:
                    delay_val = item.get("retryDelay") or item.get("retry_delay")
                    if isinstance(delay_val, str):
                        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", delay_val)
                        if m:
                            try:
                                retry_delay = float(m.group(1))
                            except ValueError:
                                pass
                    elif isinstance(delay_val, (int, float)):
                        retry_delay = float(delay_val)
                    elif isinstance(delay_val, dict):
                        sec = float(delay_val.get("seconds", 0))
                        nanos = float(delay_val.get("nanos", 0))
                        retry_delay = sec + (nanos / 1e9)

                # QuotaFailure
                if "quotafailure" in type_str or "violations" in item:
                    for v in item.get("violations", []):
                        if isinstance(v, dict):
                            desc = v.get("description", "")
                            m = re.search(r"quota metric ['\"]([^'\"]+)['\"]", desc, re.IGNORECASE)
                            if m:
                                quota_metric = m.group(1)

                # ErrorInfo
                if "errorinfo" in type_str or "metadata" in item:
                    meta = item.get("metadata", {})
                    if isinstance(meta, dict) and "quota_metric" in meta:
                        quota_metric = meta["quota_metric"]

    elif isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                type_str = str(item.get("@type", "")).lower()
                if "retryinfo" in type_str or "retrydelay" in item or "retry_delay" in item:
                    delay_val = item.get("retryDelay") or item.get("retry_delay")
                    if isinstance(delay_val, str):
                        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", delay_val)
                        if m:
                            try:
                                retry_delay = float(m.group(1))
                            except ValueError:
                                pass
                    elif isinstance(delay_val, (int, float)):
                        retry_delay = float(delay_val)
                    elif isinstance(delay_val, dict):
                        sec = float(delay_val.get("seconds", 0))
                        nanos = float(delay_val.get("nanos", 0))
                        retry_delay = sec + (nanos / 1e9)

    # Check response headers (Retry-After)
    if retry_delay is None and hasattr(err, "response") and err.response is not None:
        headers = getattr(err.response, "headers", None)
        if headers and hasattr(headers, "get"):
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                try:
                    retry_delay = float(ra)
                except ValueError:
                    pass

    # Fallback string searches on str(err)
    err_str = str(err)
    if retry_delay is None:
        m_delay = re.search(
            r"(?:retry[_-]?delay|retry[- ]after|wait|retry in|after)['\":\s=]+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds?)?",
            err_str,
            re.IGNORECASE,
        )
        if m_delay:
            try:
                retry_delay = float(m_delay.group(1))
            except ValueError:
                pass

    if quota_metric is None:
        m_metric = re.search(r"quota metric ['\"]([^'\"]+)['\"]", err_str, re.IGNORECASE)
        if m_metric:
            quota_metric = m_metric.group(1)

    return status_code, message, retry_delay, quota_metric


def classify_genai_error(err: Exception) -> APIErrorDetails:
    """Classify Google GenAI exception into actionable categories and non-leaking user messages."""
    status_code, message, suggested_wait, quota_metric = extract_genai_error_payload(err)
    err_str = f"{message} {str(err)}".lower()
    metric_str = (quota_metric or "").lower()

    # 1. Authentication & Permissions (Permanent, NOT retriable)
    if status_code in (401, 403) or "api_key_invalid" in err_str or "api key not valid" in err_str:
        return APIErrorDetails(
            error_type="auth_error" if status_code == 401 else "permission_denied",
            is_retriable=False,
            is_daily_quota=False,
            suggested_wait_sec=None,
            quota_metric=quota_metric,
            user_friendly_message="Authentication failed: Invalid, expired, or unauthorized Gemini API key. Please verify GEMINI_API_KEY.",
        )

    # 2. Daily Quota Exhaustion (Permanent for the day, NOT retriable)
    is_daily = (
        "per_day" in metric_str
        or "queries per day" in metric_str
        or "daily" in metric_str
        or "per day" in err_str
        or "daily" in err_str
        or "queries per day" in err_str
        or "perday" in err_str
        or ("free tier limit" in err_str and "day" in err_str)
    )
    if is_daily:
        return APIErrorDetails(
            error_type="daily_quota_exhausted",
            is_retriable=False,
            is_daily_quota=True,
            suggested_wait_sec=None,
            quota_metric=quota_metric,
            user_friendly_message=(
                "Google Gemini API daily quota exhausted for this API key. "
                "Daily quota resets at midnight Pacific Time. "
                "To continue immediately, configure a billing account or an API key with higher daily quota in Google AI Studio."
            ),
        )

    # 3. Rate Limits (HTTP 429 or ResourceExhausted)
    if (
        status_code == 429
        or "resource_exhausted" in err_str
        or "quota exceeded" in err_str
        or "too many requests" in err_str
    ):
        wait_desc = f" Provider suggested wait: {int(suggested_wait)}s." if suggested_wait else ""

        if "token" in metric_str or "tpm" in metric_str or "token" in err_str or "tpm" in err_str:
            return APIErrorDetails(
                error_type="tpm_limit",
                is_retriable=True,
                is_daily_quota=False,
                suggested_wait_sec=suggested_wait,
                quota_metric=quota_metric or "embed_content_free_tier_tokens",
                user_friendly_message=(
                    f"Gemini embedding Token-Per-Minute (TPM) quota limit reached (Free Tier limit: 30,000 input tokens/min).{wait_desc} "
                    "The system will pace requests to resume within your quota."
                ),
            )
        elif "request" in metric_str or "rpm" in metric_str or "request" in err_str or "rpm" in err_str:
            return APIErrorDetails(
                error_type="rpm_limit",
                is_retriable=True,
                is_daily_quota=False,
                suggested_wait_sec=suggested_wait,
                quota_metric=quota_metric or "embed_content_free_tier_requests",
                user_friendly_message=(
                    f"Gemini embedding Request-Per-Minute (RPM) limit reached.{wait_desc} "
                    "The system will pace requests."
                ),
            )
        else:
            return APIErrorDetails(
                error_type="rate_limit_transient",
                is_retriable=True,
                is_daily_quota=False,
                suggested_wait_sec=suggested_wait,
                quota_metric=quota_metric,
                user_friendly_message=(
                    f"Gemini embedding rate limit exceeded.{wait_desc} Pacing requests with exponential backoff and jitter."
                ),
            )

    # 4. Server Outages (500, 502, 503, 504)
    if status_code in (500, 502, 503, 504) or isinstance(err, genai_errors.ServerError):
        return APIErrorDetails(
            error_type="server_error",
            is_retriable=True,
            is_daily_quota=False,
            suggested_wait_sec=suggested_wait,
            quota_metric=quota_metric,
            user_friendly_message="Google Gemini API temporary service outage (500/503). Retrying shortly.",
        )

    # 5. Bad Request / Model error (400, 404)
    if status_code in (400, 404):
        return APIErrorDetails(
            error_type="bad_request",
            is_retriable=False,
            is_daily_quota=False,
            suggested_wait_sec=None,
            quota_metric=quota_metric,
            user_friendly_message="Gemini API request error. Please verify the configured embedding model and dimension.",
        )

    # 6. Default / Unknown
    return APIErrorDetails(
        error_type="unknown",
        is_retriable=False,
        is_daily_quota=False,
        suggested_wait_sec=suggested_wait,
        quota_metric=quota_metric,
        user_friendly_message=f"Gemini embedding service error: {str(err)}",
    )


def calculate_backoff_with_jitter(
    attempt: int,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SEC,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SEC,
    jitter_factor: float = 0.20,
    suggested_wait: Optional[float] = None,
) -> float:
    """Calculate exponential backoff with conservative positive jitter.

    CRITICAL INVARIANT:
    If suggested_wait is provided, the wait is NEVER shortened.
    Conservative jitter adds a positive safety margin (+0.5s to +2.0s) so the retry
    occurs strictly after the provider-specified retry delay.
    """
    if suggested_wait is not None and suggested_wait > 0:
        safety_buffer = max(0.5, min(5.0, suggested_wait * random.uniform(0.02, 0.08)))
        return suggested_wait + safety_buffer

    base = min(max_backoff, initial_backoff * (2 ** attempt))
    low = base * max(0.0, 1.0 - jitter_factor)
    high = base * (1.0 + jitter_factor)
    return random.uniform(low, high)


class BatchContent(genai_types.Content):
    """Duck-typed Content wrapper that preserves individual sub-requests in batchEmbedContents
    while remaining backwards-compatible with string operations in mocks and test suites."""

    def __init__(self, text: str, role: str = "user") -> None:
        super().__init__(parts=[genai_types.Part.from_text(text=text)], role=role)
        self._raw_text = text

    def __len__(self) -> int:
        return len(self._raw_text)

    def __str__(self) -> str:
        return self._raw_text

    def __repr__(self) -> str:
        return f"BatchContent({self._raw_text!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self._raw_text == other
        if isinstance(other, BatchContent):
            return self._raw_text == other._raw_text
        return super().__eq__(other)

    def lower(self) -> str:
        return self._raw_text.lower()


class TokenReservation:
    """Reservation handle for token and request quotas in the sliding window."""

    def __init__(self, pacer: "TokenPacer", tokens: int, timestamp: float, wait_sec: float = 0.0) -> None:
        self.pacer = pacer
        self.tokens = tokens
        self.timestamp = timestamp
        self.wait_sec = wait_sec
        self._released = False
        self._finalized = False

    def release(self) -> None:
        """Release reservation if request was aborted or failed without consuming quota."""
        if not self._released and not self._finalized:
            self._released = True
            self.pacer._cancel_reservation(self.timestamp, self.tokens)

    def finalize(self, actual_tokens: Optional[int] = None) -> None:
        """Finalize reservation after successful execution."""
        if not self._finalized and not self._released:
            self._finalized = True
            if actual_tokens is not None and actual_tokens != self.tokens:
                self.pacer._reconcile_reservation(self.timestamp, self.tokens, actual_tokens)

    def __enter__(self) -> "TokenReservation":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.release()
        else:
            self.finalize()


class TokenPacer:
    """Thread-safe sliding-window token-aware and request-aware rate limiter.

    Enforces both Tokens-Per-Minute (TPM) and Requests-Per-Minute (RPM) with a configurable
    safety margin, smooth inter-request spacing, and safe reservation cancellation.
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
        self.safety_margin = max(0.05, min(1.0, safety_margin))
        self.window_seconds = window_seconds
        self._sleeper = sleep_fn if sleep_fn is not None else sleeper
        self._time_fn = time_fn
        self._lock = threading.Lock()
        self._last_req_time: Optional[float] = None
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

    @property
    def min_request_interval(self) -> float:
        """Minimum spacing between consecutive requests to prevent unpaced bursts."""
        return self.window_seconds / float(self.effective_rpm)

    def _prune(self, now: float) -> None:
        """Prune entries outside the sliding window."""
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] <= cutoff:
            self._history.popleft()

    def current_usage(self) -> Tuple[int, int]:
        """Return (tokens_in_window, requests_in_window)."""
        with self._lock:
            now = self._time_fn()
            self._prune(now)
            tokens = sum(tok for _, tok in self._history)
            return tokens, len(self._history)

    def calculate_wait_seconds(self, tokens_needed: int) -> float:
        """Calculate wait seconds before tokens_needed can be dispatched without exceeding limits."""
        with self._lock:
            now = self._time_fn()
            self._prune(now)

            current_tokens = sum(tok for _, tok in self._history)
            current_reqs = len(self._history)
            wait_sec = 0.0

            # 1. Enforce minimum request spacing (e.g. 1.25s for 48 RPM)
            if self._last_req_time is not None:
                elapsed_since_last = now - self._last_req_time
                if elapsed_since_last < self.min_request_interval:
                    wait_sec = max(wait_sec, self.min_request_interval - elapsed_since_last)

            # 2. Check RPM window limit
            if current_reqs + 1 > self.effective_rpm and self._history:
                oldest_req_time = self._history[0][0]
                wait_sec = max(wait_sec, (oldest_req_time + self.window_seconds) - now)

            # 3. Check TPM window limit
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

    def reserve(self, tokens_needed: int = 1, estimated_tokens: Optional[int] = None) -> TokenReservation:
        """Reserve estimated tokens in budget, waiting if necessary to honor RPM and TPM limits."""
        effective_tokens = estimated_tokens if estimated_tokens is not None else tokens_needed
        wait_sec = self.calculate_wait_seconds(effective_tokens)
        if wait_sec > 0:
            self._sleeper(wait_sec)

        with self._lock:
            now = self._time_fn()
            self._prune(now)
            self._history.append((now, effective_tokens))
            self._last_req_time = now
            return TokenReservation(self, effective_tokens, now, wait_sec=wait_sec)

    def acquire(self, tokens_needed: int = 1, estimated_tokens: Optional[int] = None) -> float:
        """Wait if necessary to stay within budget, then record token usage. Returns seconds waited."""
        effective_tokens = estimated_tokens if estimated_tokens is not None else tokens_needed
        wait_sec = self.calculate_wait_seconds(effective_tokens)
        if wait_sec > 0:
            self._sleeper(wait_sec)

        with self._lock:
            now = self._time_fn()
            self._prune(now)
            self._history.append((now, effective_tokens))
            self._last_req_time = now
            return wait_sec

    def _cancel_reservation(self, timestamp: float, tokens: int) -> None:
        """Cancel an unused reservation upon failure."""
        with self._lock:
            try:
                self._history.remove((timestamp, tokens))
            except ValueError:
                pass

    def _reconcile_reservation(self, timestamp: float, old_tokens: int, new_tokens: int) -> None:
        """Reconcile estimated tokens with actual consumed tokens."""
        with self._lock:
            for i, (ts, tok) in enumerate(self._history):
                if ts == timestamp and tok == old_tokens:
                    self._history[i] = (ts, new_tokens)
                    break

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Deterministic token estimation heuristic (~4 chars/token)."""
        return estimate_token_count(text)

    @staticmethod
    def estimate_conservative_tokens(text: str) -> int:
        """Conservative token estimation for financial text, numbers, and tables."""
        return estimate_conservative_tokens(text)


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
    """Deterministic token estimation heuristic (~4 chars/token)."""
    if not text:
        return 1
    return max(1, len(text) // 4)


def estimate_conservative_tokens(text: str) -> int:
    """Conservative token estimation for financial text, tables, and numeric data."""
    if not text:
        return 1
    chars = len(text)
    words = len(text.split())
    char_est = int(chars / 3.2)
    word_est = int(words * 1.35)
    return max(1, max(char_est, word_est) + 15)


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

        # Telemetry statistics
        self.stats: Dict[str, Any] = {
            "effective_batch_size": self.batch_size,
            "api_requests": 0,
            "bulk_batches": 0,
            "individual_requests": 0,
            "vectors_generated": 0,
            "cache_hits": 0,
            "length_mismatches": 0,
            "fallbacks_to_individual": 0,
            "total_pacer_wait_sec": 0.0,
            "total_api_latency_sec": 0.0,
            "total_validation_sec": 0.0,
            "batch_sizes": collections.Counter(),
        }

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
        formatted_contents: Union[str, List[Any]],
        label: str,
        tokens_est: int = 100,
    ) -> List[List[float]]:
        """Execute embedding request with token-aware pacing and bounded exponential backoff with jitter."""
        client = self._require_client()
        attempt = 0
        start_time = time.monotonic()
        deadline_sec = getattr(self.config, "embedding_deadline_sec", 120.0)

        # Ensure batch list items serialize as distinct sub-requests in batchEmbedContents
        if isinstance(formatted_contents, list):
            contents_to_send = [
                c if isinstance(c, genai_types.Content) else BatchContent(str(c))
                for c in formatted_contents
            ]
        else:
            contents_to_send = formatted_contents

        # Reserve tokens before sending request
        reservation = self.pacer.reserve(tokens_est)
        if hasattr(self, "stats"):
            self.stats["total_pacer_wait_sec"] += getattr(reservation, "wait_sec", 0.0)

        while True:
            try:
                t_api_start = time.monotonic()
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=contents_to_send,
                    config={"output_dimensionality": self.dimension},
                )
                api_duration = time.monotonic() - t_api_start
                if hasattr(self, "stats"):
                    self.stats["total_api_latency_sec"] += api_duration
                    self.stats["api_requests"] += 1

                if not response or not response.embeddings:
                    raise EmbeddingError(f"Google GenAI returned empty embedding response for {label}")

                t_val_start = time.monotonic()
                vectors: List[List[float]] = []
                for idx, emb_item in enumerate(response.embeddings):
                    vec = emb_item.values
                    validate_embedding_vector(vec, self.dimension, context_label=f"{label}[{idx}]")
                    vectors.append(list(vec))
                val_duration = time.monotonic() - t_val_start
                if hasattr(self, "stats"):
                    self.stats["total_validation_sec"] += val_duration
                    self.stats["vectors_generated"] += len(vectors)

                reservation.finalize()
                return vectors

            except (genai_errors.ClientError, genai_errors.APIError) as err:
                info = classify_genai_error(err)

                # Daily Quota Exhaustion: FAIL IMMEDIATELY (no retries)
                if info.is_daily_quota:
                    reservation.release()
                    logger.error("Daily Gemini API quota exhausted for %s: %s", label, str(err))
                    raise EmbeddingError(
                        f"Gemini API daily quota exhausted: {str(err)}",
                        user_friendly_message=info.user_friendly_message,
                    ) from err

                # Permanent non-retriable error: FAIL IMMEDIATELY
                if not info.is_retriable:
                    reservation.release()
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
                        suggested_wait=info.suggested_wait_sec,
                    )

                    elapsed = time.monotonic() - start_time
                    if elapsed + backoff > deadline_sec:
                        reservation.release()
                        logger.error(
                            "Embedding deadline (%.1fs) reached after %.1fs (retry delay: %.1fs). Halting retries.",
                            deadline_sec,
                            elapsed,
                            backoff,
                        )
                        raise EmbeddingError(
                            f"Embedding rate-limit deadline reached after {elapsed:.1f}s (required wait {backoff:.1f}s).",
                            user_friendly_message=(
                                f"Google Gemini API rate limit reached. Provider requested a {int(backoff)}s wait. "
                                "Processing deadline reached; your partial progress is preserved in memory. "
                                "Click 'Resume Indexing' below to continue."
                            ),
                        ) from err

                    logger.warning(
                        "Retriable error (%s) on %s (attempt %d/%d). Sleeping %.2fs (suggested wait: %s)...",
                        info.error_type,
                        label,
                        attempt,
                        self.max_retries,
                        backoff,
                        str(info.suggested_wait_sec),
                    )
                    time.sleep(backoff)
                    continue

                reservation.release()
                logger.error("Exhausted retries (%d) for %s on %s: %s", self.max_retries, label, info.error_type, str(err))
                raise EmbeddingError(
                    f"Gemini embedding API failed after {attempt} retries: {str(err)}",
                    user_friendly_message=info.user_friendly_message,
                ) from err

            except EmbeddingError:
                reservation.release()
                raise

            except Exception as exc:
                reservation.release()
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
        """Embed an individual document chunk using asymmetric document format with caching."""
        cache_key = GLOBAL_CACHE.make_embedding_key(text, document_title, self.model_name, self.dimension)
        cached = GLOBAL_CACHE.get_embedding(cache_key)
        if cached is not None and len(cached) == self.dimension:
            return cached

        formatted = format_document_text(title=document_title, text=text)
        tokens_est = estimate_conservative_tokens(formatted)
        vectors = self._call_embed_with_retry(formatted, label=f"chunk ({document_title})", tokens_est=tokens_est)
        GLOBAL_CACHE.put_embedding(cache_key, vectors[0])
        return vectors[0]

    def embed_query(self, query: str) -> List[float]:
        """Embed a natural-language search query using asymmetric query task format."""
        formatted = format_query_text(query=query)
        tokens_est = estimate_conservative_tokens(formatted)
        vectors = self._call_embed_with_retry(formatted, label="query", tokens_est=tokens_est)
        return vectors[0]

    def embed_chunks(
        self,
        chunks: List[ChunkRecord],
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> List[List[float]]:
        """Embed document chunks with token pacing, partial progress preservation, and bulk batching."""
        if not chunks:
            return []

        total = len(chunks)

        # 1. First pass: check existing in-memory vectors and global cache
        cache_keys = [
            GLOBAL_CACHE.make_embedding_key(c.text, c.source_filename, self.model_name, self.dimension)
            for c in chunks
        ]
        hits, _ = GLOBAL_CACHE.get_embeddings_bulk(cache_keys)

        for i, c in enumerate(chunks):
            if c.embedding is not None and len(c.embedding) == self.dimension:
                continue
            if i in hits:
                c.embedding = hits[i]
                if hasattr(self, "stats"):
                    self.stats["cache_hits"] += 1

        already_ready = sum(1 for c in chunks if c.embedding is not None and len(c.embedding) == self.dimension)
        if already_ready > 0:
            logger.info("Embedding cache/memory hit: %d of %d chunks ready", already_ready, total)

        batch_size = max(1, self.batch_size)
        MAX_BATCH_TOKENS = 16000  # Conservative safe ceiling well below 24,000 effective TPM

        # Pre-plan batches for all un-embedded chunks
        planned_batches: List[List[int]] = []
        current_batch: List[int] = []
        current_tokens = 0

        for idx, c in enumerate(chunks):
            if c.embedding is not None and len(c.embedding) == self.dimension:
                continue
            fmt = format_document_text(c.source_filename, c.text)
            c_tok = estimate_conservative_tokens(fmt)
            if current_batch and (len(current_batch) >= batch_size or current_tokens + c_tok > MAX_BATCH_TOKENS):
                planned_batches.append(current_batch)
                current_batch = [idx]
                current_tokens = c_tok
            else:
                current_batch.append(idx)
                current_tokens += c_tok

        if current_batch:
            planned_batches.append(current_batch)

        total_batches = len(planned_batches)

        # If everything is already ready from memory/cache, trigger progress once
        if total_batches == 0:
            if progress_callback:
                try:
                    progress_callback(total, total, 1, 1)
                except TypeError:
                    progress_callback(total, total)
            return [c.embedding for c in chunks if c.embedding is not None]

        for batch_num, batch_indices in enumerate(planned_batches, start=1):
            batch_chunks = [chunks[i] for i in batch_indices]
            batch_keys = [cache_keys[i] for i in batch_indices]
            batch_formatted = [format_document_text(c.source_filename, c.text) for c in batch_chunks]
            batch_tokens = sum(estimate_conservative_tokens(fmt) for fmt in batch_formatted)

            if hasattr(self, "stats"):
                self.stats["batch_sizes"][len(batch_chunks)] += 1

            vectors: Optional[List[List[float]]] = None
            if len(batch_chunks) > 1:
                if hasattr(self, "stats"):
                    self.stats["bulk_batches"] += 1
                try:
                    res_vectors = self._call_embed_with_retry(
                        batch_formatted,
                        label=f"bulk batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks, ~{batch_tokens} tokens)",
                        tokens_est=batch_tokens,
                    )
                    if len(res_vectors) == len(batch_chunks):
                        vectors = res_vectors
                    else:
                        if hasattr(self, "stats"):
                            self.stats["length_mismatches"] += 1
                            self.stats["fallbacks_to_individual"] += 1
                        logger.warning(
                            "Bulk embedding returned %d vectors for %d chunks; falling back to 1-by-1 embedding.",
                            len(res_vectors),
                            len(batch_chunks),
                        )
                except Exception as exc:
                    if hasattr(self, "stats"):
                        self.stats["fallbacks_to_individual"] += 1
                    logger.warning(
                        "Bulk embedding failed for batch %d/%d (%s); falling back to 1-by-1 embedding.",
                        batch_num,
                        total_batches,
                        str(exc),
                    )

            if vectors is not None:
                pairs_to_cache: List[Tuple[str, List[float]]] = []
                for b_i, b_chunk in enumerate(batch_chunks):
                    b_chunk.embedding = vectors[b_i]
                    pairs_to_cache.append((batch_keys[b_i], vectors[b_i]))
                GLOBAL_CACHE.put_embeddings_bulk(pairs_to_cache)
            else:
                # 1-by-1 embedding fallback (or single item batch)
                if len(batch_chunks) == 1 and hasattr(self, "stats"):
                    self.stats["individual_requests"] += 1
                elif len(batch_chunks) > 1 and hasattr(self, "stats"):
                    self.stats["individual_requests"] += len(batch_chunks)

                for b_i, b_chunk in enumerate(batch_chunks):
                    vec = self.embed_chunk(b_chunk.text, b_chunk.source_filename)
                    b_chunk.embedding = vec

            if progress_callback:
                confirmed = sum(1 for c in chunks if c.embedding is not None and len(c.embedding) == self.dimension)
                try:
                    progress_callback(confirmed, total, batch_num, total_batches)
                except TypeError:
                    try:
                        progress_callback(confirmed, total)
                    except Exception:
                        pass

        return [c.embedding for c in chunks if c.embedding is not None]
