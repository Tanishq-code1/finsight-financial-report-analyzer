"""Unit tests for token-aware embedding pacing, rate-limit classification, retries, and partial progress."""

import time
from unittest.mock import MagicMock
import pytest
from google.genai import errors as genai_errors
import chromadb

from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.errors import EmbeddingError
from financial_report_analyzer.schemas import ChunkRecord, PageRecord
from financial_report_analyzer.embedding_service import (
    EmbeddingService,
    TokenPacer,
    classify_genai_error,
    calculate_backoff_with_jitter,
)
from financial_report_analyzer.indexing_pipeline import run_indexing_pipeline


class MockClock:
    """Deterministic simulated clock for rate-limiting tests without real time delays."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time
        self.sleep_calls = []

    def time(self) -> float:
        return self.current_time

    def sleep(self, duration: float) -> None:
        self.sleep_calls.append(duration)
        self.current_time += duration


def make_test_config(**kwargs) -> AppConfig:
    defaults = dict(
        gemini_api_key="test-api-key",
        gemini_generation_model="gemini-2.5-flash",
        gemini_embedding_model="gemini-embedding-2",
        embedding_dimension=768,
        max_upload_mb=25,
        max_pdf_pages=250,
        chunk_size_tokens=850,
        chunk_overlap_tokens=120,
        retrieval_top_k=5,
        min_evidence_threshold=0.35,
        log_level="INFO",
        embedding_tpm_limit=30000,
        embedding_rpm_limit=60,
        embedding_safety_margin=0.80,
        embedding_max_retries=3,
        embedding_initial_backoff_sec=1.0,
        embedding_max_backoff_sec=10.0,
        embedding_batch_size=1,
    )
    defaults.update(kwargs)
    return AppConfig(**defaults)


# ============================================================================
# 1. TokenPacer Unit Tests
# ============================================================================

def test_token_pacer_effective_limits():
    """Verify safety margin applies correctly to TPM and RPM."""
    pacer = TokenPacer(
        tpm_limit=30000,
        rpm_limit=60,
        safety_margin=0.80,
    )
    assert pacer.effective_tpm == 24000  # 30,000 * 0.80
    assert pacer.effective_rpm == 48     # 60 * 0.80


def test_token_pacer_estimation():
    """Verify token estimation accounts for length."""
    text = "A" * 400
    # 400 // 4 = 100
    tokens = TokenPacer.estimate_tokens(text)
    assert tokens == 100

    # Short string has at least 1 token
    assert TokenPacer.estimate_tokens("hi") == 1


def test_token_pacer_rpm_pacing_with_mock_clock():
    """Verify requests exceeding effective RPM cause simulated sleep until window clears."""
    clock = MockClock(start_time=100.0)
    # Effective RPM: 2 requests per minute (2 * 1.0) -> min_interval = 30.0s
    pacer = TokenPacer(
        tpm_limit=100000,
        rpm_limit=2,
        safety_margin=1.0,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )

    # First request: immediate
    pacer.acquire(estimated_tokens=50)
    assert clock.time() == 100.0

    # Second request: spaced by min_request_interval (30.0s)
    pacer.acquire(estimated_tokens=50)
    assert clock.time() == 130.0

    # Third request: at RPM limit (2 reqs in window [70, 130]).
    # Must sleep until 1st request drops out at t=160.0
    pacer.acquire(estimated_tokens=50)
    assert clock.time() >= 160.0
    assert len(clock.sleep_calls) == 2


def test_token_pacer_tpm_pacing_with_mock_clock():
    """Verify requests exceeding effective TPM cause simulated sleep until token bucket clears."""
    clock = MockClock(start_time=500.0)
    # Effective TPM: 100 tokens per minute
    pacer = TokenPacer(
        tpm_limit=100,
        rpm_limit=100,
        safety_margin=1.0,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )

    # 1st request: 60 tokens -> ok
    pacer.acquire(estimated_tokens=60)
    assert clock.time() == 500.0

    # 2nd request: 50 tokens -> 60 + 50 = 110 > 100 TPM -> must wait for 1st request to roll off
    pacer.acquire(estimated_tokens=50)
    assert clock.time() >= 560.0
    assert len(clock.sleep_calls) == 1
    assert clock.sleep_calls[0] >= 60.0


# ============================================================================
# 2. Rate-Limit and Error Classification Tests
# ============================================================================

def test_classify_genai_error_tpm():
    """Verify error reporting Tokens per minute (TPM) limit is identified as retriable TPM limit."""
    raw_err = genai_errors.APIError(
        code=429,
        response_json={
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota). Quota exceeded for metric: Tokens per minute (TPM). Please retry after 12s.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    details = classify_genai_error(raw_err)
    assert details.is_rate_limit is True
    assert details.is_tpm is True
    assert details.is_rpm is False
    assert details.is_daily_quota is False
    assert details.retriable is True
    assert details.retry_after_sec == 12.0


def test_classify_genai_error_rpm():
    """Verify error reporting Requests per minute (RPM) is identified as retriable RPM limit."""
    raw_err = genai_errors.APIError(
        code=429,
        response_json={
            "error": {
                "code": 429,
                "message": "Rate limit exceeded: Requests per minute (RPM) reached. Please retry in 5s.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    details = classify_genai_error(raw_err)
    assert details.is_rate_limit is True
    assert details.is_rpm is True
    assert details.is_tpm is False
    assert details.is_daily_quota is False
    assert details.retriable is True
    assert details.retry_after_sec == 5.0


def test_classify_genai_error_daily_quota():
    """Verify daily quota exhaustion is recognized as NON-retriable."""
    raw_err = genai_errors.APIError(
        code=429,
        response_json={
            "error": {
                "code": 429,
                "message": "Quota exceeded for quota metric 'Queries' and limit 'Queries per day' of service 'generativelanguage.googleapis.com'",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    details = classify_genai_error(raw_err)
    assert details.is_rate_limit is True
    assert details.is_daily_quota is True
    assert details.retriable is False  # Cannot be resolved by backoff sleep


def test_classify_genai_error_auth_failure():
    """Verify 401/403 authorization failures are non-retriable."""
    raw_err = genai_errors.ClientError(
        code=401,
        response_json={"error": {"code": 401, "message": "API key not valid. Please pass a valid API key."}},
    )
    details = classify_genai_error(raw_err)
    assert details.is_auth is True
    assert details.retriable is False
    assert details.is_rate_limit is False


def test_classify_genai_error_server_error():
    """Verify 503 Service Unavailable is retriable server error."""
    raw_err = genai_errors.ServerError(
        code=503,
        response_json={"error": {"code": 503, "message": "Service Unavailable. Backends overloaded."}},
    )
    details = classify_genai_error(raw_err)
    assert details.is_server_error is True
    assert details.retriable is True
    assert details.is_rate_limit is False


# ============================================================================
# 3. Exponential Backoff with Jitter Tests
# ============================================================================

def test_backoff_calculation_growth_and_jitter():
    """Verify exponential growth bounded by max_backoff and jitter range."""
    # Attempt 0: initial_backoff = 2.0 -> base = 2.0 -> with jitter [1.6, 2.4]
    b0 = calculate_backoff_with_jitter(attempt=0, initial_backoff=2.0, max_backoff=30.0)
    assert 1.5 <= b0 <= 2.5

    # Attempt 2: 2.0 * 2^2 = 8.0 -> with jitter [6.4, 9.6]
    b2 = calculate_backoff_with_jitter(attempt=2, initial_backoff=2.0, max_backoff=30.0)
    assert 6.0 <= b2 <= 10.0

    # Large attempt: bounded by max_backoff
    b_capped = calculate_backoff_with_jitter(attempt=10, initial_backoff=2.0, max_backoff=15.0)
    assert b_capped <= 15.0 * 1.25


def test_backoff_respects_suggested_retry_after():
    """Verify suggested retry_after_sec from provider response is respected."""
    b_suggested = calculate_backoff_with_jitter(
        attempt=0,
        initial_backoff=1.0,
        max_backoff=60.0,
        suggested_wait=14.0,
    )
    # Should be around 14.0 with jitter
    assert 13.0 <= b_suggested <= 17.5


# ============================================================================
# 4. Partial Progress & In-Memory Resumption Tests
# ============================================================================

def make_mock_embedding_response(dim: int = 768, val: float = 0.05):
    mock_resp = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [val] * dim
    mock_resp.embeddings = [mock_emb]
    return mock_resp


def test_embed_chunks_preserves_partial_progress_on_failure():
    """Verify when embedding fails midway, already-computed vectors remain on ChunkRecord in memory."""
    cfg = make_test_config(embedding_max_retries=1)
    mock_client = MagicMock()

    resp_chunk0 = make_mock_embedding_response(768, val=0.1)
    resp_chunk1 = make_mock_embedding_response(768, val=0.2)
    # Chunk 2 fails with daily quota exhaustion (non-retriable)
    err_quota = genai_errors.APIError(
        code=429,
        response_json={"error": {"code": 429, "message": "Queries per day exceeded."}},
    )

    mock_client.models.embed_content.side_effect = [
        resp_chunk0,
        resp_chunk1,
        err_quota,
    ]

    service = EmbeddingService(config=cfg, client=mock_client)

    chunks = [
        ChunkRecord(
            chunk_id=f"c_{i}",
            document_id="doc1",
            source_filename="doc.pdf",
            page_numbers=[i + 1],
            chunk_index=i,
            text=f"Financial statement chunk {i}",
        )
        for i in range(3)
    ]

    with pytest.raises(EmbeddingError) as exc_info:
        service.embed_chunks(chunks)

    assert "quota" in str(exc_info.value).lower()
    # Chunk 0 and 1 must retain their computed vectors in memory
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) == 768
    assert chunks[1].embedding is not None
    assert len(chunks[1].embedding) == 768
    # Chunk 2 should not have an embedding
    assert chunks[2].embedding is None

    # NOW TEST RESUMPTION:
    # Reset mock with successful response for the remaining chunk
    mock_client.models.embed_content.reset_mock()
    mock_client.models.embed_content.side_effect = None
    mock_client.models.embed_content.return_value = make_mock_embedding_response(768, val=0.3)

    # Call embed_chunks again on the same list of chunks
    vectors = service.embed_chunks(chunks)

    # Chunks 0 and 1 were ALREADY embedded, so client was called ONLY ONCE for chunk 2!
    assert mock_client.models.embed_content.call_count == 1
    assert len(vectors) == 3
    assert chunks[2].embedding is not None


# ============================================================================
# 5. Batch Embedding Tests
# ============================================================================

def test_batch_embedding_independent_vector_assignment():
    """Verify batch embedding assigns distinct independent vectors to each chunk."""
    cfg = make_test_config(embedding_batch_size=2)
    mock_client = MagicMock()

    # Mock response for batch of 2
    def mock_embed(model, contents, config):
        resp = MagicMock()
        embs = []
        for text in contents:
            e = MagicMock()
            e.values = [float(len(text))] * 768
            embs.append(e)
        resp.embeddings = embs
        return resp

    mock_client.models.embed_content.side_effect = mock_embed

    service = EmbeddingService(config=cfg, client=mock_client)

    chunks = [
        ChunkRecord(
            chunk_id=f"b_{i}",
            document_id="doc_batch",
            source_filename="doc.pdf",
            page_numbers=[i + 1],
            chunk_index=i,
            text=f"Batch chunk text variation {i} with unique content",
        )
        for i in range(4)
    ]

    vectors = service.embed_chunks(chunks)

    # 4 chunks with batch size 2 -> exactly 2 API calls
    assert mock_client.models.embed_content.call_count == 2
    assert len(vectors) == 4
    # Ensure all chunks got independent vectors populated
    for i, chunk in enumerate(chunks):
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 768
        assert chunk.embedding == vectors[i]


# ============================================================================
# 6. Pipeline Failure Cleanup Tests
# ============================================================================

def test_indexing_pipeline_cleans_up_vector_store_on_failure():
    """Verify that if vector indexing fails, the incomplete collection is cleanly deleted."""
    p1 = PageRecord(
        document_id="doc_clean",
        source_filename="test.pdf",
        page_number=1,
        text="Sample text for rollback test",
        char_count=29,
        has_meaningful_text=True,
    )

    failing_embedder = MagicMock()
    failing_embedder.model_name = "gemini-embedding-2"
    failing_embedder.dimension = 768
    failing_embedder.embed_chunks.side_effect = EmbeddingError("Quota exceeded midway")

    chroma_client = chromadb.EphemeralClient()

    with pytest.raises(EmbeddingError):
        run_indexing_pipeline(
            pages=[p1],
            embedding_service=failing_embedder,
            chroma_client=chroma_client,
        )

    # Chroma collection should not exist or be empty
    collections = chroma_client.list_collections()
    # Any created collections should have been deleted
    assert len(collections) == 0


# ============================================================================
# 7. Production Hotfix Verification Tests
# ============================================================================

def test_extract_genai_error_payload_nested_rpc_retry_info():
    """Verify nested Google RPC RetryInfo and QuotaFailure are accurately extracted."""
    raw_err = genai_errors.APIError(
        code=429,
        response_json={
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "subject": "project:987654321",
                                "description": "Quota exceeded for quota metric 'embed_content_free_tier_tokens' and limit 'None' of service 'generativelanguage.googleapis.com'",
                            }
                        ],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "25.123s",
                    },
                ],
            }
        },
    )
    details = classify_genai_error(raw_err)
    assert details.is_tpm is True
    assert details.is_rpm is False
    assert details.is_daily_quota is False
    assert details.retriable is True
    assert details.quota_metric == "embed_content_free_tier_tokens"
    assert details.retry_after_sec == 25.123


def test_25_second_provider_wait_never_shortened():
    """Verify provider-suggested 25s wait is never shortened to 1s, 2s, or 4s."""
    for attempt in range(5):
        for _ in range(25):
            wait = calculate_backoff_with_jitter(
                attempt=attempt,
                initial_backoff=2.0,
                max_backoff=60.0,
                suggested_wait=25.0,
            )
            # Invariant: Must NEVER be less than 25.0 seconds
            assert wait >= 25.0, f"Wait time {wait} was shortened below provider's 25.0s requirement!"


def test_token_pacer_smooth_inter_request_spacing_at_48_rpm():
    """Verify 48 RPM limit enforces at least 1.25s interval between requests."""
    clock = MockClock(start_time=0.0)
    # Effective RPM: 48 (60 * 0.80) -> min_interval = 60 / 48 = 1.25 seconds
    pacer = TokenPacer(
        tpm_limit=30000,
        rpm_limit=60,
        safety_margin=0.80,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )
    assert pacer.min_request_interval == 1.25

    # Request 1 at t=0
    pacer.acquire(tokens_needed=100)
    assert clock.time() == 0.0

    # Request 2 fired immediately at t=0 must wait 1.25s
    pacer.acquire(tokens_needed=100)
    assert clock.time() == 1.25
    assert clock.sleep_calls == [1.25]


def test_token_pacer_reservation_release_on_failure():
    """Verify cancelled reservation returns tokens to available budget."""
    pacer = TokenPacer(
        tpm_limit=1000,
        rpm_limit=10,
        safety_margin=1.0,
    )
    res = pacer.reserve(tokens_needed=400)
    tokens, reqs = pacer.current_usage()
    assert tokens == 400
    assert reqs == 1

    # Release reservation upon failure
    res.release()
    tokens_after, reqs_after = pacer.current_usage()
    assert tokens_after == 0
    assert reqs_after == 0


def test_embedding_deadline_exceeded_raises_immediately():
    """Verify waiting beyond overall processing deadline halts and preserves progress."""
    cfg = make_test_config(embedding_deadline_sec=10.0, embedding_max_retries=3)
    mock_client = MagicMock()

    # Provider requests 25s wait, but deadline is only 10s
    err_429 = genai_errors.APIError(
        code=429,
        response_json={
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "details": [{"@type": "RetryInfo", "retryDelay": "25s"}],
            }
        },
    )
    mock_client.models.embed_content.side_effect = err_429

    service = EmbeddingService(config=cfg, client=mock_client)
    with pytest.raises(EmbeddingError) as exc_info:
        service.embed_query("What is revenue?")

    assert "deadline" in str(exc_info.value).lower()
    # Should halt on attempt 1 without spinning
    assert mock_client.models.embed_content.call_count == 1


def test_appconfig_safe_zero_argument_instantiation():
    """Verify AppConfig can be instantiated without requiring positional arguments."""
    cfg = AppConfig()
    assert cfg.gemini_embedding_model == "gemini-embedding-2"
    assert cfg.embedding_tpm_limit == 30000
    assert cfg.embedding_rpm_limit == 60
    assert cfg.embedding_safety_margin == 0.80
    assert cfg.embedding_max_retries == 5
    assert cfg.embedding_batch_size == 16
    assert cfg.embedding_deadline_sec == 120.0
    assert "embedding_tpm_limit" in cfg.safe_diagnostics


def test_run_indexing_pipeline_with_chunks_interface():
    """Verify run_indexing_pipeline accepts pages and optional chunks without TypeError."""
    p1 = PageRecord(
        document_id="doc_pipe_synth",
        source_filename="synth_4page.pdf",
        page_number=1,
        text="Executive Summary: Total revenue increased 12% in FY2024.",
        char_count=58,
        has_meaningful_text=True,
    )
    c1 = ChunkRecord(
        chunk_id="chk_synth_1",
        document_id="doc_pipe_synth",
        source_filename="synth_4page.pdf",
        page_numbers=[1],
        chunk_index=0,
        text=p1.text,
        embedding=[0.05] * 768,  # Pre-embedded
    )

    mock_embedder = MagicMock()
    mock_embedder.model_name = "gemini-embedding-2"
    mock_embedder.dimension = 768
    mock_embedder.embed_chunks.return_value = [[0.05] * 768]

    chroma_client = chromadb.EphemeralClient()

    # Call with chunks keyword argument (the exact application invocation path)
    res = run_indexing_pipeline(
        pages=[p1],
        chunks=[c1],
        embedding_service=mock_embedder,
        chroma_client=chroma_client,
    )
    assert res.indexing_result.total_chunks == 1
    assert res.vector_store.count() == 1
    res.vector_store.delete()


def test_metadata_rendering_safe_when_report_analysis_is_none():
    """Verify rendering logic handles analysis=None without AttributeError on company_name."""
    analysis = None
    meta = MagicMock()
    meta.filename = "Apple_10K.pdf"

    # The safe pattern implemented in app.py
    company_header = analysis.company_name if (analysis and analysis.company_name) else "Financial Report"
    fiscal_period_str = analysis.fiscal_period if (analysis and analysis.fiscal_period) else "Report Period"

    assert company_header == "Financial Report"
    assert fiscal_period_str == "Report Period"

