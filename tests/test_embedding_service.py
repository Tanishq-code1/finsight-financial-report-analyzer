"""Unit tests for Google GenAI embedding service with mocked SDK client."""

from unittest.mock import MagicMock
import pytest
from google.genai import errors as genai_errors

from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.errors import ConfigurationError, EmbeddingError
from financial_report_analyzer.schemas import ChunkRecord
from financial_report_analyzer.embedding_service import (
    EmbeddingService,
    format_document_text,
    format_query_text,
    validate_embedding_vector,
)


def make_mock_response(dimension: int = 768) -> MagicMock:
    """Helper creating a mock Google GenAI embed_content response."""
    resp = MagicMock()
    embedding_item = MagicMock()
    embedding_item.values = [0.01 * (i % 10) for i in range(dimension)]
    resp.embeddings = [embedding_item]
    return resp


def make_test_config(has_key: bool = True, embedding_batch_size: int = 1) -> AppConfig:
    """Helper creating test AppConfig."""
    return AppConfig(
        gemini_api_key="test-secret-key" if has_key else None,
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
        embedding_batch_size=embedding_batch_size,
    )


def test_formatting_helpers():
    """Verify asymmetric retrieval prefix formatting."""
    doc_fmt = format_document_text("Apple 10-K", "Revenue was $383B.")
    assert doc_fmt == "title: Apple 10-K | text: Revenue was $383B."

    query_fmt = format_query_text("What was the annual revenue?")
    assert query_fmt == "task: question answering | query: What was the annual revenue?"


def test_validate_embedding_vector():
    """Verify validation of numeric, finite, dimension-matching vectors."""
    valid_vec = [0.1] * 768
    validate_embedding_vector(valid_vec, 768)

    # Wrong dimension
    with pytest.raises(EmbeddingError) as exc_dim:
        validate_embedding_vector([0.1] * 512, 768)
    assert "dimension mismatch" in str(exc_dim.value)

    # Non-finite value
    with pytest.raises(EmbeddingError) as exc_nan:
        validate_embedding_vector([float("nan")] * 768, 768)
    assert "non-finite value" in str(exc_nan.value)


def test_missing_credentials_raises_configuration_error():
    """Verify service without API key raises clean ConfigurationError on request."""
    cfg = make_test_config(has_key=False)
    service = EmbeddingService(config=cfg)
    assert service.is_configured is False

    with pytest.raises(ConfigurationError) as exc_info:
        service.embed_query("Sample question")
    assert "API key is not configured" in str(exc_info.value)


def test_embed_chunk_calls_sdk_correctly():
    """Verify embed_chunk calls client.models.embed_content with exact parameters."""
    cfg = make_test_config()
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = make_mock_response(768)

    service = EmbeddingService(config=cfg, client=mock_client)
    vector = service.embed_chunk(text="Cash was $29B", document_title="10K.pdf")

    assert len(vector) == 768
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs

    assert call_kwargs["model"] == "gemini-embedding-2"
    assert call_kwargs["contents"] == "title: 10K.pdf | text: Cash was $29B"
    assert call_kwargs["config"] == {"output_dimensionality": 768}


def test_embed_query_calls_sdk_correctly():
    """Verify embed_query formats query with question answering task prefix."""
    cfg = make_test_config()
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = make_mock_response(768)

    service = EmbeddingService(config=cfg, client=mock_client)
    vector = service.embed_query("What is net income?")

    assert len(vector) == 768
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["contents"] == "task: question answering | query: What is net income?"


def test_embed_chunks_individual_calls_and_callback():
    """Verify each chunk is embedded in its own request (no input aggregation)."""
    cfg = make_test_config()
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = make_mock_response(768)

    service = EmbeddingService(config=cfg, client=mock_client)
    chunks = [
        ChunkRecord(
            chunk_id=f"c_{i}",
            document_id="doc1",
            source_filename="doc.pdf",
            page_numbers=[i + 1],
            chunk_index=i,
            text=f"Content for chunk {i}",
        )
        for i in range(3)
    ]

    progress_calls = []

    def on_progress(cur, total):
        progress_calls.append((cur, total))

    vectors = service.embed_chunks(chunks, progress_callback=on_progress)

    # 3 chunks => exactly 3 individual SDK calls
    assert mock_client.models.embed_content.call_count == 3
    assert len(vectors) == 3
    assert all(len(v) == 768 for v in vectors)
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]
    # Ensure chunk object had embedding field populated
    assert chunks[0].embedding == vectors[0]


def test_retriable_error_retries_and_succeeds():
    """Verify transient 429 rate limit error retries and succeeds."""
    cfg = make_test_config()
    mock_client = MagicMock()

    # Create a 429 APIError
    err_429 = genai_errors.APIError(code=429, response_json={"error": {"message": "Rate limit exceeded"}})

    mock_client.models.embed_content.side_effect = [
        err_429,
        make_mock_response(768),
    ]

    service = EmbeddingService(config=cfg, client=mock_client, initial_backoff=0.01)
    vector = service.embed_query("Test question")

    assert len(vector) == 768
    assert mock_client.models.embed_content.call_count == 2


def test_permanent_error_fails_immediately():
    """Verify permanent 401 Unauthorized error does not enter retry loop."""
    cfg = make_test_config()
    mock_client = MagicMock()

    err_401 = genai_errors.ClientError(code=401, response_json={"error": {"message": "Invalid API Key"}})
    mock_client.models.embed_content.side_effect = err_401

    service = EmbeddingService(config=cfg, client=mock_client, max_retries=3)
    with pytest.raises(EmbeddingError) as exc_info:
        service.embed_query("Test question")

    assert "401" in str(exc_info.value)
    # Called only once, no retries!
    assert mock_client.models.embed_content.call_count == 1


@pytest.mark.skipif(
    not bool(pytest.importorskip("os").getenv("RUN_LIVE_GEMINI_TESTS")),
    reason="Live Gemini API tests are opt-in. Set RUN_LIVE_GEMINI_TESTS=1 and GEMINI_API_KEY to execute.",
)
def test_live_gemini_embedding_smoke():
    """Opt-in live integration test verifying real connection to Google GenAI gemini-embedding-2."""
    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable is required for live smoke test.")

    service = EmbeddingService(config=load_config())
    vector = service.embed_query("What was annual consolidated revenue?")
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)
