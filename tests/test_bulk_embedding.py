"""Unit tests verifying dynamic bulk batch embedding, SDK batching semantics, and fallback mechanisms."""

from unittest.mock import MagicMock
import pytest
from google.genai import types as genai_types
from google.genai._transformers import t_contents
from google.genai.models import _EmbedContentParametersPrivate_to_mldev
from google.genai._api_client import BaseApiClient

from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.schemas import ChunkRecord
from financial_report_analyzer.embedding_service import EmbeddingService, BatchContent


def make_mock_bulk_response(count: int, dimension: int = 768) -> MagicMock:
    resp = MagicMock()
    items = []
    for c_i in range(count):
        item = MagicMock()
        item.values = [float(c_i + 1) * 0.01 + float(j) * 0.0001 for j in range(dimension)]
        items.append(item)
    resp.embeddings = items
    return resp


def test_bulk_embedding_batches_chunks_and_preserves_1to1_vectors():
    """Verify embed_chunks groups requests into bulk batches and maps independent vectors 1:1."""
    cfg = AppConfig(embedding_batch_size=4, embedding_dimension=768)
    mock_client = MagicMock()
    # 6 chunks with batch_size=4 => Batch 1 has 4, Batch 2 has 2
    mock_client.models.embed_content.side_effect = [
        make_mock_bulk_response(4, 768),
        make_mock_bulk_response(2, 768),
    ]

    service = EmbeddingService(config=cfg, client=mock_client)
    chunks = [
        ChunkRecord(chunk_id=f"c_{i}", document_id="doc1", source_filename="10k.pdf", page_numbers=[i + 1], chunk_index=i, text=f"Chunk text content {i}")
        for i in range(6)
    ]

    vectors = service.embed_chunks(chunks)

    assert len(vectors) == 6
    assert mock_client.models.embed_content.call_count == 2
    # Verify 1:1 assignment to chunk objects
    for i, c in enumerate(chunks):
        assert c.embedding is not None
        assert len(c.embedding) == 768
        assert c.embedding == vectors[i]


def test_bulk_embedding_fallback_to_individual_on_mismatch():
    """Verify embed_chunks seamlessly falls back to 1-by-1 embedding if provider returns fewer vectors."""
    cfg = AppConfig(embedding_batch_size=3, embedding_dimension=768)
    mock_client = MagicMock()

    # Bulk call returns 1 vector instead of 3, then 1-by-1 calls return 1 vector each
    mock_client.models.embed_content.side_effect = [
        make_mock_bulk_response(1, 768),  # Batch response with length mismatch
        make_mock_bulk_response(1, 768),  # Fallback chunk 0
        make_mock_bulk_response(1, 768),  # Fallback chunk 1
        make_mock_bulk_response(1, 768),  # Fallback chunk 2
    ]

    service = EmbeddingService(config=cfg, client=mock_client)
    chunks = [
        ChunkRecord(chunk_id=f"c_{i}", document_id="doc1", source_filename="10k.pdf", page_numbers=[i + 1], chunk_index=i, text=f"Text {i}")
        for i in range(3)
    ]

    vectors = service.embed_chunks(chunks)
    assert len(vectors) == 3
    assert all(len(v) == 768 for v in vectors)
    # Total calls: 1 attempted batch + 3 fallback calls
    assert mock_client.models.embed_content.call_count == 4
    # Telemetry should reflect the mismatch and fallback
    assert service.stats["length_mismatches"] == 1
    assert service.stats["fallbacks_to_individual"] == 1


def test_batch_content_serialization_semantics():
    """Verify BatchContent objects serialize to distinct requests in MLDev payload instead of 1 aggregated request."""
    # 1. Plain strings merge into 1 Content object with multiple parts (causing provider to return 1 vector)
    plain_strings = [f"Text chunk number {i}" for i in range(16)]
    plain_transformed = t_contents(plain_strings)
    assert len(plain_transformed) == 1, "Plain strings in SDK are collapsed into 1 Content with multiple parts"

    # 2. BatchContent instances preserve distinct Content objects
    batch_contents = [BatchContent(s) for s in plain_strings]
    batch_transformed = t_contents(batch_contents)
    assert len(batch_transformed) == 16, "BatchContent must preserve 16 individual Content objects"

    # 3. MLDev payload serialization yields 16 distinct sub-requests in batchEmbedContents
    param_obj = genai_types._EmbedContentParametersPrivate(
        model="models/gemini-embedding-2",
        contents=batch_transformed,
        config={"output_dimensionality": 768},
    )
    fake_client = BaseApiClient(api_key="mock-key")
    mldev_payload = _EmbedContentParametersPrivate_to_mldev(fake_client, param_obj)
    assert "requests" in mldev_payload
    assert len(mldev_payload["requests"]) == 16, "MLDev serialization must produce 16 sub-requests in batchEmbedContents"


def test_bulk_embedding_with_16_distinct_chunks_and_telemetry():
    """Isolated integration test: 16 distinct chunks with batch_size=16 make 1 call, return 16 independent vectors."""
    cfg = AppConfig(embedding_batch_size=16, embedding_dimension=768)
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = make_mock_bulk_response(16, 768)

    service = EmbeddingService(config=cfg, client=mock_client)
    chunks = [
        ChunkRecord(
            chunk_id=f"chunk_page_{i+1}_{i}",
            document_id="doc_10k",
            source_filename="alphabet_10k.pdf",
            page_numbers=[i + 1],
            chunk_index=i,
            text=f"Alphabet Inc. 10-K section {i} financial disclosures with unique metrics ${100 + i}M.",
        )
        for i in range(16)
    ]

    progress_events = []

    def on_progress(cur, total, b_idx, b_total):
        progress_events.append((cur, total, b_idx, b_total))

    vectors = service.embed_chunks(chunks, progress_callback=on_progress)

    # Exactly 1 bulk API request dispatched
    assert mock_client.models.embed_content.call_count == 1
    assert len(vectors) == 16
    assert len(progress_events) == 1
    assert progress_events[0] == (16, 16, 1, 1)

    # Check contents passed to client
    called_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert called_kwargs["model"] == "gemini-embedding-2"
    called_contents = called_kwargs["contents"]
    assert len(called_contents) == 16
    assert all(isinstance(c, genai_types.Content) for c in called_contents)

    # Check 1:1 mapping and distinctness
    for i, c in enumerate(chunks):
        assert c.embedding is not None
        assert len(c.embedding) == 768
        assert c.embedding == vectors[i]
        # Vectors must be independent and distinct
        if i > 0:
            assert c.embedding != chunks[i - 1].embedding

    # Check telemetry
    assert service.stats["api_requests"] == 1
    assert service.stats["bulk_batches"] == 1
    assert service.stats["individual_requests"] == 0
    assert service.stats["vectors_generated"] == 16
    assert service.stats["length_mismatches"] == 0
    assert service.stats["fallbacks_to_individual"] == 0
    assert service.stats["batch_sizes"][16] == 1
    assert service.stats["total_api_latency_sec"] >= 0.0
    assert service.stats["total_validation_sec"] >= 0.0
