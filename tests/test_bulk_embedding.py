"""Unit tests verifying dynamic bulk batch embedding and fallback mechanisms."""

from unittest.mock import MagicMock
import pytest
from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.schemas import ChunkRecord
from financial_report_analyzer.embedding_service import EmbeddingService


def make_mock_bulk_response(count: int, dimension: int = 768) -> MagicMock:
    resp = MagicMock()
    items = []
    for c_i in range(count):
        item = MagicMock()
        item.values = [float(c_i) * 0.01] * dimension
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
