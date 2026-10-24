"""Unit tests verifying multi-tier in-memory caching and document fingerprinting."""

import pytest
import concurrent.futures
from financial_report_analyzer.cache import PerformanceCache
from financial_report_analyzer.pdf_loader import PDFIngestionResult
from financial_report_analyzer.schemas import (
    PageRecord,
    ChunkRecord,
    DocumentMetadata,
    BatchExtractionResult,
    FinancialMetric,
)


def test_performance_cache_pdf_ingestion():
    cache = PerformanceCache()
    meta = DocumentMetadata(
        document_id="doc_cache_test",
        filename="test.pdf",
        total_pages=2,
        meaningful_pages=2,
        file_size_bytes=1024,
        sha256_hash="hash123",
    )
    pages = [
        PageRecord(document_id="doc_cache_test", source_filename="test.pdf", page_number=1, text="Text 1", char_count=6, has_meaningful_text=True),
        PageRecord(document_id="doc_cache_test", source_filename="test.pdf", page_number=2, text="Text 2", char_count=6, has_meaningful_text=True),
    ]
    res = PDFIngestionResult(
        metadata=meta,
        pages=pages,
        total_characters=12,
        pages_with_text=2,
        is_sparse=False,
    )

    # Miss
    assert cache.get_ingestion("hash123") is None

    # Put and Hit
    cache.put_ingestion("hash123", res)
    hit = cache.get_ingestion("hash123")
    assert hit is not None
    hit_res, hit_pages = hit
    assert hit_res.metadata.document_id == "doc_cache_test"
    assert len(hit_pages) == 2


def test_performance_cache_chunks_and_embeddings():
    cache = PerformanceCache()
    chunks = [
        ChunkRecord(chunk_id="c1", document_id="doc1", source_filename="doc.pdf", page_numbers=[1], chunk_index=0, text="Revenue was $10M"),
        ChunkRecord(chunk_id="c2", document_id="doc1", source_filename="doc.pdf", page_numbers=[2], chunk_index=1, text="Net income was $2M"),
    ]

    # Chunk caching
    cache.put_chunks("doc1", "cfg_key_1", chunks)
    hit_chunks = cache.get_chunks("doc1", "cfg_key_1")
    assert hit_chunks is not None
    assert len(hit_chunks) == 2

    # Embedding caching (single and bulk)
    k1 = cache.make_embedding_key("Revenue was $10M", "doc.pdf", "gemini-embedding-2", 768)
    k2 = cache.make_embedding_key("Net income was $2M", "doc.pdf", "gemini-embedding-2", 768)
    vec1 = [0.1] * 768
    vec2 = [0.2] * 768

    cache.put_embeddings_bulk([(k1, vec1), (k2, vec2)])
    assert cache.get_embedding(k1) == vec1
    assert cache.get_embedding(k2) == vec2

    bulk_hits, bulk_misses = cache.get_embeddings_bulk([k1, "missing_key", k2])
    assert bulk_hits == {0: vec1, 2: vec2}
    assert bulk_misses == [1]


def test_performance_cache_batch_extraction():
    cache = PerformanceCache()
    res = BatchExtractionResult(
        report_title="Test 10-K",
        company_name="Acme Corp",
        fiscal_period="FY2024",
        metrics=[
            FinancialMetric(name="Revenue", value=100.0, currency="USD", unit_scale="millions", source_page=1, source_quote="Revenue was $100M", is_verified=True)
        ],
        risks=[],
        outlook=[],
    )

    cache.put_batch_extraction("doc_hash_1", "b_1", "gemini-3.8-flash", res)
    hit = cache.get_batch_extraction("doc_hash_1", "b_1", "gemini-3.8-flash")
    assert hit is not None
    assert hit.company_name == "Acme Corp"
    assert len(hit.metrics) == 1


def test_performance_cache_thread_safety():
    cache = PerformanceCache()

    def worker(i: int):
        k = cache.make_embedding_key(f"Text {i}", "doc.pdf", "gemini-embedding-2", 768)
        cache.put_embedding(k, [float(i)] * 768)
        return cache.get_embedding(k)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(50)]
        for i, fut in enumerate(futures):
            vec = fut.result()
            assert len(vec) == 768
            assert vec[0] == float(i)

    stats = cache.get_stats()
    assert stats["embedding_vectors"] == 50
