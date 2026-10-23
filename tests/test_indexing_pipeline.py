"""Unit tests for the indexing orchestration pipeline."""

from unittest.mock import MagicMock
import pytest
import chromadb

from financial_report_analyzer.config import AppConfig
from financial_report_analyzer.schemas import PageRecord
from financial_report_analyzer.indexing_pipeline import run_indexing_pipeline
from financial_report_analyzer.errors import EmbeddingError, VectorStoreError


def make_page(page_num: int, text: str) -> PageRecord:
    return PageRecord(
        document_id="doc_pipe",
        source_filename="statement.pdf",
        page_number=page_num,
        text=text,
        char_count=len(text),
        has_meaningful_text=True,
    )


class FakeEmbeddingService:
    """Mock embedder providing deterministic numeric vectors without API calls."""

    def __init__(self, dimension: int = 768) -> None:
        self.model_name = "gemini-embedding-2"
        self.dimension = dimension
        self.is_configured = True

    def embed_chunk(self, text: str, document_title: str) -> list:
        # Deterministic vector based on text length
        val = (len(text) % 100) / 100.0
        return [val] * self.dimension

    def embed_query(self, query: str) -> list:
        val = (len(query) % 100) / 100.0
        return [val] * self.dimension

    def embed_chunks(self, chunks, progress_callback=None) -> list:
        vecs = []
        for idx, c in enumerate(chunks):
            v = self.embed_chunk(c.text, c.source_filename)
            c.embedding = v
            vecs.append(v)
            if progress_callback:
                progress_callback(idx + 1, len(chunks))
        return vecs


def test_indexing_pipeline_success():
    """Verify offline end-to-end execution of run_indexing_pipeline with fake embedder."""
    p1 = make_page(1, "Management Discussion: Revenue was $500M in FY2024.")
    p2 = make_page(2, "Risk Factors: Foreign exchange volatility remains elevated.")

    chroma_client = chromadb.EphemeralClient()
    fake_embedder = FakeEmbeddingService(dimension=768)
    progress_updates = []

    def on_progress(desc, frac):
        progress_updates.append((desc, round(frac, 2)))

    res = run_indexing_pipeline(
        pages=[p1, p2],
        embedding_service=fake_embedder,
        chroma_client=chroma_client,
        progress_callback=on_progress,
    )

    assert res.indexing_result.total_pages == 2
    assert res.indexing_result.total_chunks == 2
    assert res.indexing_result.total_embeddings == 2
    assert res.indexing_result.embedding_model == "gemini-embedding-2"
    assert res.indexing_result.embedding_dimension == 768
    assert res.vector_store.count() == 2
    assert len(progress_updates) > 0

    # Test query against indexed store
    q_vec = fake_embedder.embed_query("Revenue inquiry")
    retrieved = res.vector_store.query(q_vec, top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].page_number in [1, 2]

    res.vector_store.delete()


def test_indexing_pipeline_empty_pages():
    """Verify pipeline handles 0 pages safely without crashing."""
    chroma_client = chromadb.EphemeralClient()
    res = run_indexing_pipeline(
        pages=[],
        chroma_client=chroma_client,
    )
    assert res.indexing_result.total_pages == 0
    assert res.indexing_result.total_chunks == 0
    assert res.vector_store.count() == 0


def test_indexing_pipeline_embedding_failure_aborts():
    """Verify failure during embedding generation raises error and prevents indexing."""
    p1 = make_page(1, "Sample page text for testing pipeline embedding failure handling.")
    failing_embedder = MagicMock()
    failing_embedder.embed_chunks.side_effect = EmbeddingError("API connection timeout")
    failing_embedder.model_name = "gemini-embedding-2"
    failing_embedder.dimension = 768

    chroma_client = chromadb.EphemeralClient()

    with pytest.raises(EmbeddingError) as exc_info:
        run_indexing_pipeline(
            pages=[p1],
            embedding_service=failing_embedder,
            chroma_client=chroma_client,
        )
    assert "timeout" in str(exc_info.value)


def test_synthetic_4page_pdf_app_invocation_path():
    """Regression test: verify 4-page synthetic PDF proceeds through the exact app invocation path.

    Steps:
    1. Parse PDF using load_pdf()
    2. Generate sentence-aware ChunkRecord objects using chunk_pages()
    3. Pass both pages and chunks to run_indexing_pipeline() with kwargs matching app.py
    4. Verify no TypeError ('unexpected keyword argument') occurs and vector index is populated.
    """
    from financial_report_analyzer.pdf_loader import load_pdf
    from financial_report_analyzer.chunker import chunk_pages, ChunkerConfig
    from evals.fixtures import create_synthetic_apex_10k_pdf

    pdf_bytes = create_synthetic_apex_10k_pdf()
    config = AppConfig()

    # Step 1: PDF Extraction (app.py:467)
    load_result = load_pdf(pdf_bytes, "Apex_Global_Technologies_FY2024_10K.pdf", config=config)
    assert load_result.metadata.total_pages == 4

    # Step 2: Chunk creation (app.py:480)
    chunker_cfg = ChunkerConfig(
        target_tokens=config.chunk_size_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    chunks = chunk_pages(load_result.pages, config=chunker_cfg)
    assert len(chunks) > 0

    # Step 3: Exact application invocation call (app.py:495)
    progress_updates = []

    def update_progress(desc: str, frac: float) -> None:
        progress_updates.append((desc, frac))

    chroma_client = chromadb.EphemeralClient()
    fake_embedder = FakeEmbeddingService(dimension=768)

    pipeline_output = run_indexing_pipeline(
        pages=load_result.pages,
        chunks=chunks,
        config=config,
        embedding_service=fake_embedder,
        chroma_client=chroma_client,
        progress_callback=update_progress,
    )

    assert pipeline_output.indexing_result.total_pages == 4
    assert pipeline_output.indexing_result.total_chunks == len(chunks)
    assert pipeline_output.vector_store.count() == len(chunks)
    assert len(progress_updates) > 0

    pipeline_output.vector_store.delete()

