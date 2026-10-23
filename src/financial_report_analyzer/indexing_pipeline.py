"""Orchestration pipeline for chunking, embedding, and vector indexing."""

import logging
from typing import List, Optional, Callable
from dataclasses import dataclass
from chromadb.api import ClientAPI

from financial_report_analyzer.config import AppConfig, load_config
from financial_report_analyzer.schemas import PageRecord, ChunkRecord, IndexingResult
from financial_report_analyzer.chunker import chunk_pages, ChunkerConfig
from financial_report_analyzer.embedding_service import EmbeddingService
from financial_report_analyzer.vector_store import VectorStore
from financial_report_analyzer.errors import EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingPipelineResult:
    """Complete output of the indexing orchestration pipeline."""

    indexing_result: IndexingResult
    vector_store: VectorStore
    chunks: List[ChunkRecord]


def run_indexing_pipeline(
    pages: List[PageRecord],
    config: Optional[AppConfig] = None,
    embedding_service: Optional[EmbeddingService] = None,
    chroma_client: Optional[ClientAPI] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    chunks: Optional[List[ChunkRecord]] = None,
    **kwargs,
) -> IndexingPipelineResult:
    """Execute end-to-end Phase 3 indexing pipeline for extracted report pages.

    Workflow:
    1. Sentence-aware chunking with 1-based page provenance (or reuses existing in-memory chunks).
    2. Vector embedding generation via EmbeddingService with token-aware pacing and partial preservation.
    3. Isolated ephemeral ChromaDB store creation.
    4. Indexing of chunks, vectors, and metadata.
    5. Safe cleanup if failures occur midway.

    Args:
        pages: Extracted PageRecord list from Phase 2.
        config: Application configuration.
        embedding_service: Configured EmbeddingService (instantiated if None).
        chroma_client: Optional Chroma ClientAPI (instantiates EphemeralClient if None).
        progress_callback: Optional callback receiving (step_description, fraction_complete).
        chunks: Optional pre-existing ChunkRecord list (preserves existing embeddings across retries).

    Returns:
        IndexingPipelineResult containing summary, active vector store, and chunk records.

    Raises:
        EmbeddingError: If vector generation fails.
        VectorStoreError: If indexing into ChromaDB fails.
    """
    app_cfg = config or load_config()

    if not pages:
        logger.warning("run_indexing_pipeline called with 0 pages; returning empty result")
        doc_id = "empty"
        empty_res = IndexingResult(
            document_id=doc_id,
            total_pages=0,
            total_chunks=0,
            total_embeddings=0,
            embedding_model=app_cfg.gemini_embedding_model,
            embedding_dimension=app_cfg.embedding_dimension,
            collection_name="col_empty",
        )
        empty_store = VectorStore(
            document_id=doc_id,
            client=chroma_client,
            embedding_model=app_cfg.gemini_embedding_model,
            expected_dimension=app_cfg.embedding_dimension,
        )
        return IndexingPipelineResult(indexing_result=empty_res, vector_store=empty_store, chunks=[])

    doc_id = pages[0].document_id

    # Step 1: Chunking (or reuse existing chunks from memory)
    if chunks is None:
        if progress_callback:
            progress_callback("Chunking report pages...", 0.1)

        chunker_cfg = ChunkerConfig(
            target_tokens=app_cfg.chunk_size_tokens,
            overlap_tokens=app_cfg.chunk_overlap_tokens,
        )
        chunks = chunk_pages(pages, config=chunker_cfg)
        logger.info("Generated %d chunks from %d pages for doc '%s'", len(chunks), len(pages), doc_id)
    else:
        logger.info("Reusing %d existing chunks in memory for doc '%s'", len(chunks), doc_id)

    if not chunks:
        logger.warning("No chunks were generated from the provided pages for doc '%s'", doc_id)
        empty_res = IndexingResult(
            document_id=doc_id,
            total_pages=len(pages),
            total_chunks=0,
            total_embeddings=0,
            embedding_model=app_cfg.gemini_embedding_model,
            embedding_dimension=app_cfg.embedding_dimension,
            collection_name=f"col_{doc_id}",
        )
        empty_store = VectorStore(
            document_id=doc_id,
            client=chroma_client,
            embedding_model=app_cfg.gemini_embedding_model,
            expected_dimension=app_cfg.embedding_dimension,
        )
        return IndexingPipelineResult(indexing_result=empty_res, vector_store=empty_store, chunks=[])

    # Step 2: Initialize Embedding Service
    embedder = embedding_service or EmbeddingService(config=app_cfg)

    # Step 3: Embed Chunks
    if progress_callback:
        progress_callback("Generating vector embeddings...", 0.3)

    def embed_progress(current: int, total: int) -> None:
        if progress_callback and total > 0:
            fraction = 0.3 + 0.4 * (current / total)
            progress_callback(f"Embedding chunk {current}/{total}...", fraction)

    try:
        embeddings = embedder.embed_chunks(chunks, progress_callback=embed_progress)
    except Exception as exc:
        logger.error("Failed to generate embeddings for document '%s': %s", doc_id, str(exc))
        raise

    # Step 4: Create Isolated Ephemeral Vector Store
    if progress_callback:
        progress_callback("Creating isolated vector index...", 0.8)

    store = VectorStore(
        document_id=doc_id,
        client=chroma_client,
        embedding_model=embedder.model_name,
        expected_dimension=embedder.dimension,
    )

    # Step 5: Index Chunks with rollback on failure
    try:
        if progress_callback:
            progress_callback("Indexing chunks into ChromaDB...", 0.9)
        store.add_chunks(chunks=chunks, embeddings=embeddings)
    except Exception as exc:
        logger.error("Indexing failed for document '%s'; rolling back collection: %s", doc_id, str(exc))
        store.delete()
        raise

    if progress_callback:
        progress_callback("Indexing complete!", 1.0)

    summary = IndexingResult(
        document_id=doc_id,
        total_pages=len(pages),
        total_chunks=len(chunks),
        total_embeddings=len(embeddings),
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
        collection_name=store.collection_name,
    )

    logger.info("Successfully indexed document '%s': %d chunks into '%s'", doc_id, len(chunks), store.collection_name)
    return IndexingPipelineResult(
        indexing_result=summary,
        vector_store=store,
        chunks=chunks,
    )
