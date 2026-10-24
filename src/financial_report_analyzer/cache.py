"""High-performance in-memory caching for PDF ingestion, chunking, embeddings, and extraction.

Implements thread-safe, bounded, content-addressable caching to eliminate redundant
file parsing, duplicate embedding requests, and repeated extraction batches.
"""

import hashlib
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from financial_report_analyzer.schemas import (
    PageRecord,
    ChunkRecord,
    BatchExtractionResult,
)

logger = logging.getLogger(__name__)

# Cache schema versions for deterministic invalidation
CACHE_PARSER_VERSION = "v1.1.0"
CACHE_CHUNKER_VERSION = "v1.1.0"
CACHE_EMBEDDING_VERSION = "v1.1.0"
CACHE_EXTRACTION_VERSION = "v1.1.0"

DEFAULT_MAX_CACHED_DOCUMENTS = 10
DEFAULT_MAX_CACHED_VECTORS = 10000
DEFAULT_MAX_CACHED_BATCHES = 100


def compute_content_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hexadecimal digest for raw binary content."""
    return hashlib.sha256(data).hexdigest()


def compute_text_sha256(text: str) -> str:
    """Compute standard SHA-256 hexadecimal digest for normalized text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PerformanceCache:
    """Thread-safe multi-tier in-memory cache for FinSight."""

    def __init__(
        self,
        max_documents: int = DEFAULT_MAX_CACHED_DOCUMENTS,
        max_vectors: int = DEFAULT_MAX_CACHED_VECTORS,
        max_batches: int = DEFAULT_MAX_CACHED_BATCHES,
    ) -> None:
        self._max_documents = max_documents
        self._max_vectors = max_vectors
        self._max_batches = max_batches
        self._lock = threading.RLock()

        # Cache tiers
        # doc_hash -> (parser_version, IngestionResult, pages)
        self._ingestion_cache: Dict[str, Tuple[str, Any, List[PageRecord]]] = {}
        # (doc_hash, chunk_cfg_key) -> (chunker_version, List[ChunkRecord])
        self._chunk_cache: Dict[Tuple[str, str], Tuple[str, List[ChunkRecord]]] = {}
        # vector_key -> vector (List[float])
        self._embedding_cache: Dict[str, List[float]] = {}
        # (doc_hash, batch_id, model_name) -> (extraction_version, BatchExtractionResult)
        self._batch_extraction_cache: Dict[Tuple[str, str, str], Tuple[str, BatchExtractionResult]] = {}

        # Telemetry counters
        self.stats = {
            "ingestion_hits": 0,
            "ingestion_misses": 0,
            "chunk_hits": 0,
            "chunk_misses": 0,
            "embedding_hits": 0,
            "embedding_misses": 0,
            "batch_hits": 0,
            "batch_misses": 0,
        }

    # -------------------------------------------------------------------------
    # PDF Ingestion Tier
    # -------------------------------------------------------------------------
    def get_ingestion(self, doc_hash: str) -> Optional[Tuple[Any, List[PageRecord]]]:
        """Retrieve cached PDF parsing result if parser version matches."""
        with self._lock:
            if doc_hash in self._ingestion_cache:
                version, ing_res, pages = self._ingestion_cache[doc_hash]
                if version == CACHE_PARSER_VERSION:
                    self.stats["ingestion_hits"] += 1
                    logger.debug("Ingestion cache HIT for doc %s", doc_hash[:12])
                    # Return defensive copies of pages
                    return ing_res, [p.model_copy() for p in pages]
            self.stats["ingestion_misses"] += 1
            return None

    def put_ingestion(self, doc_hash: str, ingestion_result: Any, pages: Optional[List[PageRecord]] = None) -> None:
        """Cache PDF parsing result with current parser version."""
        page_list = pages if pages is not None else getattr(ingestion_result, "pages", [])
        with self._lock:
            if len(self._ingestion_cache) >= self._max_documents:
                # Evict oldest entry
                oldest_key = next(iter(self._ingestion_cache))
                del self._ingestion_cache[oldest_key]
            self._ingestion_cache[doc_hash] = (
                CACHE_PARSER_VERSION,
                ingestion_result,
                [p.model_copy() for p in page_list],
            )
            logger.debug("Cached ingestion for doc %s", doc_hash[:12])

    # -------------------------------------------------------------------------
    # Chunking Tier
    # -------------------------------------------------------------------------
    def get_chunks(self, doc_hash: str, chunk_cfg_key: str) -> Optional[List[ChunkRecord]]:
        """Retrieve cached chunks if chunker version and config match."""
        with self._lock:
            key = (doc_hash, chunk_cfg_key)
            if key in self._chunk_cache:
                version, chunks = self._chunk_cache[key]
                if version == CACHE_CHUNKER_VERSION:
                    self.stats["chunk_hits"] += 1
                    logger.debug("Chunk cache HIT for doc %s", doc_hash[:12])
                    return [c.model_copy() for c in chunks]
            self.stats["chunk_misses"] += 1
            return None

    def put_chunks(self, doc_hash: str, chunk_cfg_key: str, chunks: List[ChunkRecord]) -> None:
        """Cache generated chunks."""
        with self._lock:
            key = (doc_hash, chunk_cfg_key)
            if len(self._chunk_cache) >= self._max_documents * 2:
                oldest_key = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest_key]
            self._chunk_cache[key] = (CACHE_CHUNKER_VERSION, [c.model_copy() for c in chunks])
            logger.debug("Cached %d chunks for doc %s", len(chunks), doc_hash[:12])

    # -------------------------------------------------------------------------
    # Embedding Vector Tier
    # -------------------------------------------------------------------------
    @staticmethod
    def make_embedding_key(text: str, document_title: str, model_name: str, dimension: int) -> str:
        """Generate deterministic cache key for chunk text and embedding parameters."""
        normalized_content = f"{document_title.strip()}|{text.strip()}"
        content_hash = compute_text_sha256(normalized_content)
        return f"{model_name}:{dimension}:{CACHE_EMBEDDING_VERSION}:{content_hash}"

    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Retrieve precomputed embedding vector by content key."""
        with self._lock:
            if key in self._embedding_cache:
                self.stats["embedding_hits"] += 1
                return list(self._embedding_cache[key])
            self.stats["embedding_misses"] += 1
            return None

    def put_embedding(self, key: str, vector: List[float]) -> None:
        """Store precomputed embedding vector."""
        with self._lock:
            if len(self._embedding_cache) >= self._max_vectors:
                # Evict first 100 items when capacity reached
                keys_to_evict = list(self._embedding_cache.keys())[:100]
                for k in keys_to_evict:
                    del self._embedding_cache[k]
            self._embedding_cache[key] = list(vector)

    def get_embeddings_bulk(self, keys: List[str]) -> Tuple[Dict[int, List[float]], List[int]]:
        """Bulk lookup for vectors.

        Returns:
            Tuple of (hit_dict_mapping_index_to_vector, list_of_miss_indices).
        """
        hits: Dict[int, List[float]] = {}
        misses: List[int] = []
        with self._lock:
            for idx, key in enumerate(keys):
                if key in self._embedding_cache:
                    self.stats["embedding_hits"] += 1
                    hits[idx] = list(self._embedding_cache[key])
                else:
                    self.stats["embedding_misses"] += 1
                    misses.append(idx)
        return hits, misses

    def put_embeddings_bulk(self, key_vector_pairs: List[Tuple[str, List[float]]]) -> None:
        """Bulk store precomputed vectors."""
        with self._lock:
            for key, vec in key_vector_pairs:
                if len(self._embedding_cache) >= self._max_vectors:
                    keys_to_evict = list(self._embedding_cache.keys())[:100]
                    for k in keys_to_evict:
                        del self._embedding_cache[k]
                self._embedding_cache[key] = list(vec)

    # -------------------------------------------------------------------------
    # Batch Extraction Tier
    # -------------------------------------------------------------------------
    def get_batch_extraction(
        self,
        doc_hash: str,
        batch_id: str,
        model_name: str,
    ) -> Optional[BatchExtractionResult]:
        """Retrieve previously successful batch extraction result."""
        with self._lock:
            key = (doc_hash, batch_id, model_name)
            if key in self._batch_extraction_cache:
                version, res = self._batch_extraction_cache[key]
                if version == CACHE_EXTRACTION_VERSION:
                    self.stats["batch_hits"] += 1
                    logger.debug("Batch extraction cache HIT for %s:%s", doc_hash[:8], batch_id)
                    return res.model_copy()
            self.stats["batch_misses"] += 1
            return None

    def put_batch_extraction(
        self,
        doc_hash: str,
        batch_id: str,
        model_name: str,
        result: BatchExtractionResult,
    ) -> None:
        """Cache verified batch extraction result."""
        with self._lock:
            key = (doc_hash, batch_id, model_name)
            if len(self._batch_extraction_cache) >= self._max_batches:
                oldest_key = next(iter(self._batch_extraction_cache))
                del self._batch_extraction_cache[oldest_key]
            self._batch_extraction_cache[key] = (CACHE_EXTRACTION_VERSION, result.model_copy())
            logger.debug("Cached batch extraction result for %s:%s", doc_hash[:8], batch_id)

    # -------------------------------------------------------------------------
    # Management & Reset
    # -------------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics for monitoring and diagnostics."""
        with self._lock:
            return {
                "pdf_ingestions": len(self._ingestion_cache),
                "chunk_sets": len(self._chunk_cache),
                "embedding_vectors": len(self._embedding_cache),
                "batch_extractions": len(self._batch_extraction_cache),
                **dict(self.stats),
            }

    def clear(self) -> None:
        """Clear all cache tiers and reset telemetry counters."""
        with self._lock:
            self._ingestion_cache.clear()
            self._chunk_cache.clear()
            self._embedding_cache.clear()
            self._batch_extraction_cache.clear()
            for k in self.stats:
                self.stats[k] = 0
            logger.info("PerformanceCache cleared successfully")


# Global singleton performance cache instance
GLOBAL_CACHE = PerformanceCache()
