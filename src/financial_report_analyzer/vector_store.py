"""Session-isolated ephemeral ChromaDB vector store for financial report chunks."""

import re
import uuid
import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from financial_report_analyzer.errors import VectorStoreError, RetrievalError
from financial_report_analyzer.schemas import ChunkRecord, RetrievalResult

logger = logging.getLogger(__name__)


def sanitize_collection_name(raw_name: str) -> str:
    """Ensure collection name meets ChromaDB requirements:

    - 3 to 63 characters long
    - Alphanumeric, underscores, hyphens only
    - Starts and ends with alphanumeric character
    """
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_name)
    # Trim to 50 chars to allow room for unique suffixes
    clean = clean[:50].strip("_-")
    if len(clean) < 3:
        clean = f"col_{clean}"
    # Ensure starts/ends with alphanumeric
    clean = clean.strip("_-")
    if not clean:
        clean = "col_default"
    return clean[:63]


class VectorStore:
    """Manages an isolated ephemeral ChromaDB collection for an active report session."""

    def __init__(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
        client: Optional[ClientAPI] = None,
        embedding_model: str = "gemini-embedding-2",
        expected_dimension: int = 768,
    ) -> None:
        self.document_id = document_id
        self.embedding_model = embedding_model
        self.expected_dimension = expected_dimension
        self._is_deleted = False

        # Use provided client or instantiate an ephemeral in-memory client
        self._client: ClientAPI = client if client is not None else chromadb.EphemeralClient()

        # Generate a unique, unpredictable collection name per document indexing session
        if collection_name:
            self.collection_name = sanitize_collection_name(collection_name)
        else:
            suffix = uuid.uuid4().hex[:8]
            self.collection_name = sanitize_collection_name(f"doc_{document_id}_{suffix}")

        try:
            # Configure collection with explicit cosine distance
            self._collection: Collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created isolated ephemeral Chroma collection '%s'", self.collection_name)
        except Exception as exc:
            logger.error("Failed to create collection '%s': %s", self.collection_name, str(exc))
            raise VectorStoreError(f"Could not initialize Chroma collection: {str(exc)}") from exc

    @property
    def is_active(self) -> bool:
        """Check if collection is active and has not been deleted."""
        return not self._is_deleted and self._collection is not None

    def count(self) -> int:
        """Return the count of indexed items in the collection."""
        if not self.is_active:
            return 0
        try:
            return self._collection.count()
        except Exception as exc:
            logger.warning("Failed to count items in collection '%s': %s", self.collection_name, str(exc))
            return 0

    def add_chunks(
        self,
        chunks: List[ChunkRecord],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """Add chunks and precomputed embeddings to the collection.

        NOTE: Explicit precomputed embeddings are passed to prevent Chroma from
        invoking any default embedding function.
        """
        if not self.is_active:
            raise VectorStoreError("Cannot add chunks to a deleted or inactive collection.")

        if not chunks:
            return 0

        # Validate that embeddings exist either as an argument or on chunk objects
        vectors: List[List[float]] = []
        if embeddings is not None:
            if len(embeddings) != len(chunks):
                raise VectorStoreError(
                    f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks)})"
                )
            vectors = embeddings
        else:
            for idx, c in enumerate(chunks):
                if c.embedding is None:
                    raise VectorStoreError(f"Chunk at index {idx} ({c.chunk_id}) has no embedding vector.")
                vectors.append(c.embedding)

        # Validate dimension of all vectors
        for idx, vec in enumerate(vectors):
            if len(vec) != self.expected_dimension:
                raise VectorStoreError(
                    f"Vector at index {idx} has dimension {len(vec)}, expected {self.expected_dimension}"
                )

        # Prepare payload for Chroma
        ids: List[str] = [c.chunk_id for c in chunks]
        documents: List[str] = [c.text for c in chunks]
        metadatas: List[Dict[str, Any]] = [
            {
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": int(c.page_numbers[0]) if c.page_numbers else 1,
                "chunk_index": int(c.chunk_index),
                "chunk_id": c.chunk_id,
                "embedding_model": self.embedding_model,
                "embedding_dimension": int(self.expected_dimension),
            }
            for c in chunks
        ]

        try:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=vectors,
                metadatas=metadatas,
            )
            logger.info("Indexed %d chunks into collection '%s'", len(chunks), self.collection_name)
            return len(chunks)
        except Exception as exc:
            logger.error("Failed to add chunks to collection '%s': %s", self.collection_name, str(exc))
            raise VectorStoreError(f"Failed to index chunks into Chroma collection: {str(exc)}") from exc

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Perform semantic similarity search using precomputed query vector.

        Returns top-k most relevant ChunkRecords with provenance and distance metrics.
        """
        if not self.is_active:
            raise RetrievalError("Cannot query an inactive or deleted vector store.")

        if not isinstance(query_embedding, (list, tuple)):
            raise RetrievalError("Query embedding must be a list of floats.")

        if len(query_embedding) != self.expected_dimension:
            raise RetrievalError(
                f"Query embedding dimension ({len(query_embedding)}) does not match index dimension ({self.expected_dimension})"
            )

        total_items = self.count()
        if total_items == 0:
            logger.info("Vector collection '%s' is empty; returning 0 results", self.collection_name)
            return []

        actual_k = min(top_k, total_items)
        try:
            results = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Chroma query failed on collection '%s': %s", self.collection_name, str(exc))
            raise RetrievalError(f"Vector search failed: {str(exc)}") from exc

        retrieved: List[RetrievalResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance d = 1 - cos(theta) (range [0, 2] for arbitrary vectors)
            # For unit-normalized vectors: similarity = 1.0 - distance
            cosine_dist = float(dist)
            # Heuristic similarity metric bounded in [0.0, 1.0]: 1.0 is exact match
            cosine_sim = max(0.0, min(1.0, 1.0 - (cosine_dist / 2.0)))

            retrieved.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", self.document_id),
                    source_filename=meta.get("source_filename", "document.pdf"),
                    page_number=int(meta.get("page_number", 1)),
                    text=text,
                    distance=round(cosine_dist, 4),
                    similarity=round(cosine_sim, 4),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata=dict(meta),
                )
            )

        return retrieved

    def delete(self) -> None:
        """Safely delete collection and release all resources."""
        if self._is_deleted:
            return

        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info("Deleted Chroma collection '%s'", self.collection_name)
        except Exception as exc:
            logger.warning("Error deleting collection '%s': %s", self.collection_name, str(exc))
        finally:
            self._is_deleted = True
            self._collection = None
