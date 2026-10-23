"""Unit tests for the ephemeral ChromaDB vector store and session isolation."""

import pytest
import chromadb

from financial_report_analyzer.errors import VectorStoreError, RetrievalError
from financial_report_analyzer.schemas import ChunkRecord
from financial_report_analyzer.vector_store import VectorStore, sanitize_collection_name


def make_unit_vector(hot_idx: int, dim: int = 768) -> list:
    """Helper creating a 1-hot unit vector of specified dimension."""
    vec = [0.0] * dim
    vec[hot_idx % dim] = 1.0
    return vec


def test_sanitize_collection_name():
    """Verify collection name sanitization satisfies ChromaDB requirements."""
    name = sanitize_collection_name("doc_test@123!!_sample#file.pdf")
    assert 3 <= len(name) <= 63
    assert not name.startswith("_")
    assert not name.endswith("_")
    assert "@" not in name
    assert "#" not in name


def test_add_chunks_and_count():
    """Verify adding chunks with explicit vectors updates collection count."""
    client = chromadb.EphemeralClient()
    store = VectorStore(document_id="doc_101", client=client, expected_dimension=768)

    chunks = [
        ChunkRecord(
            chunk_id=f"c_{i}",
            document_id="doc_101",
            source_filename="test.pdf",
            page_numbers=[i + 1],
            chunk_index=i,
            text=f"Financial chunk number {i}",
            embedding=make_unit_vector(i),
        )
        for i in range(4)
    ]

    added = store.add_chunks(chunks)
    assert added == 4
    assert store.count() == 4
    store.delete()


def test_semantic_similarity_search_and_provenance():
    """Verify semantic retrieval returns nearest chunk with accurate 1-based page metadata."""
    client = chromadb.EphemeralClient()
    store = VectorStore(document_id="doc_202", client=client, expected_dimension=768)

    # Chunk A is oriented along dimension 0 (e.g. Revenue)
    chunk_a = ChunkRecord(
        chunk_id="chunk_rev",
        document_id="doc_202",
        source_filename="apple.pdf",
        page_numbers=[12],
        chunk_index=0,
        text="Total Net Sales reached $383 billion in FY2024.",
        embedding=make_unit_vector(0),
    )
    # Chunk B is oriented along dimension 10 (e.g. Legal Risks)
    chunk_b = ChunkRecord(
        chunk_id="chunk_risk",
        document_id="doc_202",
        source_filename="apple.pdf",
        page_numbers=[45],
        chunk_index=1,
        text="The Company is subject to antitrust litigation worldwide.",
        embedding=make_unit_vector(10),
    )

    store.add_chunks([chunk_a, chunk_b])

    # Query with vector pointing toward dimension 0 (Revenue)
    query_vec = make_unit_vector(0)
    results = store.query(query_vec, top_k=1)

    assert len(results) == 1
    top = results[0]
    assert top.chunk_id == "chunk_rev"
    assert top.page_number == 12
    assert "Total Net Sales" in top.text
    # Cosine distance should be ~0.0 and similarity ~1.0
    assert top.distance < 0.01
    assert top.similarity > 0.99
    store.delete()


def test_query_dimension_mismatch_raises_retrieval_error():
    """Verify query with mismatched vector dimension fails with RetrievalError."""
    client = chromadb.EphemeralClient()
    store = VectorStore(document_id="doc_303", client=client, expected_dimension=768)
    store.add_chunks(
        [
            ChunkRecord(
                chunk_id="c0",
                document_id="doc_303",
                source_filename="test.pdf",
                page_numbers=[1],
                chunk_index=0,
                text="Text",
                embedding=make_unit_vector(0, 768),
            )
        ]
    )

    with pytest.raises(RetrievalError) as exc_info:
        # Pass a 512-dim vector into 768-dim index
        store.query(make_unit_vector(0, 512))
    assert "dimension" in str(exc_info.value)
    store.delete()


def test_session_isolation_independent_collections():
    """Verify strict document isolation: independent stores never retrieve each other's chunks."""
    client = chromadb.EphemeralClient()

    # Session A: Document Alpha (oriented on dim 0)
    store_a = VectorStore(document_id="doc_alpha", client=client, expected_dimension=768)
    store_a.add_chunks(
        [
            ChunkRecord(
                chunk_id="alpha_c0",
                document_id="doc_alpha",
                source_filename="alpha.pdf",
                page_numbers=[1],
                chunk_index=0,
                text="Secret content belonging to Company Alpha.",
                embedding=make_unit_vector(0, 768),
            )
        ]
    )

    # Session B: Document Beta (oriented on dim 1)
    store_b = VectorStore(document_id="doc_beta", client=client, expected_dimension=768)
    store_b.add_chunks(
        [
            ChunkRecord(
                chunk_id="beta_c0",
                document_id="doc_beta",
                source_filename="beta.pdf",
                page_numbers=[1],
                chunk_index=0,
                text="Secret content belonging to Company Beta.",
                embedding=make_unit_vector(1, 768),
            )
        ]
    )

    # Query Store A with vector pointing to Beta: it must NOT return Beta!
    query_beta = make_unit_vector(1, 768)
    results_a = store_a.query(query_beta, top_k=5)
    # Store A only has Alpha's chunk
    assert len(results_a) == 1
    assert results_a[0].document_id == "doc_alpha"
    assert "Alpha" in results_a[0].text
    assert "Beta" not in results_a[0].text

    # Query Store B with vector pointing to Alpha: it must NOT return Alpha!
    query_alpha = make_unit_vector(0, 768)
    results_b = store_b.query(query_alpha, top_k=5)
    assert len(results_b) == 1
    assert results_b[0].document_id == "doc_beta"
    assert "Beta" in results_b[0].text
    assert "Alpha" not in results_b[0].text

    store_a.delete()
    store_b.delete()


def test_delete_collection_cleans_resources():
    """Verify delete() removes the collection and prevents further queries."""
    client = chromadb.EphemeralClient()
    store = VectorStore(document_id="doc_del", client=client, expected_dimension=768)
    store.add_chunks(
        [
            ChunkRecord(
                chunk_id="c0",
                document_id="doc_del",
                source_filename="test.pdf",
                page_numbers=[1],
                chunk_index=0,
                text="Test text",
                embedding=make_unit_vector(0, 768),
            )
        ]
    )
    assert store.count() == 1

    store.delete()
    assert store.is_active is False
    assert store.count() == 0

    with pytest.raises(RetrievalError):
        store.query(make_unit_vector(0, 768))
