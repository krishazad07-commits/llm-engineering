"""
hybrid_search.py — BM25 keyword search + RRF fusion with vector search.

Provides retrieve_hybrid() as a drop-in alternative to search_documents.retrieve().
"""

import psycopg
from rank_bm25 import BM25Okapi
from search_documents import retrieve


def tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace. Minimal tokenizer, good enough for prose."""
    # TODO 1: return lowercased whitespace-split words
    return text.lower().split()


def build_bm25_index(conn: psycopg.Connection) -> tuple[BM25Okapi, list[tuple]]:
    """Load all chunks from the DB, tokenize, and build a BM25 index.

    Returns (bm25_index, chunks) where chunks[i] corresponds to the
    document BM25 scored at index i.
    """
    with conn.cursor() as cur:
        cur.execute("select id, source_doc, page, chunk_id, content from documents")
        chunks = cur.fetchall()

    tokenized_docs = [tokenize(row[4]) for row in chunks]
    bm25_index = BM25Okapi(tokenized_docs)
    return bm25_index, chunks


def search_bm25(
    query: str,
    bm25_index: BM25Okapi,
    chunks: list[tuple],
    k: int = 10,
) -> list[tuple]:
    """Score chunks against query with BM25, return top-k chunk tuples (highest score first)."""
    # TODO 1: tokenize the query
    query_token = tokenize(query)
    # TODO 2: get_scores from bm25_index
    scores = bm25_index.get_scores(query_token)
    # TODO 3: pair each chunk with its score — list of (chunk, score) tuples
    #         hint: zip(chunks, scores)
    paired = list(zip(chunks, scores))
    # TODO 4: sort the paired list by score, descending
    #         hint: sorted(..., key=.. ., reverse=True)
    ranked = sorted(paired, key=lambda x: x[1], reverse=True)
    # TODO 5: take the first k, return just the chunk tuples (drop the score)
    #         hint: a list comprehension over the top-k slice
    return [chunk for chunk, score in ranked[:k]]


def rrf_fuse(
    vector_results: list[tuple],
    bm25_results: list[tuple],
    k: int = 60,
    top_n: int = 10,
) -> list[tuple]:
    """Fuse two ranked chunk lists via Reciprocal Rank Fusion.

    Each chunk's RRF score = sum of 1/(k + rank) across every list it
    appears in (rank is 1-indexed). Returns top_n chunk tuples, ranked
    by fused score descending.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, tuple] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        chunk_id = chunk[0]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_lookup[chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        chunk_id = chunk[3]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_lookup[chunk_id] = chunk

    ranked_ids = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [chunk_lookup[chunk_id] for chunk_id, score in ranked_ids[:top_n]]


def retrieve_hybrid(
    client,
    conn: psycopg.Connection,
    bm25_index: BM25Okapi,
    chunks: list[tuple],
    query: str,
    k: int = 10,
) -> list[tuple]:
    """Hybrid retrieval: vector search + BM25, fused via RRF. Returns top-k chunk tuples."""
    # TODO 1: call retrieve() from search_documents (vector search)
    #         retrieve(client, conn, query, k)
    vector_results = retrieve(client, conn, query, k)
    # TODO 2: call search_bm25(query, bm25_index, chunks, k)
    bm25_results = search_bm25(query, bm25_index, chunks, k)
    # TODO 3: call rrf_fuse(vector_results, bm25_results, top_n=k)
    fused_chunks = rrf_fuse(vector_results, bm25_results, top_n=k)
    # TODO 4: return the fused result
    return fused_chunks
