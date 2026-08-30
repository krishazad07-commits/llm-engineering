"""
Two-stage retrieval: vector search (wide) → cross-encoder rerank (narrow).

Vector search is fast but rough at rank 1. The cross-encoder reads
(query, chunk) together and produces a sharper ranking. We only rerank
a shortlist because cross-encoders are too slow to run on the full corpus.
"""

from search_documents import retrieve  # your existing vector retriever
from sentence_transformers import CrossEncoder

# Module-level singleton so we load the model once per process, not per query.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder on first use, then reuse."""
    global _reranker
    if _reranker is None:
        # First call downloads the model from HuggingFace.
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker


def retrieve_reranked(
    client,
    conn,
    query: str,
    k: int = 10,
    wide_k: int = 20,
) -> list[tuple]:
    """
    Two-stage retrieval: vector top wide_k → cross-encoder rerank → top k.

    Returns tuples in the same shape as retrieve() so this is a drop-in swap.
    Rerank scores are NOT attached (tuples don't support named fields);
    inspect via a separate diagnostic script if needed.
    """
    # Stage 1: wide vector retrieval
    candidates = retrieve(client, conn, query, wide_k)

    # Edge case
    if len(candidates) <= k:
        return candidates

    # Stage 2: build (query, chunk_text) pairs — chunk_text is at index 4
    pairs = [[query, candidate[4]] for candidate in candidates]

    reranker = get_reranker()
    scores = reranker.predict(pairs)

    # Stage 3: sort candidates by score desc, take top-k
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [candidate for candidate, score in ranked[:k]]