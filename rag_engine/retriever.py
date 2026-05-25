"""
Hybrid retriever: BM25 (lexical) + Dense (semantic) with Reciprocal Rank Fusion.

This is the core fix for the retrieval problem. Pure semantic search fails on 
acronyms because "MCP" embeds nearly identically regardless of document context.
BM25 catches exact lexical matches (e.g., "Model Context Protocol" vs other content).
RRF properly merges the two ranking lists without needing score normalization.
"""

import logging
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import DENSE_K, BM25_K, FINAL_K, RRF_K
from .ingest import load_bm25_cache

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = RRF_K,
) -> list[tuple[Document, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.
    
    For each document, its RRF score is:
        score(doc) = Σ  1 / (k + rank_in_retriever)
    
    This properly combines rankings from different scoring systems
    (BM25 scores vs cosine similarity) without needing normalization.
    
    Args:
        ranked_lists: List of ranked lists, each containing (Document, score) tuples.
        k: Smoothing constant (default 60, standard in literature).
    
    Returns:
        Fused ranking as list of (Document, rrf_score) sorted descending.
    """
    # Use (filename, page_content_hash) as document identity key
    # since the same chunk can appear in multiple retriever results
    doc_scores: dict[str, float] = {}
    doc_objects: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list):
            # Create a unique key for this chunk
            doc_key = f"{doc.metadata.get('filename', '')}::{hash(doc.page_content)}"

            if doc_key not in doc_scores:
                doc_scores[doc_key] = 0.0
                doc_objects[doc_key] = doc

            # RRF formula: 1 / (k + rank)
            # rank is 0-indexed, so rank=0 → highest score
            doc_scores[doc_key] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    return [(doc_objects[key], doc_scores[key]) for key in sorted_keys]


def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    bm25_cache: dict | None = None,
    dense_k: int = DENSE_K,
    bm25_k: int = BM25_K,
    final_k: int = FINAL_K,
) -> list[Document]:
    """Perform hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion.
    
    Args:
        query: User query string.
        vectorstore: Chroma vectorstore for dense retrieval.
        bm25_cache: Cached BM25 index (from ingest.load_bm25_cache).
        dense_k: Number of candidates from dense search.
        bm25_k: Number of candidates from BM25 search.
        final_k: Number of final results after fusion.
    
    Returns:
        List of Document objects, ranked by relevance.
    """
    ranked_lists = []

    # ── Stage 1: Dense (Semantic) Retrieval ──
    try:
        dense_results = vectorstore.similarity_search_with_relevance_scores(
            query, k=dense_k
        )
        ranked_lists.append(dense_results)
        logger.debug(f"Dense retrieval: {len(dense_results)} results")
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")

    # ── Stage 2: BM25 (Lexical) Retrieval ──
    if bm25_cache is None:
        bm25_cache = load_bm25_cache()

    if bm25_cache is not None:
        try:
            bm25 = bm25_cache["bm25"]
            chunk_metadata = bm25_cache["chunk_metadata"]

            # Tokenize query the same way as corpus
            query_tokens = query.lower().split()

            # Get BM25 scores for all documents
            scores = bm25.get_scores(query_tokens)

            # Get top-k indices
            import numpy as np
            top_indices = np.argsort(scores)[::-1][:bm25_k]

            bm25_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include docs with positive BM25 score
                    cm = chunk_metadata[idx]
                    doc = Document(
                        page_content=cm["page_content"],
                        metadata=cm["metadata"],
                    )
                    bm25_results.append((doc, float(scores[idx])))

            ranked_lists.append(bm25_results)
            logger.debug(f"BM25 retrieval: {len(bm25_results)} results")
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
    else:
        logger.warning("BM25 cache not available — using dense retrieval only")

    # ── Stage 3: Reciprocal Rank Fusion ──
    if not ranked_lists:
        logger.error("No retrieval results from any method!")
        return []

    fused = reciprocal_rank_fusion(ranked_lists)

    # Return top-k documents
    results = [doc for doc, _score in fused[:final_k]]

    # Log which documents were retrieved for debugging
    source_summary = {}
    for doc in results:
        fn = doc.metadata.get("filename", "unknown")
        source_summary[fn] = source_summary.get(fn, 0) + 1
    logger.info(f"Retrieval results: {source_summary}")

    return results
