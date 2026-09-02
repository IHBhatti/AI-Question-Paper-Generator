"""
rag/retriever.py
----------------
Thin retrieval layer on top of the FAISS vector store. Supports filtering
by one or more chapters (or the entire document) and caps how many chunks
are ever sent to the LLM, which is the core guarantee that this app never
sends the whole document to the model.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import DEFAULT_TOP_K, MAX_TOP_K

logger = logging.getLogger(__name__)


def retrieve_chunks(
    vectorstore: FAISS,
    query: str,
    chapters: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[Document]:
    """Retrieve the top_k most relevant chunks for `query`, optionally
    restricted to a set of chapter names. Never returns more than
    MAX_TOP_K chunks, regardless of what is requested, as a hard safety
    cap against accidentally dumping the whole document into a prompt."""
    top_k = min(top_k, MAX_TOP_K)

    if chapters:
        # IMPORTANT: FAISS applies the metadata filter AFTER pulling an
        # initial candidate pool (`fetch_k`, default just 20) by raw
        # embedding similarity — it does NOT search within the filtered
        # subset. With many chapters in a document, the true top-20 by
        # similarity can easily contain zero chunks from the one chapter
        # we're filtering to, so the filter comes back empty even though
        # that chapter has plenty of relevant chunks further down the
        # full ranking. Widening fetch_k to the whole index guarantees
        # the filter actually gets to see every chunk that could match.
        total_vectors = getattr(getattr(vectorstore, "index", None), "ntotal", None)
        fetch_k = total_vectors if total_vectors else 1000

        filter_fn = lambda meta: meta.get("chapter") in chapters
        results = vectorstore.similarity_search(query, k=top_k, filter=filter_fn, fetch_k=fetch_k)
    else:
        results = vectorstore.similarity_search(query, k=top_k)

    logger.info(
        "Retrieved %d chunks for query='%s' chapters=%s", len(results), query[:60], chapters
    )
    return results


def retrieve_diverse_sample(
    vectorstore: FAISS,
    topic_pairs: List[tuple],
    per_topic_k: int = 4,
) -> List[Document]:
    """For test-paper generation we don't have one single natural-language
    query — we want broad, non-redundant coverage of the material, spread
    evenly across the chapters the user actually selected.

    `topic_pairs` is a list of (chapter_name, query_text) tuples. Each
    query is filtered to its OWN single chapter — not "any of the
    selected chapters" — which is what actually guarantees every chapter
    gets real representation. Filtering to the full selected-chapter set
    on every query would let semantic similarity alone decide which
    chapter's content shows up, and early/generic chapters (e.g. a "basic
    concepts" unit) tend to win that competition regardless of which
    chapter a query was meant to target.
    """
    seen_content = set()
    combined: List[Document] = []

    for chapter_name, topic in topic_pairs:
        chapter_filter = [chapter_name] if chapter_name else None
        docs = retrieve_chunks(vectorstore, topic, chapters=chapter_filter, top_k=per_topic_k)
        for d in docs:
            key = d.page_content.strip()
            if key not in seen_content:
                seen_content.add(key)
                combined.append(d)

    logger.info(
        "Diverse retrieval collected %d unique chunks across %d chapter-targeted queries",
        len(combined), len(topic_pairs),
    )
    return combined


def format_context(chunks: List[Document]) -> str:
    """Render retrieved chunks into a single context string, each block
    labeled with its chapter/page/source so the LLM can cite it and so
    the "insufficient information" fallback can reason about coverage."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = f"[Chunk {i} | Chapter: {meta.get('chapter', 'N/A')} | Page: {meta.get('page', 'N/A')} | Source: {meta.get('source', 'N/A')}]"
        blocks.append(f"{header}\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)
