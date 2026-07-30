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
        # LangChain FAISS supports a metadata filter callable/dict.
        filter_fn = lambda meta: meta.get("chapter") in chapters
        results = vectorstore.similarity_search(query, k=top_k, filter=filter_fn)
    else:
        results = vectorstore.similarity_search(query, k=top_k)

    logger.info(
        "Retrieved %d chunks for query='%s' chapters=%s", len(results), query[:60], chapters
    )
    return results


def retrieve_diverse_sample(
    vectorstore: FAISS,
    topics: List[str],
    chapters: Optional[List[str]] = None,
    per_topic_k: int = 4,
) -> List[Document]:
    """For test-paper generation we don't have one single natural-language
    query — we want broad, non-redundant coverage of the material. This
    runs retrieval against several representative topic queries (e.g. one
    per requested question, or per Bloom's level) and de-duplicates the
    combined result set."""
    seen_content = set()
    combined: List[Document] = []

    for topic in topics:
        docs = retrieve_chunks(vectorstore, topic, chapters=chapters, top_k=per_topic_k)
        for d in docs:
            key = d.page_content.strip()
            if key not in seen_content:
                seen_content.add(key)
                combined.append(d)

    logger.info("Diverse retrieval collected %d unique chunks", len(combined))
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
