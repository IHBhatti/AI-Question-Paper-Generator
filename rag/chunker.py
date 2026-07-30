"""
rag/chunker.py
--------------
Splits document text into semantic chunks using LangChain's
RecursiveCharacterTextSplitter, tagging every chunk with metadata:
chapter, page number, and source file. This metadata is what lets the
retriever later filter by chapter selection in the UI.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.loader import LoadedDocument
from rag.parser import Chapter, chapter_for_line

logger = logging.getLogger(__name__)


def _segment_page_by_chapter(page, chapters: List[Chapter]) -> List[tuple]:
    """Split one page's lines into contiguous (chapter_name, text) segments.

    This is the key fix that makes chapter tagging correct even when
    several chapters share a single page — which is always true for TXT
    files (loaded as one logical page) and can happen in PDFs with short
    chapters. Using per-line chapter lookup instead of a single
    per-page lookup keeps every chunk tagged with the right chapter.
    """
    if not page.lines:
        # No line-level data available — treat the whole page as one segment.
        chapter_name = (
            chapter_for_line(chapters, page.page_number, 0) if chapters else "Full Document"
        )
        return [(chapter_name, page.text)] if page.text.strip() else []

    segments = []
    current_chapter = None
    current_lines: List[str] = []

    for line_idx, (line_text, _font_size) in enumerate(page.lines):
        chapter_name = (
            chapter_for_line(chapters, page.page_number, line_idx)
            if chapters
            else "Full Document"
        )
        if current_chapter is None:
            current_chapter = chapter_name
        if chapter_name != current_chapter:
            if current_lines:
                segments.append((current_chapter, "\n".join(current_lines)))
            current_chapter = chapter_name
            current_lines = [line_text]
        else:
            current_lines.append(line_text)

    if current_lines:
        segments.append((current_chapter, "\n".join(current_lines)))

    return segments


def chunk_document(doc: LoadedDocument, chapters: List[Chapter]) -> List[Document]:
    """Turn a LoadedDocument into a list of LangChain Documents, each with
    chapter/page/source metadata attached.

    Each page is first split into chapter-pure segments (see
    `_segment_page_by_chapter`), then each segment is split with the
    recursive splitter for semantically coherent, appropriately sized
    chunks. This guarantees every chunk's `chapter` metadata is accurate
    even when multiple chapters fall on the same page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []

    for page in doc.pages:
        if not page.text.strip():
            continue

        for chapter_name, segment_text in _segment_page_by_chapter(page, chapters):
            if not segment_text.strip():
                continue
            segment_chunks = splitter.split_text(segment_text)
            for chunk_text in segment_chunks:
                if not chunk_text.strip():
                    continue
                metadata = {
                    "chapter": chapter_name,
                    "page": page.page_number,
                    "source": doc.source_filename,
                }
                all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    logger.info(
        "Created %d chunks from '%s' across %d pages",
        len(all_chunks),
        doc.source_filename,
        len(doc.pages),
    )
    return all_chunks


def get_available_chapters(chunks: List[Document]) -> List[str]:
    """Return the distinct, ordered list of chapter names present in the
    chunked document (used to populate the Streamlit chapter checklist)."""
    seen = []
    for chunk in chunks:
        chapter = chunk.metadata.get("chapter", "Unassigned")
        if chapter not in seen:
            seen.append(chapter)
    return seen
