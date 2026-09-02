"""
rag/loader.py
-------------
Loads raw text out of uploaded PDF or TXT files while preserving
page-level structure. Uses PyMuPDF (fitz) for PDFs since it is free,
fast, and gives reliable per-page text plus font-size metadata that
`parser.py` uses for heading detection.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from config import OCR_ENABLED, OCR_MIN_NATIVE_CHARS
from rag.ocr import ocr_document_pages

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Text and layout info for a single page (page 1 for TXT files)."""

    page_number: int
    text: str
    # list of (text, font_size) for lines on this page — used for heading
    # detection in parser.py. Empty for plain TXT input.
    lines: List[tuple] = field(default_factory=list)


@dataclass
class LoadedDocument:
    source_filename: str
    file_hash: str
    pages: List[PageContent]
    ocr_page_numbers: List[int] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def is_scanned_document(self) -> bool:
        """True if most pages needed OCR — useful for surfacing a message
        to the user explaining why processing took longer than usual."""
        return len(self.pages) > 0 and len(self.ocr_page_numbers) / len(self.pages) > 0.5


def _hash_bytes(data: bytes) -> str:
    """Stable content hash, used for duplicate-document detection."""
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(file_path: str | Path) -> str:
    """Public helper so callers (e.g. app.py) can check whether a document
    was already processed *before* paying the cost of loading/OCR-ing it."""
    return _hash_bytes(Path(file_path).read_bytes())


def load_pdf(
    file_path: str | Path,
    source_filename: str | None = None,
    ocr_progress_callback=None,
) -> LoadedDocument:
    """Extract text page-by-page from a PDF, preserving page numbers and
    capturing per-line font sizes so headings/chapters can be detected.

    Many real-world textbook PDFs (e.g. textbook-board scans) have no
    embedded text layer at all — every page is really just a scanned
    image. For any page whose native text is too sparse, this
    automatically falls back to OCR (with orientation auto-correction),
    so the rest of the pipeline never has to know the difference.
    """
    file_path = Path(file_path)
    raw_bytes = file_path.read_bytes()
    file_hash = _hash_bytes(raw_bytes)

    pages: List[PageContent] = []
    pages_needing_ocr: List[int] = []

    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            lines = []
            text_parts = []
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if not line_text:
                        continue
                    max_size = max(s.get("size", 0) for s in spans)
                    lines.append((line_text, max_size))
                    text_parts.append(line_text)
            page_text = "\n".join(text_parts)

            if OCR_ENABLED and len(page_text.strip()) < OCR_MIN_NATIVE_CHARS:
                # Placeholder for now; filled in by the OCR pass below.
                pages_needing_ocr.append(i)
                pages.append(PageContent(page_number=i, text="", lines=[]))
            else:
                pages.append(PageContent(page_number=i, text=page_text, lines=lines))

        if pages_needing_ocr:
            logger.info(
                "%d of %d pages have no usable text layer — running OCR "
                "(this is normal for scanned textbook PDFs and may take a "
                "while the first time; results are cached).",
                len(pages_needing_ocr),
                len(pages),
            )
            ocr_results = ocr_document_pages(
                doc, file_hash, pages_needing_ocr, progress_callback=ocr_progress_callback
            )
            for page_number in pages_needing_ocr:
                entry = ocr_results.get(str(page_number))
                if entry:
                    idx = page_number - 1
                    pages[idx] = PageContent(
                        page_number=page_number,
                        text=entry["text"],
                        lines=[(t, s) for t, s in entry["lines"]],
                    )

    logger.info("Loaded PDF '%s' with %d pages", source_filename or file_path.name, len(pages))
    return LoadedDocument(
        source_filename=source_filename or file_path.name,
        file_hash=file_hash,
        pages=pages,
        ocr_page_numbers=pages_needing_ocr,
    )


def load_txt(file_path: str | Path, source_filename: str | None = None) -> LoadedDocument:
    """Load a plain-text file. TXT has no real pages, so the whole file is
    treated as a single logical page (page_number=1); chapter headings are
    still detected in parser.py using simple heuristics (line patterns)."""
    file_path = Path(file_path)
    raw_bytes = file_path.read_bytes()
    file_hash = _hash_bytes(raw_bytes)

    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = [(ln.strip(), 0.0) for ln in text.splitlines() if ln.strip()]

    logger.info("Loaded TXT '%s' (%d chars)", source_filename or file_path.name, len(text))
    return LoadedDocument(
        source_filename=source_filename or file_path.name,
        file_hash=file_hash,
        pages=[PageContent(page_number=1, text=text, lines=lines)],
    )


def load_document(
    file_path: str | Path,
    source_filename: str | None = None,
    ocr_progress_callback=None,
) -> LoadedDocument:
    """Dispatch to the correct loader based on file extension."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path, source_filename, ocr_progress_callback=ocr_progress_callback)
    elif suffix == ".txt":
        return load_txt(file_path, source_filename)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only .pdf and .txt are supported.")
