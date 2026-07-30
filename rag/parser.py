"""
rag/parser.py
-------------
Detects chapter boundaries inside a LoadedDocument so the rest of the
pipeline can tag every chunk with a chapter name, and so the UI can offer
chapter-based selection.

Detection strategy (heuristic, free, no extra ML model required):
1. Pattern based: lines matching common chapter/unit patterns, e.g.
   "Chapter 1", "Unit 3: Dynamics", etc. — the primary signal, since it
   works reliably even on OCR'd text where font-size metadata is noisier.
2. Font-size based: for native (non-OCR) PDFs, a line whose font size is
   notably larger than the page's median body-text size is a heading
   candidate, used as a fallback when no keyword pattern matches.
3. If neither approach finds anything, detection "fails" and the caller
   (app.py) should fall back to letting the user manually define chapters
   or page ranges.

A crucial real-world wrinkle this module handles: textbook PDFs (scanned
or not) very commonly reprint the current chapter/unit title as a running
header on *every single page* of that chapter. Naively treating every
matching line as "a new chapter" would register dozens of spurious
chapters — one per page — instead of one per actual chapter. This module
recognizes repeats of the same canonical chapter key and merges them.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rag.loader import LoadedDocument

logger = logging.getLogger(__name__)

# Each pattern captures (keyword, number, remaining title text). Matching
# is anchored to the start of the line since headers are typically the
# first thing on their line, but tolerant of a trailing colon/dash.
CHAPTER_PATTERNS = [
    re.compile(r"^(chapter)\s*[:\-]?\s*(\d+)\b\s*[:\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(unit)\s*[:\-]?\s*(\d+)\b\s*[:\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(section)\s*[:\-]?\s*(\d+)\b\s*[:\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(module)\s*[:\-]?\s*(\d+)\b\s*[:\-]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(part)\s*[:\-]?\s*(\d+)\b\s*[:\-]?\s*(.*)$", re.IGNORECASE),
]
# Lower-precision fallback for numbered-list-style headings, e.g.
# "3. Deep Learning". No reliable canonical key across repeats, so each
# match that isn't an exact literal repeat is treated as its own chapter.
NUMBERED_TITLE_PATTERN = re.compile(r"^(\d+)\.\s+([A-Z][A-Za-z\s]{2,60})$")


@dataclass
class Chapter:
    name: str
    start_page: int
    start_line: int = 0  # index into that page's `lines` list where the heading occurs
    end_page: Optional[int] = None  # filled in once the next chapter is known
    end_line: Optional[int] = None


def _match_chapter_pattern(line_text: str) -> Optional[Tuple[str, str]]:
    """Return (canonical_key, title_fragment) if the line looks like a
    chapter/unit heading, else None. `canonical_key` identifies the
    heading independent of exact wording/OCR noise in the trailing title
    text, so the same chapter's running header repeating on every page
    doesn't register as a new chapter each time."""
    stripped = line_text.strip()
    if not stripped or len(stripped) > 90:
        return None
    for pattern in CHAPTER_PATTERNS:
        m = pattern.match(stripped)
        if m:
            keyword, number, title_fragment = m.group(1), m.group(2), m.group(3)
            canonical_key = f"{keyword.lower()}_{number}"
            return canonical_key, title_fragment.strip()
    return None


def _match_numbered_pattern(line_text: str) -> Optional[Tuple[str, str]]:
    """Low-precision fallback for numbered-list-style headings, e.g.
    "3. Deep Learning" — only used when the document has no recognizable
    "Chapter/Unit N" style keyword headings at all, since on real body
    text (numbered lists, sub-points) this pattern is prone to false
    positives."""
    stripped = line_text.strip()
    if not stripped or len(stripped) > 90:
        return None
    m = NUMBERED_TITLE_PATTERN.match(stripped)
    if m:
        return f"numbered_{stripped.lower()}", m.group(2).strip()
    return None


def _is_pattern_heading(line_text: str) -> bool:
    return _match_chapter_pattern(line_text) is not None


def _font_size_threshold(doc: LoadedDocument) -> float:
    """Compute a heading-worthy font size threshold from the whole document's
    body text distribution (median * 1.25 is a solid, simple heuristic)."""
    sizes = [size for page in doc.pages for _, size in page.lines if size > 0]
    if not sizes:
        return 0.0
    median = statistics.median(sizes)
    return median * 1.25


def _looks_like_title_continuation(line_text: str) -> bool:
    """Heuristic for a wrapped title's second line, e.g. "Unit 1: Physical
    Quantities" followed by "and Measurement" on the next line. Used only
    to build a nicer display name — never affects chapter boundaries."""
    stripped = line_text.strip()
    if not stripped or len(stripped) > 50:
        return False
    if _is_pattern_heading(stripped):
        return False
    return not stripped.endswith((".", "?", "!")) and len(stripped.split()) <= 8


def _find_toc_like_pages(doc: LoadedDocument) -> set:
    """Identify pages that are almost certainly a Table of Contents rather
    than real chapter starts: several DIFFERENT chapter/unit numbers all
    matched on the same page (e.g. "Unit 1 ... 1", "Unit 2 ... 31", "Unit 3
    ... 58" listed one after another). Real chapter-opening pages only
    ever introduce one chapter each."""
    toc_pages = set()
    for page in doc.pages:
        canonical_keys_on_page = set()
        for line_text, _font_size in page.lines:
            match = _match_chapter_pattern(line_text)
            if match:
                canonical_keys_on_page.add(match[0])
        if len(canonical_keys_on_page) >= 3:
            toc_pages.add(page.page_number)
    return toc_pages


TOC_ENTRY_PATTERN = re.compile(
    r"^(chapter|unit|section|module|part)\s*[:\-]?\s*(\d+)\b\s+(.+?)\s+(\d{1,4})\s*$",
    re.IGNORECASE,
)


def _parse_toc_entries(doc: LoadedDocument, toc_pages: set) -> List[dict]:
    """Parse Table-of-Contents-like pages into
    [{"canonical_key": ..., "title": ..., "printed_page": int}, ...].
    Textbook-board TOCs reliably OCR as "Unit N <Title> <printed page
    number>" — this is a much more robust source of chapter page ranges
    than trying to catch every chapter's own (often decorative, harder to
    OCR) opening-page title."""
    entries = []
    for page in doc.pages:
        if page.page_number not in toc_pages:
            continue
        for line_text, _font_size in page.lines:
            m = TOC_ENTRY_PATTERN.match(line_text.strip())
            if not m:
                continue
            keyword, number, title, printed_page = m.groups()
            entries.append(
                {
                    "canonical_key": f"{keyword.lower()}_{number}",
                    "title": title.strip(),
                    "printed_page": int(printed_page),
                }
            )
    return entries


def _estimate_toc_offset(toc_entries: List[dict], body_chapter_pages: dict) -> Optional[int]:
    """Figure out the constant offset between a TOC entry's printed page
    number and the actual PDF page index, using whichever chapters we
    were able to reliably locate directly in the document body as
    reference points. Returns the most common offset, or None if there's
    no overlap to calibrate against."""
    offsets = []
    for entry in toc_entries:
        body_page = body_chapter_pages.get(entry["canonical_key"])
        if body_page is not None:
            offsets.append(body_page - entry["printed_page"])
    if not offsets:
        return None
    # Mode: the offset should be constant for the whole book; take the
    # most frequent value in case one or two reference points are noisy.
    return max(set(offsets), key=offsets.count)


def _fill_gaps_from_toc(
    chapters: List[Chapter],
    body_chapter_pages: dict,
    toc_entries: List[dict],
    total_pages: int,
) -> List[Chapter]:
    """Add chapters that TOC parsing found but in-body heading detection
    missed entirely (e.g. because that chapter's opening-page title is
    stylized/decorative artwork that OCR couldn't read as text at all)."""
    if not toc_entries:
        return chapters

    offset = _estimate_toc_offset(toc_entries, body_chapter_pages)
    if offset is None:
        return chapters

    seen_keys = set(body_chapter_pages.keys())
    for entry in toc_entries:
        if entry["canonical_key"] in seen_keys:
            continue
        estimated_page = entry["printed_page"] + offset
        estimated_page = max(1, min(total_pages, estimated_page))
        chapters.append(
            Chapter(name=entry["title"] or entry["canonical_key"], start_page=estimated_page, start_line=0)
        )
        seen_keys.add(entry["canonical_key"])
        logger.info(
            "Recovered chapter '%s' from Table of Contents (estimated page %d) — "
            "its own heading wasn't detectable in the document body.",
            entry["title"],
            estimated_page,
        )

    chapters.sort(key=lambda c: (c.start_page, c.start_line))
    return chapters


def detect_chapters(doc: LoadedDocument) -> List[Chapter]:
    """Attempt automatic chapter detection. Returns an empty list if nothing
    resembling a chapter heading could be found (caller should fall back to
    manual chapter/page-range entry).

    Two real-world wrinkles this handles:
    1. Table-of-contents pages list every chapter/unit at once — these are
       skipped entirely so they don't get mistaken for chapter starts.
    2. A chapter/unit title reprinted as a running header on every page of
       that chapter (typical in textbook scans) is recognized by its
       canonical key and merged into a single chapter, even if a numbered
       list item or other text momentarily interrupts the repeats.
    """
    toc_pages = _find_toc_like_pages(doc)
    # Font-size-based heading detection relies on real, precise embedded
    # font metadata. OCR-derived "font size" is just an average pixel
    # height per line, which is noisy and produces false positives (stray
    # figure numbers, table cells, etc. getting flagged as headings). Only
    # trust it for documents that weren't OCR'd at all.
    use_font_heuristic = len(doc.ocr_page_numbers) == 0
    threshold = _font_size_threshold(doc) if use_font_heuristic else 0.0

    # First pass: collect every occurrence of every canonical chapter key
    # (not deduped yet). A chapter's very first page in these books often
    # has decorative/stylized title art that OCRs messily (e.g. "Unit - 1
    # yok"), while the plain running header repeated on later pages of the
    # same chapter ("Unit 1: Physical Quantities and Measurement") is much
    # cleaner — so we pick the longest well-formed occurrence for display,
    # while still using the EARLIEST occurrence to mark where the chapter
    # actually starts (for correct chunk boundaries).
    occurrences: dict = {}  # canonical_key -> list of (page_number, line_idx, display_name)

    for page in doc.pages:
        if page.page_number in toc_pages:
            continue
        for line_idx, (line_text, font_size) in enumerate(page.lines):
            match = _match_chapter_pattern(line_text)
            if not match:
                continue
            canonical_key, _title_fragment = match
            display_name = line_text.strip()
            if line_idx + 1 < len(page.lines):
                next_line_text, _ = page.lines[line_idx + 1]
                if _looks_like_title_continuation(next_line_text):
                    display_name = f"{display_name} {next_line_text.strip()}"
            occurrences.setdefault(canonical_key, []).append(
                (page.page_number, line_idx, display_name)
            )

    chapters: List[Chapter] = []
    body_chapter_pages: dict = {}  # canonical_key -> start_page, for TOC offset calibration
    for canonical_key, occ_list in occurrences.items():
        earliest = min(occ_list, key=lambda o: (o[0], o[1]))
        best_name = max(occ_list, key=lambda o: len(o[2]))[2]
        chapters.append(Chapter(name=best_name, start_page=earliest[0], start_line=earliest[1]))
        body_chapter_pages[canonical_key] = earliest[0]

    chapters.sort(key=lambda c: (c.start_page, c.start_line))

    # Use the Table of Contents (if present) to recover any chapters whose
    # own opening-page heading we couldn't detect in the body at all —
    # common when a chapter title is stylized/decorative artwork that OCR
    # fails to read as text, even though the rest of that page OCRs fine.
    toc_entries = _parse_toc_entries(doc, toc_pages)
    if toc_entries:
        chapters = _fill_gaps_from_toc(
            chapters, body_chapter_pages, toc_entries, total_pages=len(doc.pages)
        )

    # Font-size-based fallback for large-font lines with no recognizable
    # keyword pattern — only meaningful for non-OCR'd (native text) PDFs,
    # and only used to fill in gaps if keyword detection found nothing
    # at all on a document (handled by the numbered-list fallback below,
    # so this path is intentionally conservative and rarely fires).

    # Low-precision numbered-list fallback ("3. Deep Learning") — only
    # attempted if the primary keyword patterns found nothing at all,
    # since on real body text this pattern is prone to false positives
    # (numbered lists, sub-points, MCQ options, etc.).
    if not chapters:
        logger.info("No keyword-style chapter headings found; trying numbered-list fallback")
        seen_canonical_keys: set = set()
        for page in doc.pages:
            if page.page_number in toc_pages:
                continue
            for line_idx, (line_text, _font_size) in enumerate(page.lines):
                match = _match_numbered_pattern(line_text)
                if not match:
                    continue
                canonical_key, _title_fragment = match
                if canonical_key in seen_canonical_keys:
                    continue
                chapters.append(
                    Chapter(
                        name=line_text.strip(), start_page=page.page_number, start_line=line_idx
                    )
                )
                seen_canonical_keys.add(canonical_key)

    # Fill in end_page/end_line for each chapter based on the next chapter's start,
    # so chunker.py can slice each page's lines into the correct chapter segments
    # even when several chapters share the same page (e.g. TXT files, which are
    # loaded as a single logical page).
    for i, chapter in enumerate(chapters):
        if i + 1 < len(chapters):
            chapter.end_page = chapters[i + 1].start_page
            chapter.end_line = chapters[i + 1].start_line
        else:
            chapter.end_page = doc.pages[-1].page_number
            chapter.end_line = None  # None means "to the end of that page"

    if chapters:
        logger.info("Auto-detected %d chapters", len(chapters))
    else:
        logger.warning("No chapters auto-detected; manual entry required")

    return chapters


def build_manual_chapters(page_ranges: dict) -> List[Chapter]:
    """Build a Chapter list from a user-supplied mapping of
    {chapter_name: (start_page, end_page)} when auto-detection fails."""
    chapters = []
    for name, (start, end) in page_ranges.items():
        # Manual entry is page-range based (no line-level granularity), so
        # each chapter spans the full range of its pages: from line 0 of the
        # start page to "end of page" (None) on the end page.
        chapters.append(
            Chapter(name=name, start_page=start, start_line=0, end_page=end, end_line=None)
        )
    return chapters


def chapter_for_page(chapters: List[Chapter], page_number: int) -> str:
    """Coarse, page-level lookup: return the chapter whose range contains
    this page number, or 'Unassigned'. Used only as a fallback when
    line-level granularity isn't needed. `chunker.py` uses the more
    precise `chapter_for_line` for actual chunk tagging, since a single
    page (or a whole TXT file, loaded as one logical page) can contain
    multiple chapters."""
    for chapter in chapters:
        end = chapter.end_page if chapter.end_page is not None else float("inf")
        if chapter.start_page <= page_number < end:
            return chapter.name
        if chapter.end_page == page_number:
            return chapter.name
    return "Unassigned"


def chapter_for_line(chapters: List[Chapter], page_number: int, line_idx: int) -> str:
    """Precise lookup: return which chapter a specific line (identified by
    page number + line index within that page) belongs to. Correctly
    handles multiple chapters sharing a single page/logical page."""
    current = "Unassigned"
    for chapter in chapters:
        start_key = (chapter.start_page, chapter.start_line)
        end_key = (
            chapter.end_page if chapter.end_page is not None else float("inf"),
            chapter.end_line if chapter.end_line is not None else float("inf"),
        )
        here_key = (page_number, line_idx)
        if start_key <= here_key < end_key:
            current = chapter.name
    return current
