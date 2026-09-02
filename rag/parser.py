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

# Separator between the keyword and the number, and between the number
# and the title. Includes plain hyphen plus Unicode en-dash (–), em-dash
# (—), and minus sign (−) — professionally typeset textbooks (and OCR
# output derived from them) very often use these instead of a plain "-".
_SEP = r"[:\-\u2013\u2014\u2212#]?"
# Chapter/unit numbers as either Arabic digits or Roman numerals
# (I, II, III, IV, ... up to a reasonable length to avoid false-matching
# random short words that happen to consist only of I/V/X/L/C/D/M).
_NUM = r"(\d+|[IVXLCDM]{1,7})"

# Each pattern captures (keyword, number, remaining title text). Matching
# is anchored to the start of the line since headers are typically the
# first thing on their line, but tolerant of a trailing colon/dash and
# either numeral style.
CHAPTER_PATTERNS = [
    re.compile(rf"^(chapter)\s*{_SEP}\s*{_NUM}\b\s*{_SEP}\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^(unit)\s*{_SEP}\s*{_NUM}\b\s*{_SEP}\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^(section)\s*{_SEP}\s*{_NUM}\b\s*{_SEP}\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^(module)\s*{_SEP}\s*{_NUM}\b\s*{_SEP}\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^(part)\s*{_SEP}\s*{_NUM}\b\s*{_SEP}\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^(lesson)\s*{_SEP}\s*{_NUM}\b\s*{_SEP}\s*(.*)$", re.IGNORECASE),
    re.compile(rf"^(topic)\s*{_SEP}\s*{_NUM}\b\s*{_SEP}\s*(.*)$", re.IGNORECASE),
]

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> Optional[int]:
    """Convert a Roman numeral string to an int, or None if it isn't a
    valid Roman numeral (guards against the number regex accidentally
    matching a short word that happens to consist only of I/V/X/L/C/D/M
    characters, e.g. "MIX" or "CIVIL")."""
    s = s.upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_VALUES.get(ch)
        if val is None:
            return None
        total += val if val >= prev else -val
        prev = val
    return total if total > 0 else None


def _normalize_number(raw: str) -> Optional[str]:
    """Normalize a matched chapter number — Arabic or Roman — to a plain
    digit string, so "Unit II" and "Unit 2" resolve to the same canonical
    key (e.g. if a Table of Contents uses one numeral style and running
    headers in the body use the other)."""
    if raw.isdigit():
        return raw
    value = _roman_to_int(raw)
    return str(value) if value is not None else None


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
            normalized_number = _normalize_number(number)
            if normalized_number is None:
                # The "number" matched isn't a valid Arabic or Roman
                # numeral (e.g. the regex's Roman-numeral character class
                # happened to match part of an ordinary word) — not a
                # real chapter heading, skip it.
                continue
            canonical_key = f"{keyword.lower()}_{normalized_number}"
            return canonical_key, title_fragment.strip()
    return None


def _is_pattern_heading(line_text: str) -> bool:
    return _match_chapter_pattern(line_text) is not None


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
    rf"^(chapter|unit|section|module|part|lesson|topic)\s*{_SEP}\s*{_NUM}\b\s+(.+?)\s+(\d{{1,4}})\s*$",
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
            normalized_number = _normalize_number(number)
            if normalized_number is None:
                continue
            entries.append(
                {
                    "canonical_key": f"{keyword.lower()}_{normalized_number}",
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


_GENERIC_LABEL_WORDS = {
    "unit", "chapter", "section", "module", "part", "lesson", "topic",
    "contents", "preface", "index", "and", "the", "sindh", "board",
}
_TOP_N_LINES_FOR_TITLE_DETECTION = 5
_MIN_TITLE_REPEATS = 4
_BOILERPLATE_FRACTION = 0.4  # a line on more than 40% of pages is likely a page-wide footer/header, not a chapter title


def _is_plausible_title_text(text: str) -> bool:
    """Reject candidates that are clearly not a chapter title: too
    short/long, mostly non-alphabetic (page numbers, symbols), or a bare
    generic label word with nothing else ("Unit" alone, "Chapter" alone)
    that's just recurring page furniture rather than an actual title."""
    if not (3 <= len(text) <= 70):
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < max(3, len(text) * 0.5):
        return False
    if text.strip().lower() in _GENERIC_LABEL_WORDS:
        return False
    return True


def _detect_repeated_title_chapters(doc: LoadedDocument, toc_pages: set) -> List[Chapter]:
    """Detect chapters whose titles have no "Chapter N"/"Unit N" keyword
    at all — common in some textbook-board books, which simply print the
    bare chapter name (e.g. "Chemical Equilibrium", "Organic Chemistry")
    with no numbering keyword. This can't be caught by keyword-pattern
    matching, but these titles still reliably repeat as a running header
    near the top of every page within that chapter — so this detects
    chapter boundaries from that repetition pattern directly, using
    global frequency to tell a real chapter title apart from page-wide
    boilerplate (e.g. a publisher's name printed on literally every page,
    which repeats far more often than any single chapter title can).
    """
    total_pages = len(doc.pages)
    if total_pages == 0:
        return []

    line_frequency: dict = {}       # normalized_text -> count of pages it appears on
    first_occurrence: dict = {}     # normalized_text -> (page_number, line_idx, original_text)
    all_pages_seen: dict = {}       # normalized_text -> sorted list of page numbers it appeared on

    for page in doc.pages:
        if page.page_number in toc_pages:
            continue
        top_lines = list(enumerate(page.lines))[:_TOP_N_LINES_FOR_TITLE_DETECTION]
        seen_on_this_page = set()
        for line_idx, (line_text, _font_size) in top_lines:
            stripped = line_text.strip()
            if not _is_plausible_title_text(stripped):
                continue
            key = stripped.lower()
            if key in seen_on_this_page:
                continue
            seen_on_this_page.add(key)
            line_frequency[key] = line_frequency.get(key, 0) + 1
            all_pages_seen.setdefault(key, []).append(page.page_number)
            if key not in first_occurrence:
                first_occurrence[key] = (page.page_number, line_idx, stripped)

    def _longest_contiguous_run(pages: List[int], gap_tolerance: int = 2) -> int:
        """Longest run of pages where consecutive entries are within
        `gap_tolerance` of each other (allows for the odd page where an
        image-heavy layout skips the running header)."""
        if not pages:
            return 0
        pages = sorted(pages)
        best = current = 1
        for prev, nxt in zip(pages, pages[1:]):
            if nxt - prev <= gap_tolerance:
                current += 1
                best = max(best, current)
            else:
                current = 1
        return best

    upper_bound = max(_MIN_TITLE_REPEATS, int(_BOILERPLATE_FRACTION * total_pages))
    candidates = []
    for key, freq in line_frequency.items():
        if not (_MIN_TITLE_REPEATS <= freq <= upper_bound):
            continue
        # A real chapter title repeats on one contiguous stretch of pages
        # (the chapter's own length). A recurring template label that
        # just happens to appear once near the start of every chapter
        # (e.g. an administrative "Time Allocation" line in each
        # chapter's overview box) instead shows up as isolated, widely
        # scattered occurrences — require most occurrences to fall in a
        # single contiguous run to tell these apart.
        run_length = _longest_contiguous_run(all_pages_seen[key])
        if run_length >= max(3, int(0.6 * freq)):
            candidates.append(key)

    chapters = [
        Chapter(
            name=first_occurrence[key][2],
            start_page=first_occurrence[key][0],
            start_line=first_occurrence[key][1],
        )
        for key in candidates
    ]
    chapters.sort(key=lambda c: (c.start_page, c.start_line))

    if chapters:
        logger.info(
            "Detected %d chapter(s) via repeated running-title analysis "
            "(no Chapter/Unit keyword found in this document)", len(chapters)
        )
    return chapters


_LARGE_FONT_MULTIPLIER = 2.2       # relative to the document's native median body font size
_LARGE_FONT_MERGE_TOLERANCE = 0.85  # how close a following line's size must be to still count as a title continuation
_LARGE_FONT_MAX_PAGE_FRACTION = 0.35  # exclude text appearing on more than this fraction of native pages (boilerplate)


def _native_median_font_size(doc: LoadedDocument) -> float:
    sizes = [
        size
        for page in doc.pages
        if page.page_number not in doc.ocr_page_numbers
        for _, size in page.lines
        if size > 0
    ]
    return statistics.median(sizes) if sizes else 0.0


def _detect_large_font_title_chapters(doc: LoadedDocument, toc_pages: set) -> List[Chapter]:
    """Detect chapters from a large, visually distinct title font used only
    on each chapter's opening page — for books that neither use a
    "Chapter N"/"Unit N" keyword NOR reprint the title as a repeated
    running header (so Tier 1 and Tier 2 both find nothing), but do give
    each chapter a large bold/display-style title on its first page,
    which native PDF font-size metadata makes detectable directly.

    Only usable on pages with a real embedded text layer (OCR-derived
    "font size" is just approximate pixel height and isn't reliable for
    this). If the whole document is scanned, this tier simply finds
    nothing and detection falls through to manual entry.
    """
    native_page_count = sum(1 for p in doc.pages if p.page_number not in doc.ocr_page_numbers)
    if native_page_count == 0:
        return []

    median_size = _native_median_font_size(doc)
    if median_size <= 0:
        return []
    threshold = median_size * _LARGE_FONT_MULTIPLIER

    occurrences: dict = {}  # normalized text -> list of (page_number, line_idx, display_text)

    for page in doc.pages:
        if page.page_number in toc_pages or page.page_number in doc.ocr_page_numbers:
            continue
        lines = page.lines
        i = 0
        while i < len(lines):
            text, size = lines[i]
            stripped = text.strip()
            if size >= threshold and _is_plausible_title_text(stripped):
                # A title that wraps onto a second line usually keeps a
                # similarly large size on that next line too — merge it
                # in rather than treating it as a separate chapter.
                combined = [stripped]
                j = i + 1
                while (
                    j < len(lines)
                    and lines[j][1] >= threshold * _LARGE_FONT_MERGE_TOLERANCE
                    and _is_plausible_title_text(lines[j][0].strip())
                ):
                    combined.append(lines[j][0].strip())
                    j += 1
                display_name = " ".join(combined)
                key = display_name.lower()
                occurrences.setdefault(key, []).append((page.page_number, i, display_name))
                i = j
            else:
                i += 1

    upper_bound = max(2, int(_LARGE_FONT_MAX_PAGE_FRACTION * native_page_count))
    chapters = []
    for key, occ_list in occurrences.items():
        if len(occ_list) > upper_bound:
            # Appears on too large a fraction of native pages to be a
            # single chapter's title — almost certainly page furniture
            # like a publisher name printed in large decorative type.
            continue
        earliest = min(occ_list, key=lambda o: (o[0], o[1]))
        chapters.append(Chapter(name=earliest[2], start_page=earliest[0], start_line=earliest[1]))

    chapters.sort(key=lambda c: (c.start_page, c.start_line))

    if chapters:
        logger.info(
            "Detected %d chapter(s) via large-font title analysis "
            "(no keyword or repeated running header found in this document)",
            len(chapters),
        )
    return chapters


_HEADING_KEYWORDS = {"unit", "chapter", "section", "module", "part", "lesson", "topic"}
_BARE_NUMBER_LINE = re.compile(r"^\d{1,3}$")


def _looks_like_badge_title_line(text: str) -> bool:
    """Loose check for whether a line could be part of a multi-line
    chapter title in the "badge" layout (see _detect_badge_style_chapters)
    — deliberately more permissive than _is_plausible_title_text since
    short title fragments here are common (e.g. a lone "TO" between
    "INTRODUCTION" and "TRIGONOMETRY")."""
    if not text or len(text) > 40:
        return False
    if text.isdigit():
        return False
    lowered = text.lower()
    if "=" in text or "weightage" in lowered or "sindh textbook board" in lowered:
        return False
    return True


def _detect_badge_style_chapters(doc: LoadedDocument, toc_pages: set) -> List[Chapter]:
    """Detect chapter headings in a "badge" layout where the design
    splits the unit number and the keyword across separate lines instead
    of printing them together, e.g. (as extracted from the PDF's text
    layer, in this exact order):

        17
        17
        Unit
        SETS AND
        FUNCTIONS
        1

    Here "17" is the actual sequential unit number (printed twice as a
    decorative badge), followed by the bare keyword "Unit" on its own
    line, followed by the title spread across 1-3 short lines. This
    doesn't repeat as a running header — it's a one-off "chapter opener"
    page design — but the signature (a doubled number immediately
    followed by a bare heading keyword) is distinctive enough to trust
    from a single occurrence, unlike a generic "N. Title" line.
    """
    chapters: List[Chapter] = []
    seen_keys: set = set()

    for page in doc.pages:
        if page.page_number in toc_pages:
            continue
        lines = page.lines
        n = len(lines)
        for i in range(n - 2):
            line1 = lines[i][0].strip()
            line2 = lines[i + 1][0].strip()
            line3 = lines[i + 2][0].strip()

            if not (_BARE_NUMBER_LINE.match(line1) and line1 == line2):
                continue
            if line3.lower() not in _HEADING_KEYWORDS:
                continue

            badge_number = line1
            keyword = line3
            canonical_key = f"{keyword.lower()}_{badge_number}"
            if canonical_key in seen_keys:
                continue

            title_parts = []
            j = i + 3
            while j < n and len(title_parts) < 4:
                candidate = lines[j][0].strip()
                if not candidate:
                    j += 1
                    continue
                if not _looks_like_badge_title_line(candidate):
                    break
                title_parts.append(candidate)
                j += 1

            if not title_parts:
                continue

            display_name = f"{keyword.title()} {badge_number}: {' '.join(title_parts)}"
            chapters.append(
                Chapter(name=display_name, start_page=page.page_number, start_line=i)
            )
            seen_keys.add(canonical_key)

    chapters.sort(key=lambda c: (c.start_page, c.start_line))
    if chapters:
        logger.info(
            "Detected %d chapter(s) via badge-style split-line heading pattern", len(chapters)
        )
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

    # Tier 2: for books whose chapters have no "Chapter N"/"Unit N"
    # keyword at all (titled directly, e.g. "Chemical Equilibrium"),
    # detect chapter boundaries from repeated running-header titles
    # instead. Only attempted if keyword-pattern detection found fewer
    # than 2 chapters — a strong sign this document doesn't use that
    # naming convention, rather than that most chapters were just missed.
    if len(chapters) < 2:
        repeated_title_chapters = _detect_repeated_title_chapters(doc, toc_pages)
        if len(repeated_title_chapters) > len(chapters):
            chapters = repeated_title_chapters

    # Tier 3: for books whose chapters have neither a keyword NOR a
    # repeated running header — but do give each chapter a large,
    # visually distinct title (e.g. big bold display type) on just its
    # opening page. Only meaningful on pages with real embedded font
    # metadata, so this naturally does nothing for fully-scanned books.
    if len(chapters) < 2:
        large_font_chapters = _detect_large_font_title_chapters(doc, toc_pages)
        if len(large_font_chapters) > len(chapters):
            chapters = large_font_chapters

    # NOTE: an earlier version of this function had a third fallback tier
    # that matched single, non-repeating "N. Title" lines (e.g. "3. Deep
    # Learning") anywhere in the body text. It was removed: without
    # requiring repetition as a running header the way Tier 2 does, it
    # had no reliable way to distinguish an actual chapter heading from
    # an ordinary numbered list item, exercise question, or MCQ option —
    # and in practice it was confidently mislabeling exercise questions
    # as chapters. Returning no chapters (prompting manual entry) is
    # more honest than a fallback that's frequently wrong.

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
