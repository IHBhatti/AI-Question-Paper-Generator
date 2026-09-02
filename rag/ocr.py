"""
rag/ocr.py
----------
OCR fallback for scanned PDFs — the norm rather than the exception for
textbook-board material (e.g. Sindh Textbook Board Physics/Chemistry/
Biology/Computer/Mathematics books), which are almost always page-image
scans with no embedded text layer at all.

Handles two real-world problems these scans commonly have:
1. No extractable text layer -> rasterize the page and run Tesseract OCR.
2. Pages scanned upside-down or rotated -> auto-detect orientation with
   Tesseract's OSD and correct it before OCR, otherwise text comes out
   as scrambled gibberish.

Results are cached to disk per (file_hash, page_number) so re-processing
the same document never re-runs OCR, since OCR is by far the slowest
step in the pipeline (a few seconds per page).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from config import OCR_CACHE_DIR_NAME, OCR_DPI, OCR_LANGUAGES, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = VECTORSTORE_DIR / OCR_CACHE_DIR_NAME
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# On Windows, Tesseract often isn't on PATH even after installing it via the
# UB-Mannheim installer. Allow pointing at it explicitly via an environment
# variable so users don't have to fight PATH configuration.
_TESSERACT_CMD_OVERRIDE = os.environ.get("TESSERACT_CMD")
if _TESSERACT_CMD_OVERRIDE:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD_OVERRIDE
elif os.name == "nt":
    _default_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(_default_windows_path).exists():
        pytesseract.pytesseract.tesseract_cmd = _default_windows_path


class TesseractNotAvailableError(RuntimeError):
    """Raised with clear, actionable setup instructions when the Tesseract
    OCR *engine* (a system program, not a Python package) can't be found."""


def ensure_tesseract_available() -> None:
    """Fail fast with a clear, actionable message if Tesseract isn't
    installed, instead of letting a cryptic error surface deep inside an
    OCR call. `pip install pytesseract` only installs a Python wrapper —
    it does not install the actual OCR engine."""
    cmd = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(cmd) is None and not Path(cmd).exists():
        raise TesseractNotAvailableError(
            "Tesseract OCR engine not found. This document appears to be "
            "scanned and needs OCR, but the Tesseract program itself isn't "
            "installed (pip installing 'pytesseract' only installs a "
            "Python wrapper, not the engine). Install it with:\n"
            "  - Windows: download from "
            "https://github.com/UB-Mannheim/tesseract/wiki, then either "
            "add it to PATH or set the TESSERACT_CMD environment variable "
            "to its full path (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)\n"
            "  - macOS: brew install tesseract\n"
            "  - Linux: sudo apt-get install -y tesseract-ocr\n"
            "  - Streamlit Community Cloud: make sure packages.txt "
            "(containing 'tesseract-ocr') is committed to your repo"
        )


def _cache_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.json"


def load_ocr_cache(file_hash: str) -> Optional[dict]:
    """Return {page_number: {"text": ..., "lines": [[text, size], ...]}}
    if this document was already OCR'd before, else None."""
    path = _cache_path(file_hash)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_ocr_cache(file_hash: str, cache: dict) -> None:
    path = _cache_path(file_hash)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _detect_rotation(image: Image.Image) -> int:
    """Return the rotation (0/90/180/270) Tesseract's OSD thinks would
    make this page upright. Falls back to 0 if OSD can't decide, which
    happens on very sparse or low-contrast pages — better to OCR
    as-is than to crash the whole pipeline."""
    try:
        osd = pytesseract.image_to_osd(image)
        for line in osd.splitlines():
            if line.startswith("Rotate:"):
                return int(line.split(":")[1].strip())
    except Exception as e:
        logger.debug("OSD failed, assuming no rotation needed: %s", e)
    return 0


def _page_to_image(page: "fitz.Page", dpi: int = OCR_DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def ocr_page(page: "fitz.Page", dpi: int = OCR_DPI) -> Tuple[str, List[tuple]]:
    """OCR a single PyMuPDF page, auto-correcting orientation first.

    Returns (full_text, lines) where `lines` is a list of
    (line_text, approx_font_size) tuples, mirroring the shape used for
    native-text PDFs so downstream code (heading/chapter detection) works
    identically regardless of whether a page was OCR'd or text-extracted.
    """
    image = _page_to_image(page, dpi=dpi)

    rotation = _detect_rotation(image)
    if rotation:
        image = image.rotate(rotation, expand=True)

    # image_to_data gives word-level boxes with line/block grouping and
    # pixel height, which we use as a stand-in for "font size" so the
    # same heading-detection heuristic used for native PDFs still works.
    data = pytesseract.image_to_data(
        image, lang=OCR_LANGUAGES, output_type=pytesseract.Output.DICT
    )

    lines_map = {}
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        height = data["height"][i]
        lines_map.setdefault(key, {"words": [], "heights": []})
        lines_map[key]["words"].append(word)
        lines_map[key]["heights"].append(height)

    lines: List[tuple] = []
    for key in sorted(lines_map.keys()):
        entry = lines_map[key]
        line_text = " ".join(entry["words"]).strip()
        if not line_text:
            continue
        avg_height = sum(entry["heights"]) / len(entry["heights"])
        lines.append((line_text, float(avg_height)))

    full_text = "\n".join(lt for lt, _ in lines)
    return full_text, lines


def ocr_document_pages(
    doc: "fitz.Document",
    file_hash: str,
    page_numbers_needing_ocr: List[int],
    progress_callback=None,
) -> dict:
    """OCR only the pages that need it (native text was too sparse),
    using and updating a persistent cache. Returns
    {page_number: {"text": str, "lines": [[text, size], ...]}}.

    `progress_callback(done, total)` is called after each page if given,
    so the Streamlit UI can render a progress bar during what is by far
    the slowest step in the pipeline.
    """
    if page_numbers_needing_ocr:
        ensure_tesseract_available()

    cache = load_ocr_cache(file_hash) or {}
    total = len(page_numbers_needing_ocr)

    for i, page_number in enumerate(page_numbers_needing_ocr, start=1):
        cache_key = str(page_number)
        if cache_key not in cache:
            try:
                page = doc[page_number - 1]  # fitz pages are 0-indexed
                text, lines = ocr_page(page)
                cache[cache_key] = {"text": text, "lines": [[t, s] for t, s in lines]}
            except Exception as e:
                # A single corrupted/unreadable page shouldn't block the
                # rest of the document — record it as empty and move on.
                # Without this, one bad page would also get "stuck" on
                # retry, since the incremental cache would keep skipping
                # every OTHER already-OCR'd page and hitting the same
                # failure on this one every time.
                logger.warning(
                    "OCR failed on page %d (%s) — treating it as blank and continuing",
                    page_number, e,
                )
                cache[cache_key] = {"text": "", "lines": []}
            # Save incrementally so a crash/interrupt doesn't lose earlier work.
            save_ocr_cache(file_hash, cache)

        if progress_callback:
            progress_callback(i, total)

    return cache
