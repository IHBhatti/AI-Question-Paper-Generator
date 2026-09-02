"""
config.py
----------
Central configuration for the AI Test Paper Generator.

All tunable constants, paths, and default values live here so the rest
of the codebase never hard-codes "magic" values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Base directories
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in (DATA_DIR, UPLOADS_DIR, VECTORSTORE_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Load variables from a .env file in the project root (e.g. GROQ_API_KEY=...)
# into the process environment. This must happen before GROQ_API_KEY is
# read below, or a .env file gets silently ignored.
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------------------------------
# Groq LLM configuration (FREE tier)
# --------------------------------------------------------------------------
# Get a free key at https://console.groq.com/keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Any current Groq-hosted free model works. Kept as a constant so it can be
# swapped in one place if Groq deprecates a model.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

GROQ_TEMPERATURE = 0.3
# Groq's free/on-demand tier enforces a combined prompt+completion tokens
# PER MINUTE limit (e.g. 8000 TPM for some models) — NOT just a per-request
# cap. Requesting many questions (especially long questions, which need a
# full written answer) in a single call can easily exceed that. Rather
# than one large call per paper, generation is now split into smaller
# batches (see rag/generator.py) — this keeps each individual call's
# max_tokens comfortably low, and MAX_CONTEXT_CHUNKS_PER_CALL keeps the
# prompt side bounded too, regardless of how many chapters are selected.
GROQ_MAX_TOKENS = 4096
MAX_CONTEXT_CHUNKS_PER_CALL = 24
MAX_MCQ_PER_CALL = 10
MAX_SHORT_PER_CALL = 6
MAX_LONG_PER_CALL = 2

# --------------------------------------------------------------------------
# Embedding model configuration (FREE, local, HuggingFace)
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
# Alternative high-quality option: "BAAI/bge-small-en-v1.5"

EMBEDDING_DEVICE = "cpu"  # keep CPU-only so the app runs anywhere for free

# --------------------------------------------------------------------------
# OCR configuration (for scanned textbook PDFs — e.g. Sindh Textbook Board
# Physics/Chemistry/Biology/Computer/Mathematics books, which are almost
# always scanned page-images with no real text layer)
# --------------------------------------------------------------------------
OCR_ENABLED = True
# If a page's native (embedded) text layer has fewer than this many
# characters, treat it as a scanned image page and fall back to OCR.
OCR_MIN_NATIVE_CHARS = 40
# Render resolution used when rasterizing a page for OCR. 200 DPI is a
# good speed/accuracy tradeoff for textbook scans; raise to 250-300 for
# small/blurry print at the cost of roughly 2x processing time.
OCR_DPI = 200
OCR_LANGUAGES = "eng"
# Cache extracted OCR text per document (keyed by file hash) so
# re-processing the same PDF never re-runs OCR.
OCR_CACHE_DIR_NAME = "ocr_cache"

# --------------------------------------------------------------------------
# Chunking configuration
# --------------------------------------------------------------------------
CHUNK_SIZE = 900          # characters per chunk
CHUNK_OVERLAP = 150       # characters of overlap between chunks

# --------------------------------------------------------------------------
# Retrieval configuration
# --------------------------------------------------------------------------
DEFAULT_TOP_K = 8         # chunks retrieved per generation call
MAX_TOP_K = 20

# --------------------------------------------------------------------------
# Question generation defaults
# --------------------------------------------------------------------------
QUESTION_TYPES = ["MCQs", "Short Questions", "Long Questions", "Mixed Paper"]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard", "Mixed"]
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

DEFAULT_MARKS = {"MCQ": 1, "Short": 5, "Long": 10}

MIXED_PAPER_DEFAULT_SPLIT = {
    "mcq_ratio": 0.65,
    "short_ratio": 0.25,
    "long_ratio": 0.10,
}

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "app.log"
