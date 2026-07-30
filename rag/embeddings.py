"""
rag/embeddings.py
-----------------
Wraps HuggingFace sentence-transformer embeddings and FAISS vector
storage. Everything here runs locally and is 100% free (no API calls).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Lazily instantiate (and cache) the HuggingFace embedding model so it
    is only loaded into memory once per process."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Embed all chunks and build an in-memory FAISS index."""
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks.")
    embeddings = get_embedding_model()
    logger.info("Embedding %d chunks into FAISS...", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, name: str) -> Path:
    """Persist a FAISS index to disk under vectorstore/<name>/ so a user's
    session can be reloaded without re-embedding (duplicate-document
    detection in app.py relies on this)."""
    out_dir = VECTORSTORE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))
    logger.info("Saved vector store to %s", out_dir)
    return out_dir


def save_chunks_cache(chunks: List[Document], name: str) -> None:
    """Cache the chunk list alongside the FAISS index (same `name` key,
    typically the document's file hash) so a fully cached document can be
    reloaded without re-running OCR, chunking, or embedding at all."""
    out_dir = VECTORSTORE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_chunks_cache(name: str) -> Optional[List[Document]]:
    path = VECTORSTORE_DIR / name / "chunks.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_vectorstore(name: str) -> Optional[FAISS]:
    """Load a previously saved FAISS index, if it exists."""
    out_dir = VECTORSTORE_DIR / name
    if not out_dir.exists():
        return None
    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(out_dir), embeddings, allow_dangerous_deserialization=True
    )


def compute_and_cache_hash_index(file_hash: str) -> Path:
    """Return the path used to track which file hashes have already been
    processed, enabling duplicate-document detection."""
    return VECTORSTORE_DIR / "processed_hashes.pkl"


def is_duplicate_document(file_hash: str) -> bool:
    index_path = compute_and_cache_hash_index(file_hash)
    if not index_path.exists():
        return False
    with open(index_path, "rb") as f:
        hashes = pickle.load(f)
    return file_hash in hashes


def register_document_hash(file_hash: str) -> None:
    index_path = compute_and_cache_hash_index(file_hash)
    hashes = set()
    if index_path.exists():
        with open(index_path, "rb") as f:
            hashes = pickle.load(f)
    hashes.add(file_hash)
    with open(index_path, "wb") as f:
        pickle.dump(hashes, f)
