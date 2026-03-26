"""
build_index.py  –  Build a FAISS index from all knowledge-base text files.

Embedding strategy:
  • Local HuggingFace sentence-transformers (no API keys, no rate limits).
  • Model: BAAI/bge-base-en-v1.5 (768-dim, MTEB=61.6, Top-5=84.7%).
  • Pre-filters empty, too-short, or oversized chunks before embedding.
  • FAISS IndexFlatIP (exact cosine on L2-normalised vectors).

Usage:
    python -m rag_faiss.build_index
"""

import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import faiss

from rag_faiss.config import (
    KNOWLEDGE_FILES,
    EMBEDDINGS_DIR,
    PICKLES_DIR,
    FAISS_INDEX_PATH,
    INDEX_MAP_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from rag_faiss.embedder import embed_texts as _embed_texts_local

# Chunk limits
_MAX_EMBED_CHARS  = 9_000
_MIN_CHUNK_CHARS  = 10


# ── Text helpers ─────────────────────────────────────────────────────────────

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunk_size = max(100, int(chunk_size))
    overlap    = max(0, min(int(overlap), chunk_size - 1))
    step       = max(1, chunk_size - overlap)

    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk[:_MAX_EMBED_CHARS])  # hard-truncate oversized chunks
        start += step
    return chunks


# ── Embedding wrapper ─────────────────────────────────────────────────────────

def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed all texts using the local HuggingFace model.
    No API keys, no rate limits, no quotas.
    Returns np.float32 array of shape (N, dim), L2-normalised.
    """
    # Filter invalid texts
    texts = [t for t in texts if t and t.strip() and len(t.strip()) >= _MIN_CHUNK_CHARS]
    if not texts:
        raise RuntimeError("All text chunks were empty/too-short after filtering.")

    return _embed_texts_local(texts)


# ── Index builder ────────────────────────────────────────────────────────────

def build_index() -> Tuple[int, int]:

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(PICKLES_DIR,    exist_ok=True)

    index_map:   Dict[int, Tuple[str, int]] = {}
    all_vectors: List[np.ndarray] = []
    next_id      = 0
    total_chunks = 0

    for name, file_path in KNOWLEDGE_FILES.items():
        if not os.path.exists(file_path):
            print(f"[build_index] ⚠️  Skipping missing file: {file_path}")
            continue

        raw    = _read_text_file(file_path)
        chunks = _chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"[build_index] ⚠️  No valid chunks for: {file_path}")
            continue

        print(f"[build_index] 📄 {name}: {len(chunks)} chunks – embedding …")

        # Persist chunks to pickle BEFORE embedding (so re-runs can skip if desired)
        pickle_filename = f"{name}.pkl"
        with open(os.path.join(PICKLES_DIR, pickle_filename), "wb") as pf:
            pickle.dump(chunks, pf)

        vecs = _embed_texts(chunks)
        all_vectors.append(vecs)

        for i in range(len(chunks)):
            index_map[next_id] = (pickle_filename, i)
            next_id += 1

        total_chunks += len(chunks)
        print(f"[build_index] ✅ {name}: {len(chunks)} chunks embedded")

    if not all_vectors:
        raise RuntimeError("No documents/chunks found. Check KNOWLEDGE_FILES paths.")

    vectors = np.vstack(all_vectors)
    dim     = int(vectors.shape[1])

    # IndexFlatIP = exact cosine search on L2-normalised vectors (ideal for <10K vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(INDEX_MAP_PATH, "wb") as f:
        pickle.dump(index_map, f)

    return total_chunks, index.ntotal


if __name__ == "__main__":
    t0 = time.time()
    chunks, ntotal = build_index()
    elapsed = time.time() - t0
    print(f"\n[build_index] ✅ Done in {elapsed:.1f}s")
    print(f"[build_index]    Vectors : {ntotal}")
    print(f"[build_index]    Chunks  : {chunks}")
    print(f"[build_index]    Index   → {FAISS_INDEX_PATH}")
    print(f"[build_index]    Map     → {INDEX_MAP_PATH}")
