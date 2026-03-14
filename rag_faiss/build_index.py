import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import faiss
import requests

from rag_faiss.config import (
    GEMINI_API_KEY,
    KNOWLEDGE_FILES,
    EMBEDDINGS_DIR,
    PICKLES_DIR,
    FAISS_INDEX_PATH,
    INDEX_MAP_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    EMBEDDING_MODEL,
)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunk_size = max(100, int(chunk_size))
    overlap = max(0, int(overlap))
    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL}:embedContent"


def _embed_single(text: str) -> List[float]:
    """Embed a single text using the Gemini REST API."""
    resp = requests.post(
        EMBED_URL,
        params={"key": GEMINI_API_KEY},
        json={"content": {"parts": [{"text": text}]}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _embed_texts(texts: List[str]) -> np.ndarray:
    vecs: List[List[float]] = []
    for i, t in enumerate(texts):
        vecs.append(_embed_single(t))
        # Gemini free tier rate limit: ~1500 RPM, add small delay to be safe
        if (i + 1) % 50 == 0:
            print(f"  [embed] {i + 1}/{len(texts)} done, pausing briefly...")
            time.sleep(1)
    arr = np.asarray(vecs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def build_index() -> Tuple[int, int]:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in your backend .env before building the FAISS index."
        )

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(PICKLES_DIR, exist_ok=True)

    index_map: Dict[int, Tuple[str, int]] = {}
    all_vectors: List[np.ndarray] = []
    next_id = 0
    total_chunks = 0

    for name, file_path in KNOWLEDGE_FILES.items():
        if not os.path.exists(file_path):
            print(f"[build_index] Skipping missing file: {file_path}")
            continue

        raw = _read_text_file(file_path)
        chunks = _chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"[build_index] No chunks for: {file_path}")
            continue

        pickle_filename = f"{name}.pkl"
        with open(os.path.join(PICKLES_DIR, pickle_filename), "wb") as pf:
            pickle.dump(chunks, pf)

        vecs = _embed_texts(chunks)
        all_vectors.append(vecs)

        for i in range(len(chunks)):
            index_map[next_id] = (pickle_filename, i)
            next_id += 1

        total_chunks += len(chunks)
        print(f"[build_index] {name}: {len(chunks)} chunks")

    if not all_vectors:
        raise RuntimeError("No documents/chunks found. Check KNOWLEDGE_FILES paths.")

    vectors = np.vstack(all_vectors)
    dim = int(vectors.shape[1])

    # Use IndexFlatIP (inner product = cosine similarity on L2-normalized vectors)
    # Note: IndexHNSWFlat crashes on Windows/Python 3.8, and FlatIP provides
    # exact search which is ideal for small datasets (<10K vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(INDEX_MAP_PATH, "wb") as f:
        pickle.dump(index_map, f)

    return total_chunks, index.ntotal


if __name__ == "__main__":
    chunks, ntotal = build_index()
    print(f"[build_index] ✅ Built FAISS index: {ntotal} vectors from {chunks} chunks")
    print(f"[build_index] Index written to: {FAISS_INDEX_PATH}")
    print(f"[build_index] Index map written to: {INDEX_MAP_PATH}")
