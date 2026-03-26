import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import faiss
import requests

from rag_faiss.config import (
    GEMINI_API_KEY,
    GEMINI_API_KEYS,
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


EMBED_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/"
    + EMBEDDING_MODEL
    + ":embedContent"
)


def _embed_single(text: str, key: str) -> List[float]:
    """Embed a single text using the embedContent endpoint."""
    payload = {
        "model": EMBEDDING_MODEL,
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT",
    }
    resp = requests.post(
        EMBED_URL_TEMPLATE,
        params={"key": key},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _embed_texts(texts: List[str], batch_size: int = 20) -> np.ndarray:
    """Embed texts using Gemini embedContent API with automatic key rotation on rate limits."""
    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY found. Set it in your .env file.")

    vecs: List[List[float]] = []
    key_index = 0

    for i, text in enumerate(texts):
        max_attempts = len(GEMINI_API_KEYS) * 3
        attempt = 0
        success = False

        while attempt < max_attempts:
            current_key = GEMINI_API_KEYS[key_index % len(GEMINI_API_KEYS)]
            key_label = f"Key {(key_index % len(GEMINI_API_KEYS)) + 1}/{len(GEMINI_API_KEYS)}"

            try:
                vec = _embed_single(text, current_key)
                vecs.append(vec)
                success = True
                if (i + 1) % 10 == 0 or (i + 1) == len(texts):
                    print(f"  [embed] {i + 1}/{len(texts)} texts embedded ({key_label})")
                time.sleep(0.1)  # small delay to avoid hitting per-minute quota
                break
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    attempt += 1
                    key_index += 1
                    next_key_label = f"Key {(key_index % len(GEMINI_API_KEYS)) + 1}/{len(GEMINI_API_KEYS)}"
                    if attempt % len(GEMINI_API_KEYS) == 0:
                        wait = 65
                        print(f"  [embed] All keys rate-limited. Sleeping {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        print(f"  [embed] {key_label} rate-limited → switching to {next_key_label}")
                else:
                    raise  # non-rate-limit HTTP error — propagate immediately

        if not success:
            raise RuntimeError(f"Failed to embed text #{i} after {max_attempts} attempts across all keys.")

    if not vecs:
        raise RuntimeError("API returned no embeddings.")

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
