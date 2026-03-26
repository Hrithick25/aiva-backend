"""
build_index.py  –  Build a FAISS index from all knowledge-base text files.

Embedding strategy:
  • Uses batchEmbedContents (up to 100 texts per call) for speed.
  • Falls back to one-at-a-time embedContent when a single chunk is too large.
  • Automatically rotates Gemini API keys on 429 rate-limits.
  • Pre-filters empty, too-short, or oversized chunks before embedding.

Usage:
    python -m rag_faiss.build_index
"""

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

# ── API endpoints ────────────────────────────────────────────────────────────
_BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBED_SINGLE_URL = f"{_BASE}/{EMBEDDING_MODEL}:embedContent"
EMBED_BATCH_URL  = f"{_BASE}/{EMBEDDING_MODEL}:batchEmbedContents"

# Gemini hard limits
_MAX_EMBED_CHARS  = 9_000   # ~2 000 tokens – safely within the 2 048-token limit
_MAX_BATCH_SIZE   = 100     # Gemini batchEmbedContents allows up to 100 requests/call
_MIN_CHUNK_CHARS  = 10      # discard trivially short chunks

# Persistent HTTP session reuses TLS/TCP connections
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


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


# ── Key-rotation helper ──────────────────────────────────────────────────────

def _pick_key(key_index: int) -> str:
    keys = GEMINI_API_KEYS if GEMINI_API_KEYS else [GEMINI_API_KEY]
    return keys[key_index % len(keys)]


def _num_keys() -> int:
    return len(GEMINI_API_KEYS) if GEMINI_API_KEYS else 1


# ── Single embed (fallback for oversized / non-batchable chunks) ─────────────

def _embed_single(text: str, key: str) -> List[float]:
    """Embed one text via embedContent; raises on any HTTP error."""
    payload = {
        "model":    EMBEDDING_MODEL,
        "content":  {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT",
    }
    resp = _session.post(
        EMBED_SINGLE_URL,
        headers={"x-goog-api-key": key},
        json=payload,
        timeout=30,
    )
    if not resp.ok:
        print(f"  [embed-single] ❌ {resp.status_code}: {resp.text[:500]}")
    resp.raise_for_status()

    data = resp.json()
    if "embedding" in data:
        return data["embedding"]["values"]
    raise RuntimeError(f"Unexpected single-embed response keys: {list(data.keys())}")


# ── Batch embed (main path) ──────────────────────────────────────────────────

def _embed_batch(texts: List[str], key: str) -> List[List[float]]:
    """
    Embed up to _MAX_BATCH_SIZE texts in a single batchEmbedContents call.
    Returns a list of float vectors in the same order as `texts`.
    Raises on HTTP error.
    """
    payload = {
        "requests": [
            {
                "model":    EMBEDDING_MODEL,
                "content":  {"parts": [{"text": t}]},
                "taskType": "RETRIEVAL_DOCUMENT",
            }
            for t in texts
        ]
    }
    resp = _session.post(
        EMBED_BATCH_URL,
        headers={"x-goog-api-key": key},
        json=payload,
        timeout=60,
    )
    if not resp.ok:
        print(f"  [embed-batch] ❌ {resp.status_code}: {resp.text[:800]}")
    resp.raise_for_status()

    data = resp.json()
    # Response: { "embeddings": [ { "values": [...] }, ... ] }
    if "embeddings" not in data:
        raise RuntimeError(f"Unexpected batch-embed response keys: {list(data.keys())}")
    return [item["values"] for item in data["embeddings"]]


# ── Orchestrator with key rotation + retries ─────────────────────────────────

def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed all texts using batchEmbedContents.
    • Splits into batches of _MAX_BATCH_SIZE.
    • Rotates API keys on 429.
    • Retries each batch up to (num_keys * 3) times before giving up.
    Returns np.float32 array of shape (N, dim), L2-normalised.
    """
    if not (GEMINI_API_KEYS or GEMINI_API_KEY):
        raise RuntimeError("No GEMINI_API_KEY found. Set it in your .env file.")

    # Filter invalid texts
    texts = [t for t in texts if t and t.strip() and len(t.strip()) >= _MIN_CHUNK_CHARS]
    if not texts:
        raise RuntimeError("All text chunks were empty/too-short after filtering.")

    all_vecs: List[List[float]] = []
    key_index = 0
    total = len(texts)

    # Process in batches
    for batch_start in range(0, total, _MAX_BATCH_SIZE):
        batch = texts[batch_start : batch_start + _MAX_BATCH_SIZE]
        max_attempts = _num_keys() * 3
        success = False

        for attempt in range(max_attempts):
            key = _pick_key(key_index)
            key_label = f"key {(key_index % _num_keys()) + 1}/{_num_keys()}"
            try:
                vecs = _embed_batch(batch, key)
                all_vecs.extend(vecs)
                success = True
                end_idx = min(batch_start + _MAX_BATCH_SIZE, total)
                print(f"  [embed] ✅ {end_idx}/{total} texts | batch {batch_start//  _MAX_BATCH_SIZE + 1} | {key_label}")
                time.sleep(0.15)   # gentle pacing to avoid rate-limits
                break

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    key_index += 1
                    next_label = f"key {(key_index % _num_keys()) + 1}/{_num_keys()}"
                    all_rotated = (attempt + 1) % _num_keys() == 0
                    if all_rotated:
                        wait = 65
                        print(f"  [embed] ⏳ All keys rate-limited. Sleeping {wait}s …")
                        time.sleep(wait)
                    else:
                        print(f"  [embed] 🔄 {key_label} rate-limited → {next_label}")
                else:
                    # For non-429 HTTP errors, log and re-raise immediately
                    raise

        if not success:
            raise RuntimeError(
                f"Failed to embed batch starting at index {batch_start} "
                f"after {max_attempts} attempts across all keys."
            )

    if not all_vecs:
        raise RuntimeError("No embeddings were produced.")

    arr = np.asarray(all_vecs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


# ── Index builder ────────────────────────────────────────────────────────────

def build_index() -> Tuple[int, int]:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in your backend .env before building the FAISS index."
        )

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
