"""
retriever.py  –  Load FAISS index once; answer queries with fast sub-50ms retrieval.

Embedding strategy (consistent with build_index.py):
  • Single embedContent call per query (batchEmbedContents for 1 text adds overhead).
  • Persistent HTTP session – reuses TLS/TCP connections.
  • LRU embedding cache – identical queries skip the API entirely.
  • Key rotation on 429.

Usage (standalone test):
    python -m rag_faiss.retriever
"""

import os
import pickle
import time
import numpy as np
import faiss
import requests

from typing import Optional, Dict, Tuple, List
import logging

from rag_faiss.config import (
    GEMINI_API_KEY,
    GEMINI_API_KEYS,
    FAISS_INDEX_PATH,
    INDEX_MAP_PATH,
    PICKLES_DIR,
    TOP_K,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# ── Gemini Embedding REST endpoint ───────────────────────────────────────────
_BASE      = "https://generativelanguage.googleapis.com/v1beta"
EMBED_URL  = f"{_BASE}/{EMBEDDING_MODEL}:embedContent"

# ── Module-level singletons (loaded once at startup) ─────────────────────────
_faiss_index:       Optional[faiss.Index]             = None
_index_map:         Optional[Dict[int, Tuple[str, int]]] = None
_pickle_cache:      Dict[str, List[str]]               = {}
_loaded_successfully: bool                             = False

# Persistent HTTP session – reuses TLS/TCP (~50-150 ms saved per call)
_http_session: Optional[requests.Session] = None

# LRU embedding cache – skip API for repeated queries
_embed_cache:     Dict[str, np.ndarray] = {}
_EMBED_CACHE_MAX  = 256   # bumped from 128 for better hit rate

# Key rotation state
_embed_key_idx: int = 0


# ── HTTP session factory ─────────────────────────────────────────────────────

def _get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.headers.update({
            "Content-Type": "application/json",
            "Connection":   "keep-alive",
        })
    return _http_session


# ── FAISS index loader ───────────────────────────────────────────────────────

def _ensure_loaded() -> None:
    """Load FAISS index and index map into memory on first call."""
    global _faiss_index, _index_map, _loaded_successfully

    if _loaded_successfully:
        return

    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning(
            "[RETRIEVER] FAISS index not found at %s. "
            "RAG disabled until the index is built.",
            FAISS_INDEX_PATH,
        )
        _faiss_index = None
        _index_map   = None
        return

    try:
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(INDEX_MAP_PATH, "rb") as f:
            _index_map = pickle.load(f)
        _loaded_successfully = True
        logger.info("[RETRIEVER] ✅ FAISS index loaded (%d vectors)", _faiss_index.ntotal)
    except Exception as exc:
        logger.error("[RETRIEVER] ❌ Failed to load FAISS index: %s", exc)
        _faiss_index = None
        _index_map   = None


# ── Pickle chunk loader (cached) ─────────────────────────────────────────────

def _load_pickle(pickle_filename: str) -> List[str]:
    if pickle_filename not in _pickle_cache:
        path = os.path.join(PICKLES_DIR, pickle_filename)
        with open(path, "rb") as f:
            _pickle_cache[pickle_filename] = pickle.load(f)
    return _pickle_cache[pickle_filename]


# ── Query embedder ───────────────────────────────────────────────────────────

def _embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string.
    • Returns a (1, dim) float32 array, L2-normalised.
    • Uses LRU cache to skip identical repeated queries.
    • Rotates API keys on 429.
    """
    global _embed_key_idx

    cache_key = text.strip().lower()
    if cache_key in _embed_cache:
        logger.debug("[RETRIEVER] Embedding cache HIT")
        return _embed_cache[cache_key]

    session = _get_http_session()
    keys = GEMINI_API_KEYS if GEMINI_API_KEYS else ([GEMINI_API_KEY] if GEMINI_API_KEY else [])
    if not keys:
        raise RuntimeError("[RETRIEVER] No GEMINI_API_KEY configured")

    last_err: Optional[str] = None
    max_attempts = len(keys) * 3

    for attempt in range(max_attempts):
        key = keys[_embed_key_idx % len(keys)]
        try:
            resp = session.post(
                EMBED_URL,
                headers={"x-goog-api-key": key},
                json={
                    "model":    EMBEDDING_MODEL,
                    "content":  {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_QUERY",
                },
                timeout=(4, 12),   # (connect_timeout, read_timeout)
            )

            if resp.status_code == 429:
                logger.warning(
                    "[RETRIEVER] Key %d/%d rate-limited, rotating …",
                    _embed_key_idx % len(keys) + 1, len(keys),
                )
                _embed_key_idx += 1
                last_err = "rate_limited"
                # Back-off before the next attempt
                time.sleep(2)
                continue

            if not resp.ok:
                logger.error(
                    "[RETRIEVER] Embed %d error → %s",
                    resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()

            data = resp.json()
            if "embedding" in data:
                values = data["embedding"]["values"]
            elif "embeddings" in data:
                values = data["embeddings"][0]["values"]
            else:
                raise RuntimeError(
                    f"[RETRIEVER] Unexpected embed response keys: {list(data.keys())}"
                )

            vec = np.array([values], dtype=np.float32)
            faiss.normalize_L2(vec)

            # Store in LRU cache
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                oldest = next(iter(_embed_cache))
                del _embed_cache[oldest]
            _embed_cache[cache_key] = vec
            return vec

        except requests.exceptions.HTTPError:
            raise   # already logged above; let caller handle
        except Exception as exc:
            last_err = str(exc)
            logger.warning("[RETRIEVER] Embed attempt %d failed: %s", attempt + 1, exc)
            _embed_key_idx += 1

    raise RuntimeError(
        f"[RETRIEVER] Embedding failed after {max_attempts} attempts: {last_err}"
    )


# ── Public retrieval API ─────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """
    Find the top-k most relevant document chunks for a query.

    Returns:
        {
            "context": "<concatenated chunk text>",
            "sources": ["Achievements.txt", ...]
        }
    """
    import time as _time   # local import avoids polluting module namespace

    _ensure_loaded()
    if _faiss_index is None or _index_map is None:
        logger.warning("[RETRIEVER] FAISS index not available – returning empty context")
        return {"context": "", "sources": []}

    t0 = _time.perf_counter()
    query_vec = _embed_query(query)
    embed_ms  = (_time.perf_counter() - t0) * 1000

    t1 = _time.perf_counter()
    distances, ids = _faiss_index.search(query_vec, top_k)
    search_ms = (_time.perf_counter() - t1) * 1000

    logger.info("[RETRIEVER] embed=%.0fms  faiss=%.1fms", embed_ms, search_ms)

    chunks:       List[str] = []
    sources_seen: List[str] = []

    for faiss_id in ids[0]:
        if faiss_id == -1:
            continue
        pickle_filename, chunk_idx = _index_map[int(faiss_id)]
        pickle_chunks = _load_pickle(pickle_filename)
        chunks.append(pickle_chunks[chunk_idx])

        source_txt = pickle_filename.replace(".pkl", ".txt")
        if source_txt not in sources_seen:
            sources_seen.append(source_txt)

    return {
        "context": "\n\n".join(chunks),
        "sources": sources_seen,
    }


# ── Quick CLI smoke-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    _ensure_loaded()

    if _faiss_index is None:
        print("[RETRIEVER] ❌ No FAISS index found – run build_index first.")
    else:
        print(f"[RETRIEVER] Index loaded: {_faiss_index.ntotal} vectors")

        test_queries = [
            "What awards did students win?",
            "CSE Department BC cutoff mark?",
            "What are the hostel facilities?",
        ]
        for q in test_queries:
            start  = time.perf_counter()
            result = retrieve(q)
            ms     = (time.perf_counter() - start) * 1000
            print(f"\nQuery   : {q}")
            print(f"Sources : {result['sources']}")
            print(f"Latency : {ms:.1f} ms")
            print(f"Context : {result['context'][:200]} …")
