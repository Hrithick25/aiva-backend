"""
retriever.py – Load FAISS index once, then answer queries with
sub-50 ms retrieval by looking up pickle chunks via the index map.

Usage (standalone test):
    python -m rag_faiss.retriever
"""

import os
import pickle
import numpy as np
import faiss
import requests

from typing import Optional, Dict, Tuple, List
import logging

from rag_faiss.config import (
    GEMINI_API_KEY,
    FAISS_INDEX_PATH,
    INDEX_MAP_PATH,
    PICKLES_DIR,
    TOP_K,
    EMBEDDING_MODEL,
)

# ── Gemini Embedding REST API ────────────────────────────────────────
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL}:embedContent"

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once) ────────────────────────────
_faiss_index: Optional[faiss.Index] = None
_index_map: Optional[Dict[int, Tuple[str, int]]] = None
_pickle_cache: Dict[str, List[str]] = {}
_loaded_successfully: bool = False

# OPTIMIZED: Persistent HTTP session — reuses TCP/TLS connections (~50-150ms saved per call)
_http_session: Optional[requests.Session] = None


def _get_http_session() -> requests.Session:
    """Get or create a persistent HTTP session with connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.headers.update({"Connection": "keep-alive"})
    return _http_session


def _ensure_loaded():
    """Load FAISS index and index map into memory on first call.
    Re-checks the filesystem until the index is successfully loaded."""
    global _faiss_index, _index_map, _loaded_successfully

    if _loaded_successfully:
        return

    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning(
            "FAISS index not found at %s. RAG will be disabled until the index is built.",
            FAISS_INDEX_PATH,
        )
        _faiss_index = None
        _index_map = None
        return

    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(INDEX_MAP_PATH, "rb") as f:
        _index_map = pickle.load(f)

    _loaded_successfully = True


def _load_pickle(pickle_filename: str) -> List[str]:
    """Load a pickle file, caching it for repeat access."""
    if pickle_filename not in _pickle_cache:
        path = os.path.join(PICKLES_DIR, pickle_filename)
        with open(path, "rb") as f:
            _pickle_cache[pickle_filename] = pickle.load(f)
    return _pickle_cache[pickle_filename]


def _embed_query(text: str) -> np.ndarray:
    """Embed a single query string via REST API and normalise for cosine similarity.
    
    OPTIMIZED: Uses persistent session for connection reuse.
    """
    session = _get_http_session()
    resp = session.post(
        EMBED_URL,
        params={"key": GEMINI_API_KEY},
        json={"content": {"parts": [{"text": text}]}},
        timeout=(5, 10),  # (connect_timeout, read_timeout) — tighter than 15s flat
    )
    resp.raise_for_status()
    values = resp.json()["embedding"]["values"]
    vec = np.array([values], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve the most relevant chunks for a query.

    Returns:
        {
            "context": "...concatenated chunk text...",
            "sources": ["Achievements.txt", ...]
        }
    """
    _ensure_loaded()
    if _faiss_index is None or _index_map is None:
        return {
            "context": "",
            "sources": [],
        }

    query_vec = _embed_query(query)
    distances, ids = _faiss_index.search(query_vec, top_k)

    chunks: List[str] = []
    sources_seen: set = set()
    sources_ordered: List[str] = []

    for faiss_id in ids[0]:
        if faiss_id == -1:
            continue

        pickle_filename, chunk_idx = _index_map[int(faiss_id)]
        pickle_chunks = _load_pickle(pickle_filename)
        chunks.append(pickle_chunks[chunk_idx])

        source_txt = pickle_filename.replace(".pkl", ".txt")
        if source_txt not in sources_seen:
            sources_seen.add(source_txt)
            sources_ordered.append(source_txt)

    return {
        "context": "\n".join(chunks),   # single \n saves tokens vs double
        "sources": sources_ordered,
    }


# ── Quick CLI test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    _ensure_loaded()
    print(f"[RETRIEVER] FAISS index loaded: {_faiss_index.ntotal} vectors")

    test_queries = [
        "What awards did students win?",
        "CSE Department BC cutoff mark?",
    ]

    for q in test_queries:
        start = time.perf_counter()
        result = retrieve(q)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\nQuery: {q}")
        print(f"Sources: {result['sources']}")
        print(f"Latency: {elapsed_ms:.1f} ms")
        print(f"Context (first 300 chars): {result['context'][:300]}")
