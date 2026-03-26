import json
import os
import hashlib
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from config.settings import (
    CHROMA_PERSIST_DIR,
    CHUNK_COLLECTION_NAME,
    KNOWLEDGE_FILES,
    CHUNK_RESULTS,
    ROUTER_MAX_SOURCES,
    ROUTER_TEXT_LIMIT,
)

_source_routing_context = {}
_STATE_FILE = os.path.join(CHROMA_PERSIST_DIR, "knowledge_base_state.json")


def _get_client():
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def _read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def _load_knowledge_sources() -> tuple[list[dict], dict[str, str], str]:
    sources = []
    source_routing_context = {}
    signature_parts = []

    for source_name, file_path in KNOWLEDGE_FILES.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Knowledge file not found at {file_path}")

        raw_text = _read_text_file(file_path)
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        signature_parts.append(f"{source_name}:{content_hash}")

        if not raw_text:
            continue

        sources.append(
            {
                "source_name": source_name,
                "file_path": file_path,
                "raw_text": raw_text,
            }
        )
        source_routing_context[source_name] = _build_routing_text(source_name, raw_text)

    signature = hashlib.sha256("|".join(signature_parts).encode("utf-8")).hexdigest()
    return sources, source_routing_context, signature


def _read_cached_signature() -> str | None:
    if not os.path.exists(_STATE_FILE):
        return None

    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    return payload.get("signature")


def _write_cached_signature(signature: str, chunk_count: int) -> None:
    payload = {
        "signature": signature,
        "chunk_count": chunk_count,
    }
    with open(_STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def _get_existing_collection(client, ef):
    try:
        return client.get_collection(
            name=CHUNK_COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception:
        return None


def _build_routing_text(source_name: str, text: str) -> str:
    normalized_text = " ".join(text.split())
    return f"{source_name}\n\n{normalized_text[:ROUTER_TEXT_LIMIT]}"


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def _route_sources(query: str) -> list[str]:
    """Return top-N sources. FAISS handles semantic matching — no LLM routing needed."""
    available_sources = list(_source_routing_context.keys())
    return available_sources[:ROUTER_MAX_SOURCES]


def load_knowledge_base():
    """Load the configured text sources into the chunk collection and source-routing cache."""
    client = _get_client()
    ef = _get_embedding_fn()
    sources, source_routing_context, current_signature = _load_knowledge_sources()

    global _source_routing_context
    _source_routing_context = source_routing_context

    cached_signature = _read_cached_signature()
    existing_collection = _get_existing_collection(client, ef)

    if existing_collection is not None and cached_signature == current_signature:
        print(
            f"[RAG] Reusing existing ChromaDB embeddings for {len(source_routing_context)} source documents."
        )
        return existing_collection

    if existing_collection is not None:
        client.delete_collection(name=CHUNK_COLLECTION_NAME)

    chunk_collection = client.get_or_create_collection(
        name=CHUNK_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []
    for source in sources:
        source_name = source["source_name"]
        file_path = source["file_path"]
        raw_text = source["raw_text"]
        chunks = splitter.split_text(raw_text)

        for index, chunk in enumerate(chunks):
            chunk_ids.append(f"chunk::{source_name}::{index}")
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                "source": source_name,
                "file_name": os.path.basename(file_path),
                "chunk_index": index,
                "level": "chunk",
            })

    if chunk_ids:
        chunk_collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas,
        )

    _write_cached_signature(current_signature, len(chunk_ids))

    print(
        f"[RAG] Indexed {len(source_routing_context)} source documents and {len(chunk_ids)} chunks into ChromaDB."
    )
    return chunk_collection


def query_knowledge_base(query: str, n_results: int = CHUNK_RESULTS) -> dict:
    """Route the query with Gemini, then retrieve the best chunks from the selected sources."""
    if not _source_routing_context:
        chunk_collection = load_knowledge_base()
    else:
        client = _get_client()
        ef = _get_embedding_fn()
        chunk_collection = client.get_collection(
            name=CHUNK_COLLECTION_NAME,
            embedding_function=ef,
        )

    routed_sources = _route_sources(query)

    if not routed_sources:
        return {"context": "", "sources": []}

    chunk_limit = max(n_results, 2)
    retrieved_chunks = []
    for source in routed_sources:
        source_results = chunk_collection.query(
            query_texts=[query],
            n_results=chunk_limit,
            where={"source": source},
        )

        documents = source_results.get("documents", [[]])[0]
        distances = source_results.get("distances", [[]])[0]
        metadatas = source_results.get("metadatas", [[]])[0]

        for document, distance, metadata in zip(documents, distances, metadatas):
            retrieved_chunks.append(
                {
                    "document": document,
                    "distance": distance,
                    "source": metadata.get("source", source),
                }
            )

    retrieved_chunks.sort(key=lambda item: item["distance"])
    selected_chunks = retrieved_chunks[:n_results]
    context = "\n\n".join(chunk["document"] for chunk in selected_chunks)
    sources = []
    for chunk in selected_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    return {"context": context, "sources": sources}
