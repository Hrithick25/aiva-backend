import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Always load the backend .env from the project root, regardless of where scripts are run.
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Support multiple API keys for automatic rotation when one is rate-limited
_all_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
GEMINI_API_KEYS = [k for k in _all_keys if k]  # filter out None/empty


# ── Source documents ─────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "rag_faiss", "data")

KNOWLEDGE_FILES = {
    "Achievements": os.path.join(BASE_DIR, "Achievements.txt"),
    "Admissions": os.path.join(BASE_DIR, "Admissions.txt"),
    "Cutoffs": os.path.join(BASE_DIR, "Cutoffs.txt"),
    "Dataset": os.path.join(BASE_DIR, "Dataset.txt"),
    "Departments": os.path.join(BASE_DIR, "Departments.txt"),
    "Fees_Structure": os.path.join(BASE_DIR, "Fees_Structure.txt"),
    "Higher_Education": os.path.join(BASE_DIR, "Higher_Education.txt"),
    "Hostel": os.path.join(BASE_DIR, "Hostel.txt"),
    "Overview": os.path.join(BASE_DIR, "Overview.txt"),
    "Placements": os.path.join(BASE_DIR, "Placements.txt"),
    "Bus_Details": os.path.join(BASE_DIR, "Bus_Details.txt"),
}

# ── Persistence paths ────────────────────────────────────────────────
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "rag_faiss", "embeddings")
PICKLES_DIR = os.path.join(BASE_DIR, "rag_faiss", "pickles")

FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
INDEX_MAP_PATH = os.path.join(EMBEDDINGS_DIR, "index_map.pkl")

# ── Chunking ─────────────────────────────────────────────────────────
# OPTIMIZED: Smaller chunks = more precise retrieval, more overlap = no info loss
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── FAISS HNSW tuning ────────────────────────────────────────────────
HNSW_M = 40
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# ── Retrieval defaults ───────────────────────────────────────────────
# OPTIMIZED: Fewer results = less noise in LLM context, faster generation
TOP_K = 5

# ── Gemini embedding model ───────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"
