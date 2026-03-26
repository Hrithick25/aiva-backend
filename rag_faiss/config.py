import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Always load the backend .env from the project root, regardless of where scripts are run.
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ── Source documents ──────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "rag_faiss", "data")

KNOWLEDGE_FILES = {
    "Achievements": os.path.join(BASE_DIR, "Achievements.txt"),
    "Admissions":   os.path.join(BASE_DIR, "Admissions.txt"),
    "Cutoffs":      os.path.join(BASE_DIR, "Cutoffs.txt"),
    "Dataset":      os.path.join(BASE_DIR, "Dataset.txt"),
    "Departments":  os.path.join(BASE_DIR, "Departments.txt"),
    "Fees_Structure":   os.path.join(BASE_DIR, "Fees_Structure.txt"),
    "Higher_Education": os.path.join(BASE_DIR, "Higher_Education.txt"),
    "Hostel":       os.path.join(BASE_DIR, "Hostel.txt"),
    "Overview":     os.path.join(BASE_DIR, "Overview.txt"),
    "Placements":   os.path.join(BASE_DIR, "Placements.txt"),
    "Bus_Details":  os.path.join(BASE_DIR, "Bus_Details.txt"),
}

# ── Persistence paths ─────────────────────────────────────────────────
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "rag_faiss", "embeddings")
PICKLES_DIR    = os.path.join(BASE_DIR, "rag_faiss", "pickles")

FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
INDEX_MAP_PATH   = os.path.join(EMBEDDINGS_DIR, "index_map.pkl")

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ── FAISS tuning ──────────────────────────────────────────────────────
HNSW_M              = 40
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 64

# ── Retrieval defaults ────────────────────────────────────────────────
TOP_K = 5
