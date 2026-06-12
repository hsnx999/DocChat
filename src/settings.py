# ── Document processing ─────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".txt": "Text",
    ".docx": "Word",
}

# ── Vector store ────────────────────────────────────────────
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── RAG chain ───────────────────────────────────────────────
LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
]
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.0
RETRIEVE_K = 6
MAX_HISTORY_TURNS = 6
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 4

# ── Hybrid search ─────────────────────────────────────────────
BM25_K = 10
HYBRID_ALPHA = 0.5

# ── Session isolation ─────────────────────────────────────────
SESSION_CLEANUP_AGE_HOURS = 24
