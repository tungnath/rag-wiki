"""Configuration constants for the RAG engine."""

import os

# ─── Paths ───────────────────────────────────────────────────────────
PDF_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
BM25_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bm25_cache.pkl")

# ─── Models ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"   # 1024 dims, strong semantic model
LLM_MODEL = "llama3.2:3b"           # Fastest CPU option

# ─── Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 500            # Chars — with ~60 char prefix, stays within mxbai 512-token window
CHUNK_OVERLAP = 100         # 20% overlap preserves sentence boundaries
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ─── Retrieval ───────────────────────────────────────────────────────
DENSE_K = 15                # Candidates from dense (semantic) search
BM25_K = 15                 # Candidates from BM25 (lexical) search
FINAL_K = 6                 # Final chunks sent to LLM after fusion
RRF_K = 60                  # Reciprocal Rank Fusion smoothing constant

# ─── Collection ──────────────────────────────────────────────────────
CHROMA_COLLECTION = "rag_wiki_v2"
