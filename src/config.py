"""
config.py — Central configuration for Fasih-Docs.
Change settings here. Everything else reads from here.
"""

# ── Paths ──────────────────────────────────────────────────────
DOCS_DIR        = "docs"          # Put your PDFs here
CHROMA_DIR      = "chroma_db"     # Vector DB stored here (auto-created)
LOG_FILE        = "logs/queries.log"

# ── Embedding model (free, runs locally via HuggingFace) ───────
# This model converts text → numbers so similarity search works.
# multilingual-e5 supports Arabic + English — perfect for this project.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ── LLM (free, runs locally via Ollama) ────────────────────────
# Ollama runs Mistral 7B on your machine. No internet needed after download.
OLLAMA_MODEL    = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── RAG settings ───────────────────────────────────────────────
CHUNK_SIZE      = 600    # How many characters per chunk
CHUNK_OVERLAP   = 100    # Overlap between chunks so context isn't lost
TOP_K_RESULTS   = 4      # How many chunks to retrieve per question

# ── UI ─────────────────────────────────────────────────────────
APP_TITLE       = "Fasih-Docs"
APP_DESCRIPTION = "AI assistant for engineering documentation — ask anything about your technical PDFs."
