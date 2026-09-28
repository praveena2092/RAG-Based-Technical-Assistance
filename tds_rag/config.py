"""Central configuration for every stage of the RAG pipeline.

All secrets and tunables come from environment variables (see `.env.example`).
Secrets are read lazily, so e.g. the scrapers do not need Qdrant credentials.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── Data layout ────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
COURSE_REPO_DIR = DATA_DIR / "raw" / "course_repo"          # cloned course repo
COURSE_ASSETS_DIR = DATA_DIR / "raw" / "course_assets"      # images / py / html copies
COURSE_SUMMARY_FILE = DATA_DIR / "raw" / "summary_of_files.txt"
DISCOURSE_DIR = DATA_DIR / "raw" / "discourse"
DISCOURSE_IMAGES_DIR = DISCOURSE_DIR / "images"
THREADS_FILE = DISCOURSE_DIR / "tds_threads.json"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.json"
VECTORS_FILE = DATA_DIR / "embeddings" / "vectors.json"

# ── Sources ────────────────────────────────────────────────────────────────
COURSE_REPO_URL = "https://github.com/sanand0/tools-in-data-science-public.git"
COURSE_SITE_URL = "https://tds.s-anand.net/#/"
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
DISCOURSE_CATEGORY_URL = f"{DISCOURSE_BASE_URL}/c/courses/tds-kb/34"

# ── Models / services ──────────────────────────────────────────────────────
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536  # text-embedding-3-small; change together with the model
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tds_kb")

# ── Tunables ───────────────────────────────────────────────────────────────
CHUNK_TOKEN_LIMIT = int(os.getenv("CHUNK_TOKEN_LIMIT", "4096"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "10"))
TOP_K = int(os.getenv("TOP_K", "7"))


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def get_openai_key() -> str:
    """OPENAI_API_KEY (AIPipe or OpenAI). AIPIPE_TOKEN is accepted as a legacy alias."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("AIPIPE_TOKEN")
    if not key:
        raise EnvironmentError("Missing OPENAI_API_KEY (or legacy AIPIPE_TOKEN)")
    return key
