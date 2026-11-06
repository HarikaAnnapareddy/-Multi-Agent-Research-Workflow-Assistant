import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_REVIEW_LOOPS = int(os.getenv("MAX_REVIEW_LOOPS", "2"))
