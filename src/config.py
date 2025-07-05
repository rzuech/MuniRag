import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "phi3:mini")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_data")
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", 500))
TOP_K = int(os.getenv("TOP_K", 4))
