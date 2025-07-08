from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Embedding model settings
    EMBEDDING_MODEL: str = "jinaai/jina-embeddings-v3"
    EMBEDDING_DIMENSION: int = 1024
    MAX_SEQ_LENGTH: int = 8192
    
    # LLM settings
    LLM_MODEL: str = "llama3.1:8b"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    
    # Chunking settings
    CHUNK_SIZE: int = 600  # tokens
    CHUNK_OVERLAP: int = 100  # tokens
    
    # Vector store settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "munirag_docs"
    
    # Search settings
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    USE_HYBRID_SEARCH: bool = True
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Legacy settings (for backward compatibility)
    OLLAMA_HOST: str = "http://ollama:11434"
    CHROMA_DIR: str = "/app/chroma_data"
    MAX_CHUNK_TOKENS: int = 500
    TOP_K: int = 4
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES_CRAWL: int = 20
    REQUEST_TIMEOUT: int = 30
    LOG_LEVEL: str = "INFO"
    RESET_DATA_ON_STARTUP: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
