from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Embedding model settings
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"  # Default to BGE for best GPU performance
    
    # Available models with their characteristics
    # BGE: Best GPU performance (30x faster than Jina), 1024 dims, 512 tokens
    # GTE: Lightweight alternative, 768 dims, 512 tokens  
    # Instructor-XL: Task-aware embeddings, 768 dims, requires 16GB+ GPU
    # E5: Requires query/passage prefixes, 1024 dims, good performance
    # Jina: Long context (8192), CPU-optimized, GPU performance issues
    AVAILABLE_MODELS: list = [
        "BAAI/bge-large-en-v1.5",
        "thenlper/gte-large", 
        "hkunlp/instructor-xl",
        "intfloat/e5-large-v2",
        "jinaai/jina-embeddings-v3"
    ]
    
    # Model-specific settings override
    MODEL_OVERRIDES: dict = {}
    
    # General embedding settings
    EMBEDDING_MODE: str = "auto"  # auto, gpu, cpu, cpu_parallel
    FORCE_CPU_EMBEDDINGS: bool = False  # Force CPU even if GPU available
    EMBEDDING_BATCH_SIZE: Optional[int] = None  # Auto-detect based on model and hardware
    EMBEDDING_WORKERS: int = 4  # Workers for CPU parallel mode (mainly for Jina)
    
    # LLM settings
    LLM_MODEL: str = "llama3.1:8b"
    LLM_TEMPERATURE: float = 0.1  # Lower for more factual responses
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.9  # Nucleus sampling
    LLM_PROVIDER: str = "ollama"  # Future: openai, anthropic, etc.
    
    # System message for RAG-focused responses
    SYSTEM_MESSAGE: str = """You are MuniRAG, a helpful AI assistant for municipal government information. 

Your primary purpose is to answer questions using ONLY the information provided in the context. Follow these guidelines:

1. Base your answers strictly on the provided context/documents
2. If the context doesn't contain relevant information, say "I don't have information about that in the provided documents"
3. Never use information from your training data - only use the context provided
4. Be precise and factual in your responses
5. If you're uncertain, express that uncertainty rather than guessing
6. Cite specific sections or documents when possible
7. Keep answers concise and relevant to municipal operations

Remember: It's better to say you don't know than to provide incorrect information."""
    
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
    
    # GPU Resource Management
    GPU_MEMORY_THRESHOLD: float = 0.7  # Switch to CPU if GPU memory > 70%
    GPU_UTILIZATION_THRESHOLD: float = 0.8  # Switch to CPU if GPU util > 80%
    PRIORITIZE_LLM_INFERENCE: bool = True  # Give GPU priority to user queries
    MAX_CONCURRENT_EMBEDDINGS: int = 1  # Limit concurrent embedding jobs
    
    # Performance tuning
    PDF_WORKERS: Optional[int] = None  # Auto-detect if None
    USE_OCR: bool = True  # Enable OCR for scanned PDFs
    SEMANTIC_CHUNKING: bool = False  # Experimental - not yet implemented
    
    # Model performance benchmarks (texts/second on RTX 4090)
    # BGE: ~3,743 texts/sec (GPU)
    # GTE: ~3,200 texts/sec (GPU)
    # E5: ~2,800 texts/sec (GPU)
    # Instructor-XL: ~1,500 texts/sec (GPU)
    # Jina: ~117 texts/sec (CPU-bound even on GPU)
    
    # Municipality integration
    ALLOWED_ORIGINS: str = "*"  # CORS origins (comma-separated)
    REQUIRE_API_KEY: bool = False  # API authentication
    API_RATE_LIMIT: int = 100  # Requests per minute
    
    # Service endpoints
    OLLAMA_HOST: str = "http://ollama:11434"
    
    # Document processing settings
    MAX_CHUNK_TOKENS: int = 500
    TOP_K: int = 4  # Number of chunks to retrieve
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES_CRAWL: int = 20
    REQUEST_TIMEOUT: int = 30
    
    # Logging and debugging
    LOG_LEVEL: str = "INFO"
    SUPPRESS_WARNINGS: bool = True  # Suppress transformer warnings
    DEBUG_GPU_USAGE: bool = False  # Enable detailed GPU logging
    
    # Data management
    RESET_DATA_ON_STARTUP: bool = True  # Set to False for production
    RESET_MUNIRAG_ONLY: bool = True    # Only reset munirag_* collections, not others
    QDRANT_DIMENSION: Optional[int] = None  # Auto-detect from model
    
    class Config:
        env_file = ".env"
    
    def get_embedding_dimension(self) -> int:
        """Get dimension for current embedding model"""
        if self.QDRANT_DIMENSION:
            return self.QDRANT_DIMENSION
            
        # Model-specific dimensions
        model_dimensions = {
            "BAAI/bge-large-en-v1.5": 1024,
            "thenlper/gte-large": 768,
            "hkunlp/instructor-xl": 768,
            "intfloat/e5-large-v2": 1024,
            "jinaai/jina-embeddings-v3": 1024
        }
        
        return model_dimensions.get(self.EMBEDDING_MODEL, 768)
    
    def get_max_context_length(self) -> int:
        """Get max context length for current embedding model"""
        # Model-specific context lengths
        model_contexts = {
            "BAAI/bge-large-en-v1.5": 512,
            "thenlper/gte-large": 512,
            "hkunlp/instructor-xl": 512,
            "intfloat/e5-large-v2": 512,
            "jinaai/jina-embeddings-v3": 8192
        }
        
        return model_contexts.get(self.EMBEDDING_MODEL, 512)

settings = Settings()
