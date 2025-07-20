"""
Embedding Model Registry
Central registry for all supported embedding models and their configurations
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model"""
    name: str
    dimension: int
    max_tokens: int
    description: str
    recommended_batch_size: Optional[int] = None
    requires_gpu: bool = False
    memory_usage_mb: Optional[int] = None
    
# Central registry of all supported embedding models
EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
    # BGE Models - Best GPU performance
    "BAAI/bge-large-en-v1.5": EmbeddingModelConfig(
        name="BAAI/bge-large-en-v1.5",
        dimension=1024,
        max_tokens=512,
        description="Best GPU performance, 32x faster than Jina",
        recommended_batch_size=1024,
        requires_gpu=False,
        memory_usage_mb=1300
    ),
    "BAAI/bge-base-en-v1.5": EmbeddingModelConfig(
        name="BAAI/bge-base-en-v1.5",
        dimension=768,
        max_tokens=512,
        description="Smaller BGE model, good balance",
        recommended_batch_size=1024,
        requires_gpu=False,
        memory_usage_mb=430
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelConfig(
        name="BAAI/bge-small-en-v1.5",
        dimension=384,
        max_tokens=512,
        description="Lightweight BGE model",
        recommended_batch_size=2048,
        requires_gpu=False,
        memory_usage_mb=130
    ),
    
    # GTE Models - Google models
    "thenlper/gte-large": EmbeddingModelConfig(
        name="thenlper/gte-large",
        dimension=768,
        max_tokens=512,
        description="Google's text embedding model",
        recommended_batch_size=512,
        requires_gpu=False,
        memory_usage_mb=1300
    ),
    "thenlper/gte-base": EmbeddingModelConfig(
        name="thenlper/gte-base",
        dimension=768,
        max_tokens=512,
        description="Base GTE model",
        recommended_batch_size=1024,
        requires_gpu=False,
        memory_usage_mb=430
    ),
    "thenlper/gte-small": EmbeddingModelConfig(
        name="thenlper/gte-small",
        dimension=384,
        max_tokens=512,
        description="Small GTE model",
        recommended_batch_size=2048,
        requires_gpu=False,
        memory_usage_mb=130
    ),
    
    # E5 Models - Microsoft models
    "intfloat/e5-large-v2": EmbeddingModelConfig(
        name="intfloat/e5-large-v2",
        dimension=1024,
        max_tokens=512,
        description="Microsoft E5, requires query/passage prefixes",
        recommended_batch_size=512,
        requires_gpu=False,
        memory_usage_mb=1300
    ),
    "intfloat/e5-base-v2": EmbeddingModelConfig(
        name="intfloat/e5-base-v2",
        dimension=768,
        max_tokens=512,
        description="Base E5 model",
        recommended_batch_size=1024,
        requires_gpu=False,
        memory_usage_mb=430
    ),
    "intfloat/e5-small-v2": EmbeddingModelConfig(
        name="intfloat/e5-small-v2",
        dimension=384,
        max_tokens=512,
        description="Small E5 model",
        recommended_batch_size=2048,
        requires_gpu=False,
        memory_usage_mb=130
    ),
    
    # Instructor Models - Task-aware embeddings
    "hkunlp/instructor-xl": EmbeddingModelConfig(
        name="hkunlp/instructor-xl",
        dimension=768,
        max_tokens=512,
        description="Task-aware embeddings, needs 16GB+ GPU",
        recommended_batch_size=256,
        requires_gpu=True,
        memory_usage_mb=5000
    ),
    "hkunlp/instructor-large": EmbeddingModelConfig(
        name="hkunlp/instructor-large",
        dimension=768,
        max_tokens=512,
        description="Large instructor model",
        recommended_batch_size=512,
        requires_gpu=False,
        memory_usage_mb=1300
    ),
    
    # Jina Models - Long context
    "jinaai/jina-embeddings-v3": EmbeddingModelConfig(
        name="jinaai/jina-embeddings-v3",
        dimension=1024,
        max_tokens=8192,
        description="Long context (8K), CPU-optimized, GPU issues",
        recommended_batch_size=32,
        requires_gpu=False,
        memory_usage_mb=2000
    ),
    "jinaai/jina-embeddings-v2-base-en": EmbeddingModelConfig(
        name="jinaai/jina-embeddings-v2-base-en",
        dimension=768,
        max_tokens=8192,
        description="Jina v2 base model",
        recommended_batch_size=32,
        requires_gpu=False,
        memory_usage_mb=1300
    ),
    
    # Sentence Transformers models
    "sentence-transformers/all-mpnet-base-v2": EmbeddingModelConfig(
        name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        max_tokens=384,
        description="General purpose sentence embeddings",
        recommended_batch_size=512,
        requires_gpu=False,
        memory_usage_mb=430
    ),
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_tokens=256,
        description="Fast, lightweight model",
        recommended_batch_size=2048,
        requires_gpu=False,
        memory_usage_mb=80
    ),
    
    # OpenAI Ada (for future support)
    "text-embedding-ada-002": EmbeddingModelConfig(
        name="text-embedding-ada-002",
        dimension=1536,
        max_tokens=8191,
        description="OpenAI's embedding model (requires API key)",
        recommended_batch_size=100,
        requires_gpu=False,
        memory_usage_mb=0  # API-based
    ),
    
    # Cohere Embed (for future support)
    "embed-english-v3.0": EmbeddingModelConfig(
        name="embed-english-v3.0",
        dimension=1024,
        max_tokens=512,
        description="Cohere's embedding model (requires API key)",
        recommended_batch_size=96,
        requires_gpu=False,
        memory_usage_mb=0  # API-based
    ),
}

def get_model_config(model_name: str) -> Optional[EmbeddingModelConfig]:
    """Get configuration for a specific model"""
    return EMBEDDING_MODELS.get(model_name)

def list_available_models() -> List[str]:
    """List all available model names"""
    return list(EMBEDDING_MODELS.keys())

def get_models_by_dimension(dimension: int) -> List[str]:
    """Get all models with a specific dimension"""
    return [
        name for name, config in EMBEDDING_MODELS.items()
        if config.dimension == dimension
    ]

def get_lightweight_models(max_memory_mb: int = 500) -> List[str]:
    """Get models that use less than specified memory"""
    return [
        name for name, config in EMBEDDING_MODELS.items()
        if config.memory_usage_mb and config.memory_usage_mb <= max_memory_mb
    ]

def get_gpu_optimized_models() -> List[str]:
    """Get models optimized for GPU performance"""
    # BGE and GTE models are best for GPU
    return [
        name for name in EMBEDDING_MODELS.keys()
        if name.startswith(("BAAI/bge", "thenlper/gte"))
    ]

def add_custom_model(config: EmbeddingModelConfig):
    """Add a custom model to the registry"""
    EMBEDDING_MODELS[config.name] = config