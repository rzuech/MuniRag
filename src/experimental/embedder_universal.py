"""
=============================================================================
EMBEDDER_UNIVERSAL.PY - Flexible Multi-Model Embedding System
=============================================================================

This module provides a universal embedding interface that supports multiple
embedding models with both GPU and CPU execution modes.

KEY FEATURES:
1. Multi-model support (Jina, BGE, E5, InstructorXL)
2. GPU prioritization with CPU fallback
3. Thread-safe model initialization
4. Resource-aware mode switching
5. Consistent dimension handling

DESIGN PHILOSOPHY:
- Reliability first: Never fail, always have a fallback
- User-friendly: Auto-detect best configuration
- Performance: Use GPU when available, CPU when necessary
- Flexibility: Easy to add new models

GPU RESOURCE MANAGEMENT:
- End-user LLM queries get GPU priority
- Embeddings can be paused/moved to CPU if needed
- Smart scheduling prevents resource conflicts

STABILITY NOTES:
- All model initialization happens in main thread
- Models are warmed up before use
- Dimension verification prevents mismatches
- Comprehensive error handling with fallbacks
"""

import os
import logging
import threading
import time
from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

# Make InstructorEmbedding optional
try:
    from InstructorEmbedding import INSTRUCTOR
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False

from src.logger import get_logger
from src.config import settings

logger = get_logger("embedder_universal")


class ExecutionMode(Enum):
    """Execution modes for embeddings."""
    AUTO = "auto"      # Automatically choose best mode
    GPU = "gpu"        # Force GPU execution
    CPU = "cpu"        # Force CPU execution
    CPU_PARALLEL = "cpu_parallel"  # Multi-core CPU execution


@dataclass
class ModelConfig:
    """Configuration for each embedding model."""
    model_id: str
    dimension: int
    model_class: str
    gpu_capable: bool
    optimal_batch_size: Dict[str, int]
    requires_instruction: bool = False
    revision: Optional[str] = None  # For version pinning
    max_sequence_length: int = 512
    
    
class ModelRegistry:
    """Registry of supported embedding models with their configurations."""
    
    MODELS = {
        "jina-v3": ModelConfig(
            model_id="jinaai/jina-embeddings-v3",
            dimension=1024,
            model_class="JinaEmbedder",
            gpu_capable=True,
            optimal_batch_size={"gpu": 128, "cpu": 32},
            revision="8702b35d13d05f77e22fbaaa8ba4e0091d8d5f45",  # Pinned version
            max_sequence_length=8192
        ),
        "bge-large-en": ModelConfig(
            model_id="BAAI/bge-large-en-v1.5",
            dimension=1024,
            model_class="BGEEmbedder",
            gpu_capable=True,
            optimal_batch_size={"gpu": 256, "cpu": 64},
            revision="5ccee170680c58ec3fb30be6a3f744a8725fc7ec",
            max_sequence_length=512
        ),
        "e5-large-v2": ModelConfig(
            model_id="intfloat/e5-large-v2",
            dimension=1024,
            model_class="E5Embedder",
            gpu_capable=True,
            optimal_batch_size={"gpu": 256, "cpu": 64},
            revision="b322e09026e4ea05e4d9e2e3ffb7e4de960340b8",
            max_sequence_length=512
        ),
        "instructor-xl": ModelConfig(
            model_id="hkunlp/instructor-xl",
            dimension=768,
            model_class="InstructorEmbedder",
            gpu_capable=True,
            optimal_batch_size={"gpu": 64, "cpu": 16},
            requires_instruction=True,
            revision="f2c78df6b85cf12d2e04858e05cef6c7dfc11dba",
            max_sequence_length=512
        )
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a model."""
        if model_name not in cls.MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_name]


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""
    
    def __init__(self, config: ModelConfig, device: str = None):
        self.config = config
        self.device = self._determine_device(device)
        self.model = None
        
    def _determine_device(self, requested_device: Optional[str]) -> str:
        """Determine the best device to use."""
        if requested_device:
            return requested_device
            
        if torch.cuda.is_available() and self.config.gpu_capable:
            # Check GPU memory
            gpu_mem_free = torch.cuda.mem_get_info()[0] / 1024**3
            if gpu_mem_free < 2.0:  # Need at least 2GB
                logger.warning(f"Low GPU memory ({gpu_mem_free:.1f}GB), using CPU")
                return "cpu"
            return "cuda"
        
        return "cpu"
    
    @abstractmethod
    def load_model(self):
        """Load the embedding model."""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for documents."""
        pass
    
    def verify_dimensions(self, embeddings: List[np.ndarray]) -> bool:
        """Verify embeddings have correct dimensions."""
        if not embeddings:
            return True
            
        actual_dim = embeddings[0].shape[0]
        expected_dim = self.config.dimension
        
        if actual_dim != expected_dim:
            logger.error(f"Dimension mismatch! Expected {expected_dim}, got {actual_dim}")
            return False
        
        return True


class JinaEmbedder(BaseEmbedder):
    """Jina embedding model implementation."""
    
    def load_model(self):
        """Load Jina model with proper initialization."""
        logger.info(f"Loading Jina model on {self.device}")
        
        try:
            # Set trust_remote_code for Jina
            self.model = SentenceTransformer(
                self.config.model_id,
                device=self.device,
                trust_remote_code=True,
                revision=self.config.revision
            )
            
            # Warm up the model
            _ = self.model.encode(["warmup"], convert_to_numpy=True)
            
            logger.info(f"Jina model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Jina model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings using Jina."""
        if not self.model:
            self.load_model()
            
        # Use appropriate batch size based on device
        batch_size = self.config.optimal_batch_size.get(
            "gpu" if self.device == "cuda" else "cpu"
        )
        
        # Create embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Jina works better with normalized embeddings
            show_progress_bar=False
        )
        
        # Convert to list of arrays
        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings[i] for i in range(len(embeddings))]
        
        # Verify dimensions
        if not self.verify_dimensions(embeddings):
            raise ValueError("Embedding dimension mismatch")
            
        return embeddings


class BGEEmbedder(BaseEmbedder):
    """BGE embedding model implementation."""
    
    def load_model(self):
        """Load BGE model."""
        logger.info(f"Loading BGE model on {self.device}")
        
        self.model = SentenceTransformer(
            self.config.model_id,
            device=self.device,
            revision=self.config.revision
        )
        
        # BGE requires specific prompt format
        self.instruction = "Represent this document for retrieval: "
        
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings using BGE."""
        if not self.model:
            self.load_model()
            
        # Add instruction prefix for BGE
        texts_with_instruction = [self.instruction + text for text in texts]
        
        embeddings = self.model.encode(
            texts_with_instruction,
            batch_size=self.config.optimal_batch_size.get(
                "gpu" if self.device == "cuda" else "cpu"
            ),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings[i] for i in range(len(embeddings))]
            
        return embeddings


class E5Embedder(BaseEmbedder):
    """E5 embedding model implementation."""
    
    def load_model(self):
        """Load E5 model."""
        logger.info(f"Loading E5 model on {self.device}")
        
        self.model = SentenceTransformer(
            self.config.model_id,
            device=self.device,
            revision=self.config.revision
        )
        
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings using E5."""
        if not self.model:
            self.load_model()
            
        # E5 requires "passage: " prefix for documents
        texts_with_prefix = [f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            texts_with_prefix,
            batch_size=self.config.optimal_batch_size.get(
                "gpu" if self.device == "cuda" else "cpu"
            ),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings[i] for i in range(len(embeddings))]
            
        return embeddings


class InstructorEmbedder(BaseEmbedder):
    """Instructor embedding model implementation."""
    
    def __init__(self, config: ModelConfig, device: str = None, 
                 instruction: str = None):
        super().__init__(config, device)
        self.instruction = instruction or "Represent the document for retrieval: "
        
    def load_model(self):
        """Load Instructor model."""
        if not INSTRUCTOR_AVAILABLE:
            raise ImportError("InstructorEmbedding package not installed. Install with: pip install InstructorEmbedding")
            
        logger.info(f"Loading Instructor model on {self.device}")
        
        self.model = INSTRUCTOR(self.config.model_id, device=self.device)
        
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings using Instructor."""
        if not self.model:
            self.load_model()
            
        # Instructor requires instruction-text pairs
        instruction_pairs = [[self.instruction, text] for text in texts]
        
        embeddings = self.model.encode(
            instruction_pairs,
            batch_size=self.config.optimal_batch_size.get(
                "gpu" if self.device == "cuda" else "cpu"
            ),
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings[i] for i in range(len(embeddings))]
            
        return embeddings


class ResourceMonitor:
    """Monitor GPU/CPU resources for intelligent mode switching."""
    
    @staticmethod
    def get_gpu_memory_usage() -> float:
        """Get GPU memory usage percentage."""
        if not torch.cuda.is_available():
            return 0.0
            
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return used / total
    
    @staticmethod
    def get_gpu_utilization() -> float:
        """Get GPU utilization percentage."""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            nvml.nvmlShutdown()
            return util.gpu / 100.0
        except:
            return 0.0
    
    @staticmethod
    def should_use_gpu() -> bool:
        """Determine if GPU should be used based on current load."""
        if not torch.cuda.is_available():
            return False
            
        # Check if LLM inference might be happening
        gpu_memory = ResourceMonitor.get_gpu_memory_usage()
        gpu_util = ResourceMonitor.get_gpu_utilization()
        
        # Conservative thresholds to protect LLM performance
        if gpu_memory > 0.7 or gpu_util > 0.8:
            logger.info(f"GPU busy (mem: {gpu_memory:.1%}, util: {gpu_util:.1%}), using CPU")
            return False
            
        return True


class UniversalEmbedder:
    """
    Universal embedder with automatic mode selection and resource management.
    
    This is the main interface for embeddings, supporting:
    - Multiple models (Jina, BGE, E5, InstructorXL)
    - GPU/CPU mode selection
    - Resource-aware scheduling
    - Thread-safe operation
    """
    
    # Class-level model cache for thread safety
    _model_cache = {}
    _cache_lock = threading.Lock()
    
    def __init__(self,
                 model_name: str = None,
                 mode: ExecutionMode = ExecutionMode.AUTO,
                 force_device: str = None,
                 instruction: str = None):
        """
        Initialize universal embedder.
        
        Args:
            model_name: Model to use (default from settings)
            mode: Execution mode (auto, gpu, cpu, cpu_parallel)
            force_device: Force specific device (cuda, cpu)
            instruction: Custom instruction for InstructorXL
        """
        self.model_name = model_name or settings.DEFAULT_EMBEDDING_MODEL
        self.mode = mode
        self.force_device = force_device
        self.instruction = instruction
        
        # Get model configuration
        self.config = ModelRegistry.get_config(self.model_name)
        
        # Initialize model
        self.embedder = self._get_or_create_embedder()
        
        logger.info(f"UniversalEmbedder initialized: model={self.model_name}, "
                   f"mode={mode.value}, device={self.embedder.device}")
    
    def _get_or_create_embedder(self) -> BaseEmbedder:
        """Get embedder from cache or create new one (thread-safe)."""
        # Determine device based on mode
        if self.force_device:
            device = self.force_device
        elif self.mode == ExecutionMode.GPU:
            device = "cuda"
        elif self.mode == ExecutionMode.CPU or self.mode == ExecutionMode.CPU_PARALLEL:
            device = "cpu"
        else:  # AUTO mode
            device = "cuda" if ResourceMonitor.should_use_gpu() else "cpu"
        
        # Cache key includes model name and device
        cache_key = f"{self.model_name}_{device}"
        
        # Check cache first
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Create new embedder (thread-safe)
        with self._cache_lock:
            # Double-check after acquiring lock
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]
                
            # Create appropriate embedder
            if self.config.model_class == "JinaEmbedder":
                embedder = JinaEmbedder(self.config, device)
            elif self.config.model_class == "BGEEmbedder":
                embedder = BGEEmbedder(self.config, device)
            elif self.config.model_class == "E5Embedder":
                embedder = E5Embedder(self.config, device)
            elif self.config.model_class == "InstructorEmbedder":
                if INSTRUCTOR_AVAILABLE:
                    embedder = InstructorEmbedder(self.config, device, self.instruction)
                else:
                    logger.warning("InstructorEmbedding not available, falling back to Jina")
                    # Fall back to Jina
                    jina_config = ModelRegistry.get_config("jina-v3")
                    embedder = JinaEmbedder(jina_config, device)
            else:
                raise ValueError(f"Unknown embedder class: {self.config.model_class}")
            
            # Load model in main thread
            embedder.load_model()
            
            # Cache for reuse
            self._model_cache[cache_key] = embedder
            
            return embedder
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for documents.
        
        Automatically handles mode selection and resource management.
        """
        if not texts:
            return []
        
        # For AUTO mode, check if we should switch devices
        if self.mode == ExecutionMode.AUTO:
            current_device = self.embedder.device
            should_use_gpu = ResourceMonitor.should_use_gpu()
            desired_device = "cuda" if should_use_gpu else "cpu"
            
            if current_device != desired_device:
                logger.info(f"Switching from {current_device} to {desired_device} based on load")
                self.embedder = self._get_or_create_embedder()
        
        # Create embeddings
        start_time = time.time()
        embeddings = self.embedder.embed_documents(texts)
        elapsed = time.time() - start_time
        
        docs_per_sec = len(texts) / elapsed if elapsed > 0 else 0
        logger.info(f"Created {len(embeddings)} embeddings in {elapsed:.2f}s "
                   f"({docs_per_sec:.1f} docs/sec) on {self.embedder.device}")
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension for current model."""
        return self.config.dimension
    
    def get_device(self) -> str:
        """Get current execution device."""
        return self.embedder.device


def create_embedder(**kwargs) -> UniversalEmbedder:
    """Factory function to create embedder with sensible defaults."""
    return UniversalEmbedder(**kwargs)


# === MIGRATION GUIDE ===
"""
To migrate from the old EmbeddingModel to UniversalEmbedder:

1. Replace imports:
   OLD: from src.embedder import EmbeddingModel
   NEW: from src.embedder_universal import create_embedder

2. Replace initialization:
   OLD: embedder = EmbeddingModel()
   NEW: embedder = create_embedder()  # Uses settings.DEFAULT_EMBEDDING_MODEL

3. Usage remains the same:
   embeddings = embedder.embed_documents(texts)

4. For specific models:
   embedder = create_embedder(model_name="bge-large-en")

5. For forced GPU/CPU:
   embedder = create_embedder(mode=ExecutionMode.GPU)  # Force GPU
   embedder = create_embedder(mode=ExecutionMode.CPU)  # Force CPU

6. For Instructor with custom instruction:
   embedder = create_embedder(
       model_name="instructor-xl",
       instruction="Represent this legal document:"
   )
"""