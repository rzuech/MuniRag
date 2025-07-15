import torch
torch.set_float32_matmul_precision("high")  # Speed optimization for RTX 4090
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Dict, Any
import numpy as np
import os
import warnings
import logging
from src.config import settings

# Suppress specific warnings that clutter logs
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Universal embedding model supporting multiple architectures:
    - BGE (BAAI/bge-large-en-v1.5) - Best GPU performance
    - GTE (thenlper/gte-large) - Lightweight alternative
    - Instructor (hkunlp/instructor-xl) - Task-aware embeddings
    - E5 (intfloat/e5-large-v2) - Requires query/passage prefixes
    - Jina (jinaai/jina-embeddings-v3) - Long context, CPU-optimized
    """
    
    # Model specifications
    MODEL_SPECS = {
        "BAAI/bge-large-en-v1.5": {
            "dimension": 1024,
            "max_tokens": 512,
            "gpu_optimized": True,
            "requires_prefix": False,
            "trust_remote_code": False
        },
        "thenlper/gte-large": {
            "dimension": 768,
            "max_tokens": 512,
            "gpu_optimized": True,
            "requires_prefix": False,
            "trust_remote_code": False
        },
        "hkunlp/instructor-xl": {
            "dimension": 768,
            "max_tokens": 512,
            "gpu_optimized": True,
            "requires_prefix": False,  # Uses instructions instead
            "trust_remote_code": False,
            "requires_instructions": True
        },
        "intfloat/e5-large-v2": {
            "dimension": 1024,
            "max_tokens": 512,
            "gpu_optimized": True,
            "requires_prefix": True,  # "query: " or "passage: "
            "trust_remote_code": False
        },
        "jinaai/jina-embeddings-v3": {
            "dimension": 1024,
            "max_tokens": 8192,
            "gpu_optimized": False,  # Known GPU issues
            "requires_prefix": False,
            "trust_remote_code": True,
            "requires_task": True  # task parameter for encode
        }
    }
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model specifications
        self.model_spec = self.MODEL_SPECS.get(self.model_name, {})
        
        # Warn about GPU performance if using Jina on GPU
        if "jina" in self.model_name.lower() and self.device == "cuda":
            logger.warning("⚠️  Jina models have known GPU performance issues")
            logger.warning("   Consider using BAAI/bge-large-en-v1.5 for 30x faster GPU performance")
            logger.warning("   Or use CPU mode with multiple workers for better Jina performance")
        
        # Detect hardware capabilities for optimization
        self._detect_hardware_capabilities()
        
        # OVERRIDE: Force optimal settings for RTX 4090
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "NVIDIA GeForce RTX 4090" in gpu_name:
                self.batch_size = 1024  # Optimal from testing
                logger.info(f"RTX 4090 detected - forcing batch size to 1024")
        
        # Enable GPU optimizations BEFORE model initialization
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        
        # Initialize the model based on type
        logger.info(f"Loading embedding model: {self.model_name}")
        self._initialize_model()
        
        # Ensure model is on correct device
        self._verify_device_placement()
        
        # Set model to eval mode for inference
        if self.device == "cuda" and hasattr(self.model, 'eval'):
            self.model.eval()
    
    def _initialize_model(self):
        """Initialize model with architecture-specific settings"""
        model_kwargs = {
            "device": self.device,
            "cache_folder": "/app/models/huggingface"
        }
        
        # Add trust_remote_code for models that need it
        if self.model_spec.get("trust_remote_code", False):
            model_kwargs["trust_remote_code"] = True
        
        # Special handling for Instructor model
        if "instructor" in self.model_name.lower():
            # Instructor models need special initialization
            try:
                from InstructorEmbedding import INSTRUCTOR
                self.model = INSTRUCTOR(self.model_name, device=self.device)
                self.is_instructor = True
                logger.info("Initialized Instructor model with special handling")
                return
            except ImportError:
                logger.warning("InstructorEmbedding not available, falling back to SentenceTransformers")
                self.is_instructor = False
        else:
            self.is_instructor = False
        
        # Standard SentenceTransformers initialization
        self.model = SentenceTransformer(self.model_name, **model_kwargs)
    
    def _verify_device_placement(self):
        """Ensure model is actually on the correct device"""
        if self.device == "cuda" and torch.cuda.is_available():
            if not self.is_instructor:
                self.model = self.model.to("cuda")
            logger.info(f"Model moved to CUDA device")
            
            # Verify placement
            try:
                if hasattr(self.model, 'device'):
                    logger.info(f"Model device: {self.model.device}")
                elif hasattr(self.model, '_modules'):
                    first_module = next(iter(self.model._modules.values()))
                    if hasattr(first_module, 'parameters'):
                        device = next(first_module.parameters()).device
                        logger.info(f"Verified model is on: {device}")
            except:
                pass
    
    def _detect_hardware_capabilities(self):
        """Detect hardware and set optimal batch sizes"""
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)
            
            # Model-specific batch size optimization
            if "bge" in self.model_name.lower() or "gte" in self.model_name.lower():
                # BGE/GTE models are highly optimized for GPU
                if gpu_memory >= 16:  # RTX 4090, A100
                    self.batch_size = 512  # Optimal for RTX 4090
                elif gpu_memory >= 8:  # RTX 3070, T4
                    self.batch_size = 128
                else:
                    self.batch_size = 64
            elif "instructor" in self.model_name.lower():
                # Instructor-XL needs more memory
                if gpu_memory >= 16:
                    self.batch_size = 64
                elif gpu_memory >= 8:
                    self.batch_size = 32
                else:
                    self.batch_size = 16
            elif "e5" in self.model_name.lower():
                # E5 models are memory efficient
                if gpu_memory >= 16:
                    self.batch_size = 192
                elif gpu_memory >= 8:
                    self.batch_size = 96
                else:
                    self.batch_size = 48
            elif "jina" in self.model_name.lower():
                # Jina: small batches due to CPU-bound operations
                self.batch_size = 8 if gpu_memory >= 8 else 4
            else:
                # Conservative default
                self.batch_size = 32 if gpu_memory >= 8 else 16
                
            logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB) | Batch size: {self.batch_size}")
        else:
            # CPU optimization
            cpu_count = os.cpu_count() or 4
            if "jina" in self.model_name.lower() and cpu_count >= 8:
                # Jina can benefit from parallel CPU processing
                self.batch_size = min(16, cpu_count // 2)
                logger.info(f"🖥️  CPU mode: {cpu_count} cores | Batch size: {self.batch_size}")
                logger.info("   Consider parallel CPU processing for Jina models")
            else:
                self.batch_size = 4
                logger.info(f"🖥️  CPU mode: Batch size {self.batch_size}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents for storage with model-specific handling"""
        if not texts:
            return []
        
        # Track performance
        import time
        start_time = time.time()
        num_texts = len(texts)
        
        # Log GPU usage before embedding
        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated() / 1e9
            logger.info(f"📊 Embedding {num_texts} texts | GPU memory: {gpu_mem_before:.2f}GB")
        
        # Prepare texts based on model requirements
        prepared_texts = self._prepare_texts_for_embedding(texts, is_query=False)
        
        # Encode with model-specific parameters
        try:
            if self.is_instructor:
                # Instructor model needs instructions
                instructions = ["Represent the municipal document for retrieval:"] * len(prepared_texts)
                embeddings = self.model.encode(
                    sentences=[(inst, text) for inst, text in zip(instructions, prepared_texts)],
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )
            elif "jina" in self.model_name.lower():
                # Jina with task parameter (keep numpy for CPU operations)
                embeddings = self.model.encode(
                    prepared_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,  # Jina is CPU-bound anyway
                    task='retrieval.passage'
                )
            else:
                # Standard encoding - keep on GPU for speed
                # Use optimal settings from gpu_spike_test
                if self.device == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True):
                        embeddings = self.model.encode(
                            prepared_texts,
                            batch_size=self.batch_size,
                            show_progress_bar=False,  # CRITICAL: Progress bar causes 33x slowdown!
                            convert_to_tensor='cuda',
                            device='cuda',
                            normalize_embeddings=False  # Skip normalization for speed
                        )
                else:
                    embeddings = self.model.encode(
                        prepared_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
            
            # Performance metrics
            elapsed = time.time() - start_time
            texts_per_sec = num_texts / elapsed if elapsed > 0 else 0
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                gpu_mem_after = torch.cuda.memory_allocated() / 1e9
                gpu_used = gpu_mem_after - gpu_mem_before
                logger.info(f"✅ Embedded {num_texts} texts in {elapsed:.1f}s ({texts_per_sec:.0f} texts/sec)")
                logger.info(f"   GPU memory used: {gpu_used:.2f}GB | Total: {gpu_mem_after:.2f}GB")
            else:
                logger.info(f"✅ Embedded {num_texts} texts in {elapsed:.1f}s ({texts_per_sec:.0f} texts/sec)")
                
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Fallback to smaller batch size
            logger.warning(f"Memory error with batch size {self.batch_size}, reducing to {self.batch_size // 2}")
            self.batch_size = max(2, self.batch_size // 2)
            return self.embed_documents(texts)  # Retry with smaller batch
        
        # Convert to list format
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        elif torch.is_tensor(embeddings):
            # Convert GPU tensor to list efficiently
            return embeddings.cpu().numpy().tolist()
        else:
            return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query for search with model-specific handling"""
        # Prepare query based on model requirements
        prepared_query = self._prepare_texts_for_embedding([query], is_query=True)[0]
        
        # Encode with model-specific parameters
        if self.is_instructor:
            # Instructor model needs instruction
            embedding = self.model.encode(
                sentences=[("Represent the question for retrieving relevant documents:", prepared_query)]
            )[0]
        elif "jina" in self.model_name.lower():
            # Jina with task parameter
            embedding = self.model.encode(
                prepared_query,
                convert_to_numpy=True,
                task='retrieval.query'
            )
        else:
            # Standard encoding
            embedding = self.model.encode(
                prepared_query,
                convert_to_numpy=True
            )
        
        # Convert to list
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return embedding
    
    def _prepare_texts_for_embedding(self, texts: List[str], is_query: bool) -> List[str]:
        """Prepare texts with model-specific prefixes or formatting"""
        if self.model_spec.get("requires_prefix") and "e5" in self.model_name.lower():
            # E5 models need "query: " or "passage: " prefixes
            prefix = "query: " if is_query else "passage: "
            return [prefix + text for text in texts]
        
        # Truncate texts if needed (except Jina which handles long context)
        if "jina" not in self.model_name.lower():
            max_tokens = self.model_spec.get("max_tokens", 512)
            # Simple truncation (could be improved with proper tokenization)
            return [text[:max_tokens * 4] for text in texts]  # Rough estimate: 4 chars per token
        
        return texts
    
    def get_dimension(self) -> int:
        """Get embedding dimension for this model"""
        return self.model_spec.get("dimension", 768)
    
    def get_max_tokens(self) -> int:
        """Get maximum token length for this model"""
        return self.model_spec.get("max_tokens", 512)

# Legacy compatibility functions
def embed(texts):
    """Legacy function for backward compatibility"""
    embedder = EmbeddingModel()
    if isinstance(texts, str):
        return [embedder.embed_query(texts)]
    elif isinstance(texts, list) and len(texts) == 1:
        return [embedder.embed_query(texts[0])]
    else:
        return embedder.embed_documents(texts)
