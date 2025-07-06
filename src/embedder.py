"""
=============================================================================
EMBEDDER.PY - Text Embedding Service for MuniRag
=============================================================================

This module provides text embedding functionality using sentence transformers.
It implements a singleton pattern to avoid reloading the model multiple times.

PURPOSE:
- Convert text into numerical vectors (embeddings) for similarity search
- Use pre-trained sentence transformer models for high-quality embeddings
- Implement singleton pattern for efficient memory usage
- Handle GPU/CPU device selection automatically

WHAT ARE EMBEDDINGS?
Embeddings are numerical representations of text that capture semantic meaning.
Similar texts have similar embeddings, which allows us to find relevant documents
by comparing their embeddings to a query embedding.

EXAMPLE:
- "municipal budget" and "city finances" would have similar embeddings
- "dog food" and "city budget" would have very different embeddings

SINGLETON PATTERN:
The EmbeddingService class uses the singleton pattern to ensure that:
1. Only one instance of the embedding model is loaded in memory
2. The model is shared across all parts of the application
3. Memory usage is optimized (models can be 1GB+ in size)

USAGE:
    from embedder import EmbeddingService
    
    # Get the singleton instance
    embedder = EmbeddingService()
    
    # Convert text to embeddings
    embeddings = embedder.encode(["Hello world", "Municipal services"])
    
    # Or use the legacy function
    embeddings = embed(["Hello world"])
"""

from sentence_transformers import SentenceTransformer
import torch
import threading
from config import EMBEDDING_MODEL
from logger import get_logger

# Get a logger specific to this module
logger = get_logger("embedder")


class EmbeddingService:
    """
    Singleton service for text embedding operations.
    
    This class ensures that only one instance of the embedding model
    is loaded in memory, which is important because:
    1. Embedding models are large (100MB to 1GB+)
    2. Loading them takes time (5-30 seconds)
    3. Multiple instances waste memory and slow down the system
    
    The singleton pattern ensures efficient resource usage across
    the entire application.
    """
    
    # Class variables for singleton implementation
    _instance = None        # The single instance of this class
    _model = None          # The loaded embedding model
    _lock = threading.Lock()  # Thread safety for multi-threaded access
    
    def __new__(cls):
        """
        Override the __new__ method to implement singleton pattern.
        
        This method is called before __init__ and controls object creation.
        It ensures that only one instance of EmbeddingService exists.
        
        Returns:
            EmbeddingService: The singleton instance
        """
        if cls._instance is None:
            # Use a lock to ensure thread safety
            # This prevents race conditions in multi-threaded environments
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    # Create the single instance
                    cls._instance = super().__new__(cls)
                    # Initialize the model
                    cls._instance._initialize_model()
        
        return cls._instance
    
    def _initialize_model(self):
        """
        Initialize the embedding model.
        
        This method:
        1. Detects if GPU is available and uses it for faster processing
        2. Loads the specified sentence transformer model
        3. Handles errors gracefully if model loading fails
        
        The model is loaded once and reused for all embedding operations.
        """
        try:
            # Detect the best available device
            # GPU is much faster than CPU for embedding operations
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Initializing embedding model '{EMBEDDING_MODEL}' on device: {device}")
            
            # Load the sentence transformer model
            # This downloads the model if it's not already cached locally
            self._model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            
            logger.info(f"Embedding model initialized successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")
    
    def encode(self, texts):
        """
        Encode a list of texts into embeddings.
        
        This method converts text into numerical vectors that can be used
        for similarity search and document retrieval.
        
        Args:
            texts (list): List of strings to encode
                         Can be single string or list of strings
        
        Returns:
            numpy.ndarray: Array of embeddings, one per input text
                          Shape: (num_texts, embedding_dimension)
        
        Example:
            embedder = EmbeddingService()
            embeddings = embedder.encode(["Hello world", "Municipal budget"])
            # Returns: array([[0.1, 0.2, ...], [0.3, 0.4, ...]])
        """
        # Handle empty input
        if not texts:
            logger.warning("Empty text list provided to encode()")
            return []
        
        # Convert single string to list for consistent handling
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Generate embeddings using the loaded model
            # normalize_embeddings=True ensures all vectors have unit length
            # This makes cosine similarity equivalent to dot product
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            
            logger.debug(f"Successfully encoded {len(texts)} texts into embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise RuntimeError(f"Failed to encode texts: {str(e)}")
    
    @property
    def device(self):
        """
        Get the device the model is running on.
        
        Returns:
            str: Device name ("cuda" or "cpu")
        """
        return self._model.device if self._model else "unknown"
    
    @property
    def model_name(self):
        """
        Get the name of the loaded model.
        
        Returns:
            str: Model name from configuration
        """
        return EMBEDDING_MODEL
    
    def get_model_info(self):
        """
        Get detailed information about the loaded model.
        
        Returns:
            dict: Model information including name, device, and dimensions
        """
        if self._model is None:
            return {"error": "Model not initialized"}
        
        try:
            # Get embedding dimensions by encoding a test string
            test_embedding = self._model.encode(["test"], normalize_embeddings=True)
            dimensions = len(test_embedding[0])
            
            return {
                "model_name": EMBEDDING_MODEL,
                "device": str(self.device),
                "dimensions": dimensions,
                "max_sequence_length": getattr(self._model, 'max_seq_length', 'unknown')
            }
        except Exception as e:
            return {"error": f"Could not get model info: {str(e)}"}


# === LEGACY FUNCTIONS ===
# These functions maintain backward compatibility with existing code
# New code should use EmbeddingService directly for better performance

def embed(texts):
    """
    Legacy function for backward compatibility.
    
    This function provides the same interface as the original embed() function
    but uses the singleton EmbeddingService under the hood.
    
    Args:
        texts (list): List of strings to encode
    
    Returns:
        numpy.ndarray: Array of embeddings
    
    Note:
        For new code, use EmbeddingService() directly for better performance
        and more control over the embedding process.
    """
    service = EmbeddingService()
    return service.encode(texts)


def get_embedder():
    """
    Legacy function for backward compatibility.
    
    Returns the singleton EmbeddingService instance.
    
    Returns:
        EmbeddingService: The singleton embedding service
    
    Note:
        For new code, use EmbeddingService() directly
    """
    return EmbeddingService()
