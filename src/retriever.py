"""
Simple document retriever
"""

from typing import List, Tuple, Dict, Any, Optional
from src.vector_store_v2 import MultiModelVectorStore
from src.config import settings
from src.logger import get_logger

logger = get_logger("retriever")


def retrieve(query_embedding: List[float], 
            top_k: Optional[int] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieve relevant documents for a query
    
    Args:
        query_embedding: Query vector from embedder
        top_k: Number of results (defaults to settings.TOP_K)
        
    Returns:
        List of (content, metadata) tuples
    """
    if top_k is None:
        top_k = settings.TOP_K
    
    try:
        # Use multi-model vector store
        vector_store = MultiModelVectorStore()
        
        # Log search details
        logger.info(f"Searching in collection: {vector_store.collection_name}")
        logger.info(f"Query dimension: {len(query_embedding)}")
        
        # Perform search
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
        
    except ValueError as e:
        if "dimensions" in str(e):
            logger.error(f"Dimension mismatch: {e}")
            logger.error("This usually means the query was embedded with a different model than the stored documents")
            logger.error("Try using the same model for both embedding and retrieval")
        raise
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        # Return empty results on error
        return []

