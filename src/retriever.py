"""
Enhanced retriever with multi-model support
"""

from typing import List, Tuple, Dict, Any, Optional
from src.vector_store_v2 import MultiModelVectorStore
from src.config import settings
from src.logger import get_logger

logger = get_logger("retriever")


def retrieve(query_embedding: List[float], 
            model_name: Optional[str] = None,
            top_k: Optional[int] = None,
            filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieve relevant documents for a query
    
    Args:
        query_embedding: Query vector from embedder
        model_name: Specific model to search (defaults to current model)
        top_k: Number of results (defaults to settings.TOP_K)
        filter_dict: Optional metadata filters
        
    Returns:
        List of (content, metadata) tuples
    """
    if top_k is None:
        top_k = settings.TOP_K
    
    if model_name is None:
        model_name = settings.EMBEDDING_MODEL
    
    try:
        # Use multi-model vector store
        vector_store = MultiModelVectorStore(model_name)
        
        # Log search details
        logger.info(f"Searching in collection: {vector_store.collection_name}")
        logger.info(f"Query dimension: {len(query_embedding)}, Expected: {vector_store.dimension}")
        
        # Perform search
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
        
    except ValueError as e:
        if "dimensions" in str(e):
            logger.error(f"Dimension mismatch: {e}")
            logger.error("This usually means the query was embedded with a different model than the stored documents")
            logger.error(f"Current model: {model_name}")
            logger.error("Try using the same model for both embedding and retrieval")
        raise
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        # Return empty results on error
        return []


def retrieve_cross_model(query_embeddings: Dict[str, List[float]], 
                        top_k: Optional[int] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieve from multiple models and merge results
    
    Args:
        query_embeddings: Dict of {model_name: embedding}
        top_k: Number of results per model
        
    Returns:
        Merged and deduplicated results
    """
    if top_k is None:
        top_k = settings.TOP_K
    
    all_results = []
    seen_contents = set()
    
    for model_name, embedding in query_embeddings.items():
        try:
            results = retrieve(embedding, model_name=model_name, top_k=top_k)
            
            # Deduplicate by content
            for content, metadata in results:
                content_hash = hash(content[:200])  # Hash first 200 chars
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    # Add model info to metadata
                    metadata["retrieved_from_model"] = model_name
                    all_results.append((content, metadata))
                    
        except Exception as e:
            logger.warning(f"Error retrieving from {model_name}: {e}")
            continue
    
    # Sort by relevance (could be enhanced with scoring)
    return all_results[:top_k]


def list_available_collections() -> List[Dict[str, Any]]:
    """List all available collections with their info"""
    return MultiModelVectorStore.list_all_collections()