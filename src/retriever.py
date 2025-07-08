from .config import settings
from .vector_store import VectorStore

# Legacy ChromaDB implementation (kept for backward compatibility)
try:
    import chromadb
    from chromadb.config import Settings
    
    _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    _collection = None
    _use_chromadb = True
except ImportError:
    _use_chromadb = False

# New Qdrant implementation
_vector_store = None

def get_collection():
    global _collection
    if _collection is None and _use_chromadb:
        try:
            _collection = _client.get_collection("municipal_docs")
        except chromadb.errors.NotFoundError:
            _collection = _client.create_collection("municipal_docs")
    return _collection

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def retrieve(query_embedding):
    """Legacy retrieve function - tries new Qdrant first, falls back to ChromaDB"""
    try:
        # Try new Qdrant implementation first
        vector_store = get_vector_store()
        results = vector_store.search(query_embedding, top_k=settings.TOP_K)
        
        # Convert to legacy format: list of (content, metadata) tuples
        return [(result["content"], result["metadata"]) for result in results]
        
    except Exception as e:
        print(f"Qdrant search failed, falling back to ChromaDB: {e}")
        
        if _use_chromadb:
            # Fallback to ChromaDB
            collection = get_collection()
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=settings.TOP_K,
                include=["documents", "metadatas"]
            )
            return list(zip(result["documents"][0], result["metadatas"][0]))
        else:
            # No fallback available
            raise Exception("Both Qdrant and ChromaDB are unavailable")
