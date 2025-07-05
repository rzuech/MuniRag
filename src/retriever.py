import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, TOP_K

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        try:
            _collection = _client.get_collection("municipal_docs")
        except chromadb.errors.NotFoundError:
            _collection = _client.create_collection("municipal_docs")
    return _collection

def retrieve(query_embedding):
    collection = get_collection()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )
    return list(zip(result["documents"][0], result["metadatas"][0]))
