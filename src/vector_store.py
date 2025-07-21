"""
Enhanced Vector Store with Multi-Model Support
Handles different embedding dimensions gracefully
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest, SearchParams
)
from src.config import settings
from src.logger import get_logger
from src.model_registry import get_model_config, EMBEDDING_MODELS

logger = get_logger("vector_store")


class MultiModelVectorStore:
    """Vector store that handles multiple embedding models with different dimensions"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize vector store for a specific model
        
        Args:
            model_name: The embedding model name (e.g., "BAAI/bge-large-en-v1.5")
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        
        # Get or detect dimension
        self.dimension = self._get_model_dimension(self.model_name)
        
        # Collection name includes model identifier
        self.collection_name = self._get_collection_name(self.model_name)
        
        # Initialize collection if needed
        self._ensure_collection_exists()
        
        logger.info(f"Initialized MultiModelVectorStore for {self.model_name}")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info(f"  Dimension: {self.dimension}")
    
    def _get_model_dimension(self, model_name: str) -> int:
        """Get dimension for a model, with fallback to dynamic detection"""
        # Try model registry first
        config = get_model_config(model_name)
        if config:
            return config.dimension
        
        # Try to get from settings
        try:
            return settings.get_embedding_dimension()
        except:
            logger.warning(f"Unknown model {model_name}, defaulting to 768 dimensions")
            return 768
    
    def _get_collection_name(self, model_name: str) -> str:
        """Generate collection name from model name"""
        # Clean model name for use as collection name
        clean_name = model_name.lower().replace("/", "_").replace("-", "_")
        return f"munirag_{clean_name}"
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                logger.info(f"Creating collection {self.collection_name} with dimension {self.dimension}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created collection {self.collection_name}")
            else:
                # Verify dimension matches
                info = self.client.get_collection(self.collection_name)
                existing_dim = info.config.params.vectors.size
                
                if existing_dim != self.dimension:
                    logger.error(f"Dimension mismatch! Collection {self.collection_name} has {existing_dim}D vectors but model uses {self.dimension}D")
                    raise ValueError(f"Dimension mismatch: collection has {existing_dim}D, model needs {self.dimension}D")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """
        Add documents with embeddings to the collection
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            embeddings: List of embedding vectors
            
        Returns:
            List of document IDs
        """
        if not documents or not embeddings:
            return []
        
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings")
        
        # Validate embedding dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) != self.dimension:
                raise ValueError(f"Embedding {i} has {len(emb)} dimensions, expected {self.dimension}")
        
        points = []
        doc_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.get("id") or str(uuid4())
            doc_ids.append(doc_id)
            
            # Add model info to metadata
            metadata = doc.get("metadata", {})
            metadata["embedding_model"] = self.model_name
            metadata["embedding_dimension"] = self.dimension
            
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": doc.get("content", ""),
                    "metadata": metadata
                }
            )
            points.append(point)
        
        # Batch upload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Added {len(documents)} documents to {self.collection_name}")
        return doc_ids
    
    def search(self, query_embedding: List[float], top_k: int = 4, 
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Optional metadata filters
            
        Returns:
            List of (content, metadata) tuples
        """
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding has {len(query_embedding)} dimensions, expected {self.dimension}")
        
        # Build filter if provided
        filter_obj = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                filter_obj = Filter(must=conditions)
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=filter_obj,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                content = result.payload.get("content", "")
                metadata = result.payload.get("metadata", {})
                formatted_results.append((content, metadata))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error in {self.collection_name}: {e}")
            # Return empty results on error
            return []
    
    def delete_collection(self):
        """Delete this model's collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about this collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "model": self.model_name,
                "dimension": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "status": "ready"
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "model": self.model_name,
                "dimension": self.dimension,
                "vectors_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    @classmethod
    def list_all_collections(cls) -> List[Dict[str, Any]]:
        """List all MuniRAG collections across all models"""
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        collections_info = []
        
        try:
            collections = client.get_collections()
            for collection in collections.collections:
                if collection.name.startswith("munirag_"):
                    try:
                        info = client.get_collection(collection.name)
                        # Try to extract model name from collection name
                        model_name = collection.name.replace("munirag_", "").replace("_", "/")
                        
                        collections_info.append({
                            "name": collection.name,
                            "model": model_name,
                            "dimension": info.config.params.vectors.size,
                            "vectors_count": info.vectors_count
                        })
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
        
        return collections_info
    
    def migrate_from_collection(self, source_collection: str, batch_size: int = 100) -> int:
        """
        Migrate data from another collection to this one
        
        Args:
            source_collection: Name of source collection
            batch_size: Number of points to migrate at a time
            
        Returns:
            Number of documents migrated
        """
        logger.info(f"Starting migration from {source_collection} to {self.collection_name}")
        
        try:
            # Get source collection info
            source_info = self.client.get_collection(source_collection)
            total_points = source_info.points_count
            
            if total_points == 0:
                logger.info("Source collection is empty, nothing to migrate")
                return 0
            
            logger.info(f"Migrating {total_points} documents...")
            
            migrated = 0
            offset = None
            
            while True:
                # Scroll through source collection
                records, next_offset = self.client.scroll(
                    collection_name=source_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not records:
                    break
                
                # Check if dimensions match
                if records and len(records[0].vector) != self.dimension:
                    logger.error(f"Dimension mismatch! Source has {len(records[0].vector)}D, target needs {self.dimension}D")
                    logger.error("Migration aborted - dimensions must match")
                    return migrated
                
                # Convert to points for new collection
                points = []
                for record in records:
                    # Update metadata with model info
                    payload = record.payload.copy()
                    if "metadata" not in payload:
                        payload["metadata"] = {}
                    payload["metadata"]["embedding_model"] = self.model_name
                    payload["metadata"]["embedding_dimension"] = self.dimension
                    payload["metadata"]["migrated_from"] = source_collection
                    
                    point = PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=payload
                    )
                    points.append(point)
                
                # Insert into target collection
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                migrated += len(records)
                logger.info(f"  Migrated {migrated}/{total_points} documents...")
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"✅ Migration complete! Migrated {migrated} documents")
            return migrated
            
        except Exception as e:
            logger.error(f"Migration error: {e}")
            raise


# For backward compatibility - we DON'T want this anymore
# Instead, we'll use MultiModelVectorStore everywhere
VectorStore = MultiModelVectorStore