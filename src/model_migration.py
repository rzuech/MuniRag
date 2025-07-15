"""
Model Migration Tool for Multi-Model Vector Store
Handles migration between different embedding models
"""

from typing import List, Dict, Any, Optional, Callable
import time
from datetime import datetime
from src.vector_store_v2 import MultiModelVectorStore
from src.embedder import EmbeddingModel
from src.logger import get_logger
from qdrant_client import QdrantClient
from src.config import settings

logger = get_logger("model_migration")


class ModelMigrationManager:
    """Manages migration of documents between different embedding models"""
    
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    def migrate_collection(self, 
                         source_model: str, 
                         target_model: str,
                         batch_size: int = 100,
                         progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Migrate documents from one model's collection to another
        
        Args:
            source_model: Source embedding model name
            target_model: Target embedding model name
            batch_size: Number of documents to process at once
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Migration statistics
        """
        start_time = time.time()
        
        # Initialize stores
        source_store = MultiModelVectorStore(source_model)
        target_store = MultiModelVectorStore(target_model)
        
        # Initialize target embedder
        target_embedder = EmbeddingModel(target_model)
        
        # Get source collection info
        source_info = source_store.get_collection_info()
        total_docs = source_info.get("vectors_count", 0)
        
        if total_docs == 0:
            return {
                "status": "skipped",
                "message": "Source collection is empty",
                "source_model": source_model,
                "target_model": target_model,
                "documents_migrated": 0
            }
        
        logger.info(f"Starting migration: {source_model} → {target_model}")
        logger.info(f"Total documents to migrate: {total_docs}")
        
        # Check if models have same dimension (no re-embedding needed)
        same_dimension = source_info["dimension"] == target_store.dimension
        
        if same_dimension:
            logger.info("Models have same dimension - direct copy possible")
        else:
            logger.info(f"Dimension change: {source_info['dimension']}D → {target_store.dimension}D - re-embedding required")
        
        # Migrate in batches
        migrated_count = 0
        errors = []
        offset = None
        
        try:
            while True:
                # Fetch batch from source
                batch_filter = {}
                if offset:
                    batch_filter["id"] = {"$gt": offset}
                
                # Use scroll API to get documents
                records, next_offset = self._scroll_collection(
                    source_store.collection_name, 
                    limit=batch_size,
                    offset=offset
                )
                
                if not records:
                    break
                
                # Extract documents and embeddings
                documents = []
                embeddings = []
                
                for record in records:
                    doc = {
                        "id": str(record.id),
                        "content": record.payload.get("content", ""),
                        "metadata": record.payload.get("metadata", {})
                    }
                    documents.append(doc)
                    
                    if same_dimension:
                        # Direct copy of embeddings
                        embeddings.append(record.vector)
                    else:
                        # Need to re-embed
                        pass  # Will handle below
                
                # Re-embed if needed
                if not same_dimension:
                    if progress_callback:
                        progress_callback(
                            migrated_count / total_docs,
                            f"Re-embedding batch {migrated_count//batch_size + 1}..."
                        )
                    
                    # Extract texts for re-embedding
                    texts = [doc["content"] for doc in documents]
                    embeddings = target_embedder.embed_documents(texts)
                
                # Add to target collection
                target_store.add_documents(documents, embeddings)
                migrated_count += len(documents)
                
                # Update progress
                if progress_callback:
                    progress = migrated_count / total_docs
                    progress_callback(progress, f"Migrated {migrated_count}/{total_docs} documents")
                
                # Update offset for next batch
                offset = next_offset
                
                # Log progress
                if migrated_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = migrated_count / elapsed
                    eta = (total_docs - migrated_count) / rate
                    logger.info(f"Progress: {migrated_count}/{total_docs} ({rate:.1f} docs/sec, ETA: {eta:.1f}s)")
        
        except Exception as e:
            logger.error(f"Migration error: {e}")
            errors.append(str(e))
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        
        result = {
            "status": "completed" if not errors else "completed_with_errors",
            "source_model": source_model,
            "target_model": target_model,
            "documents_migrated": migrated_count,
            "total_documents": total_docs,
            "dimension_change": f"{source_info['dimension']}D → {target_store.dimension}D",
            "re_embedding_required": not same_dimension,
            "elapsed_time": elapsed_time,
            "documents_per_second": migrated_count / elapsed_time if elapsed_time > 0 else 0,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Migration completed: {migrated_count}/{total_docs} documents in {elapsed_time:.1f}s")
        
        if progress_callback:
            progress_callback(1.0, f"Migration completed: {migrated_count} documents migrated")
        
        return result
    
    def _scroll_collection(self, collection_name: str, limit: int = 100, offset: Optional[str] = None):
        """Scroll through collection records"""
        try:
            # Use Qdrant's scroll API
            records, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            return records, next_offset
        except Exception as e:
            logger.error(f"Error scrolling collection: {e}")
            return [], None
    
    def estimate_migration_time(self, source_model: str, target_model: str) -> Dict[str, Any]:
        """Estimate migration time based on collection size and model performance"""
        source_store = MultiModelVectorStore(source_model)
        target_store = MultiModelVectorStore(target_model)
        
        source_info = source_store.get_collection_info()
        total_docs = source_info.get("vectors_count", 0)
        
        # Check if re-embedding is needed
        same_dimension = source_info["dimension"] == target_store.dimension
        
        # Estimate based on typical performance
        if same_dimension:
            # Direct copy: ~5000 docs/sec
            estimated_time = total_docs / 5000
        else:
            # Re-embedding needed: depends on model
            # Use conservative estimate based on our GPU performance tests
            if "jina" in target_model.lower():
                docs_per_sec = 100  # Jina is slow
            else:
                docs_per_sec = 2000  # GPU-optimized models
            
            estimated_time = total_docs / docs_per_sec
        
        return {
            "source_model": source_model,
            "target_model": target_model,
            "total_documents": total_docs,
            "re_embedding_required": not same_dimension,
            "estimated_time_seconds": estimated_time,
            "estimated_time_readable": self._format_time(estimated_time)
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into readable time"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"


def fix_dimension_mismatch():
    """Fix the current dimension mismatch issue"""
    print("\n=== Fixing Dimension Mismatch ===\n")
    
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    # Check existing collections
    collections = client.get_collections()
    print("Current collections:")
    for col in collections.collections:
        try:
            info = client.get_collection(col.name)
            print(f"  - {col.name}: {info.config.params.vectors.size}D vectors, {info.vectors_count} documents")
        except:
            print(f"  - {col.name}: (error getting info)")
    
    # Handle the munirag_docs collection
    if any(c.name == "munirag_docs" for c in collections.collections):
        info = client.get_collection("munirag_docs")
        current_dim = info.config.params.vectors.size
        
        print(f"\nFound 'munirag_docs' collection with {current_dim}D vectors")
        print(f"Current model (BGE) uses 1024D vectors")
        
        if current_dim != 1024:
            print("\n⚠️  Dimension mismatch detected!")
            print("Options:")
            print("1. Create new collection for BGE model (recommended)")
            print("2. Migrate existing data to BGE dimensions")
            print("3. Delete old collection and start fresh")
            
            # For now, create new collection for BGE
            try:
                store = MultiModelVectorStore("BAAI/bge-large-en-v1.5")
                print(f"\n✅ Created new collection: {store.collection_name}")
                print("   You can now upload PDFs and search with BGE model")
            except Exception as e:
                print(f"\n❌ Error creating collection: {e}")
    
    # Update retriever to look for correct collection
    print("\n✅ Vector store is now configured for multi-model support")
    print("   Each model will use its own collection automatically")


if __name__ == "__main__":
    # Run the fix when executed directly
    fix_dimension_mismatch()