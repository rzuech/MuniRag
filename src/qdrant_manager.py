"""
Qdrant Database Manager
Handles initialization, purging, and management of Qdrant collections
"""

import os
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config import settings
from src.logger import get_logger

logger = get_logger("qdrant_manager")


class QdrantManager:
    """Manages Qdrant database lifecycle and operations"""
    
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        
    def purge_all_collections(self) -> int:
        """
        Purge all collections from Qdrant
        
        Returns:
            Number of collections deleted
        """
        try:
            collections = self.client.get_collections()
            collection_count = len(collections.collections)
            
            if collection_count == 0:
                logger.info("No collections to purge")
                return 0
            
            logger.warning(f"Purging {collection_count} collections...")
            
            for collection in collections.collections:
                try:
                    self.client.delete_collection(collection.name)
                    logger.info(f"  ✓ Deleted collection: {collection.name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to delete {collection.name}: {e}")
            
            logger.info(f"Purge complete. Deleted {collection_count} collections")
            return collection_count
            
        except Exception as e:
            logger.error(f"Error during purge: {e}")
            return 0
    
    def purge_munirag_collections(self) -> int:
        """
        Purge only MuniRAG collections (those starting with 'munirag_')
        
        Returns:
            Number of collections deleted
        """
        try:
            collections = self.client.get_collections()
            munirag_collections = [
                c.name for c in collections.collections 
                if c.name.startswith("munirag_")
            ]
            
            if not munirag_collections:
                logger.info("No MuniRAG collections to purge")
                return 0
            
            logger.warning(f"Purging {len(munirag_collections)} MuniRAG collections...")
            
            for collection_name in munirag_collections:
                try:
                    self.client.delete_collection(collection_name)
                    logger.info(f"  ✓ Deleted collection: {collection_name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to delete {collection_name}: {e}")
            
            logger.info(f"Purge complete. Deleted {len(munirag_collections)} MuniRAG collections")
            return len(munirag_collections)
            
        except Exception as e:
            logger.error(f"Error during purge: {e}")
            return 0
    
    def initialize_on_startup(self):
        """
        Initialize Qdrant on application startup
        Handles purging based on configuration
        """
        # Check if we should reset data on startup
        if settings.RESET_DATA_ON_STARTUP:
            logger.warning("RESET_DATA_ON_STARTUP is enabled - purging all collections")
            self.purge_munirag_collections()
        else:
            # In production mode, just log existing collections
            try:
                collections = self.client.get_collections()
                logger.info(f"Qdrant initialized with {len(collections.collections)} existing collections")
                
                # Log collection details
                for collection in collections.collections:
                    try:
                        info = self.client.get_collection(collection.name)
                        logger.info(f"  - {collection.name}: {info.vectors_count} vectors")
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Error checking collections: {e}")
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and accessible
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collections as a health check
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def create_collection_if_not_exists(self, 
                                      collection_name: str, 
                                      vector_size: int,
                                      distance: Distance = Distance.COSINE) -> bool:
        """
        Create a collection if it doesn't already exist
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric to use
            
        Returns:
            True if created or already exists, False on error
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)
            
            if exists:
                # Verify dimensions match
                info = self.client.get_collection(collection_name)
                existing_dim = info.config.params.vectors.size
                
                if existing_dim != vector_size:
                    logger.error(f"Collection {collection_name} exists with {existing_dim}D vectors, but {vector_size}D requested")
                    return False
                    
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Created collection {collection_name} with {vector_size}D vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            return False


# Singleton instance
_qdrant_manager = None

def get_qdrant_manager() -> QdrantManager:
    """Get or create the singleton QdrantManager instance"""
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager()
    return _qdrant_manager