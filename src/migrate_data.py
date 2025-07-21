#!/usr/bin/env python3
"""
Data Migration Script
Migrates data from old munirag_docs collection to new model-specific collections
"""

import sys
import argparse
from typing import Optional
from qdrant_client import QdrantClient
from src.vector_store import MultiModelVectorStore
from src.config import settings
from src.logger import get_logger
from src.model_registry import EMBEDDING_MODELS, get_model_config

logger = get_logger("migrate_data")


def check_collection_info(client: QdrantClient, collection_name: str) -> Optional[dict]:
    """Get information about a collection"""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "dimension": info.config.params.vectors.size,
            "points_count": info.points_count,
            "status": info.status
        }
    except Exception as e:
        logger.error(f"Collection {collection_name} not found: {e}")
        return None


def find_compatible_model(dimension: int) -> Optional[str]:
    """Find a model compatible with the given dimension"""
    compatible_models = []
    for model_name, config in EMBEDDING_MODELS.items():
        if config.dimension == dimension:
            compatible_models.append(model_name)
    
    if compatible_models:
        # Prefer BGE models for best performance
        for model in compatible_models:
            if "bge" in model:
                return model
        # Otherwise return first compatible
        return compatible_models[0]
    
    return None


def migrate_collection(source_name: str, target_model: Optional[str] = None) -> bool:
    """
    Migrate a collection to the new multi-model architecture
    
    Args:
        source_name: Source collection name
        target_model: Target model name (auto-detected if not provided)
        
    Returns:
        Success status
    """
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    # Check source collection
    source_info = check_collection_info(client, source_name)
    if not source_info:
        logger.error(f"Source collection {source_name} not found")
        return False
    
    logger.info(f"Source collection: {source_name}")
    logger.info(f"  Dimension: {source_info['dimension']}")
    logger.info(f"  Documents: {source_info['points_count']}")
    
    if source_info['points_count'] == 0:
        logger.info("Source collection is empty, nothing to migrate")
        return True
    
    # Determine target model if not specified
    if not target_model:
        target_model = find_compatible_model(source_info['dimension'])
        if not target_model:
            logger.error(f"No compatible model found for dimension {source_info['dimension']}")
            logger.error("Please specify a target model manually")
            return False
        logger.info(f"Auto-selected model: {target_model}")
    
    # Verify model compatibility
    model_config = get_model_config(target_model)
    if not model_config:
        logger.error(f"Unknown model: {target_model}")
        return False
    
    if model_config.dimension != source_info['dimension']:
        logger.error(f"Dimension mismatch! Source: {source_info['dimension']}D, Target model: {model_config.dimension}D")
        logger.error("Cannot migrate between different dimensions")
        return False
    
    # Create target vector store
    target_store = MultiModelVectorStore(model_name=target_model)
    logger.info(f"Target collection: {target_store.collection_name}")
    
    # Check if target already has data
    target_info = check_collection_info(client, target_store.collection_name)
    if target_info and target_info['points_count'] > 0:
        logger.warning(f"Target collection already has {target_info['points_count']} documents")
        response = input("Continue with migration? This will add to existing data (y/N): ")
        if response.lower() != 'y':
            logger.info("Migration cancelled")
            return False
    
    # Perform migration
    try:
        migrated = target_store.migrate_from_collection(source_name)
        logger.info(f"✅ Successfully migrated {migrated} documents")
        
        # Verify migration
        target_info = check_collection_info(client, target_store.collection_name)
        if target_info:
            logger.info(f"Target collection now has {target_info['points_count']} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def list_collections():
    """List all collections in Qdrant"""
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    try:
        collections = client.get_collections()
        logger.info(f"Found {len(collections.collections)} collections:")
        
        for collection in collections.collections:
            info = check_collection_info(client, collection.name)
            if info:
                logger.info(f"  - {info['name']}: {info['dimension']}D, {info['points_count']} documents")
    
    except Exception as e:
        logger.error(f"Error listing collections: {e}")


def main():
    parser = argparse.ArgumentParser(description="Migrate Qdrant collections for multi-model support")
    parser.add_argument("--list", action="store_true", help="List all collections")
    parser.add_argument("--source", type=str, help="Source collection name")
    parser.add_argument("--target-model", type=str, help="Target model name (auto-detected if not specified)")
    parser.add_argument("--delete-source", action="store_true", help="Delete source collection after successful migration")
    
    args = parser.parse_args()
    
    if args.list:
        list_collections()
        return
    
    if not args.source:
        # Default migration path
        logger.info("No source specified, checking for default migration...")
        args.source = "munirag_docs"
    
    success = migrate_collection(args.source, args.target_model)
    
    if success and args.delete_source:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        try:
            client.delete_collection(args.source)
            logger.info(f"Deleted source collection {args.source}")
        except Exception as e:
            logger.error(f"Failed to delete source collection: {e}")


if __name__ == "__main__":
    main()