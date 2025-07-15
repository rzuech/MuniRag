#!/usr/bin/env python3
"""
Reset Qdrant collection with correct dimensions for current embedding model
Use this when switching between models with different dimensions
"""

import sys
sys.path.append('/app')

from src.vector_store import VectorStore
from src.config import settings
from qdrant_client.models import Distance, VectorParams

print("=== Resetting Qdrant Collection ===")

# Get dimension from current model configuration
model_name = settings.EMBEDDING_MODEL
dimension = settings.get_embedding_dimension()

print(f"Current model: {model_name}")
print(f"Dimension: {dimension}")

try:
    vs = VectorStore()
    
    # Delete existing collection
    print("\n1. Deleting existing collection...")
    try:
        vs.client.delete_collection(settings.COLLECTION_NAME)
        print("   ✓ Collection deleted")
    except:
        print("   ⚠ Collection didn't exist")
    
    # Recreate with correct dimensions for current model
    print(f"\n2. Creating fresh collection with {dimension} dimensions...")
    
    vs.client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE
        )
    )
    print(f"   ✓ Collection created for {model_name}")
    
    # Verify
    info = vs.client.get_collection(settings.COLLECTION_NAME)
    actual_dim = info.config.params.vectors.size
    print(f"\n3. Verification: dimension = {actual_dim}")
    
    if actual_dim == dimension:
        print(f"\n✅ Qdrant reset complete! Ready for {model_name} embeddings.")
    else:
        print(f"\n⚠️  Warning: Dimension mismatch! Expected {dimension}, got {actual_dim}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("You may need to restart Qdrant: docker-compose restart qdrant")