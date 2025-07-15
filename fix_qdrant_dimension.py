#!/usr/bin/env python3
"""Fix Qdrant dimension mismatch by recreating collection"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import sys

print("=== Fixing Qdrant Dimension Mismatch ===\n")

# Connect to Qdrant
client = QdrantClient(host='qdrant', port=6333)

# Check current state
try:
    collections = client.get_collections()
    print("Current collections:", [c.name for c in collections.collections])
    
    if any(c.name == 'documents' for c in collections.collections):
        info = client.get_collection('documents')
        print(f"\nCurrent 'documents' collection:")
        print(f"  Vectors count: {info.vectors_count}")
        print(f"  Vector dimension: {info.config.params.vectors.size}")
        
        if info.config.params.vectors.size != 1024:
            print(f"\n❌ Dimension mismatch! Collection has {info.config.params.vectors.size}D vectors")
            print("   but BGE model uses 1024D vectors")
            
            response = input("\nDelete and recreate collection? (y/n): ")
            if response.lower() == 'y':
                # Delete old collection
                client.delete_collection('documents')
                print("✓ Deleted old collection")
                
                # Create new collection with correct dimensions
                client.create_collection(
                    collection_name='documents',
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
                print("✓ Created new collection with 1024 dimensions")
                print("\n✅ Fixed! You can now upload PDFs and search normally.")
            else:
                print("\nTo fix manually, run:")
                print("  docker-compose exec munirag python3 /app/fix_qdrant_dimension.py")
        else:
            print("\n✅ Collection already has correct dimensions (1024)")
    else:
        print("\nNo 'documents' collection found. It will be created automatically.")
        
except Exception as e:
    print(f"\nError checking collection: {e}")
    print("\nCreating new collection with correct dimensions...")
    
    try:
        client.create_collection(
            collection_name='documents',
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print("✓ Created new collection with 1024 dimensions")
    except Exception as e2:
        print(f"Error creating collection: {e2}")