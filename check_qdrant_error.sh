#!/bin/bash
echo "=== Checking Qdrant Error ==="
echo ""
echo "1. Qdrant logs (last 50 lines):"
docker-compose logs --tail=50 qdrant

echo ""
echo "2. MuniRAG logs (last 30 lines):"
docker-compose logs --tail=30 munirag

echo ""
echo "3. Current collections in Qdrant:"
docker-compose exec munirag python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='qdrant', port=6333)
collections = client.get_collections()
print('Collections:', [c.name for c in collections.collections])
"

echo ""
echo "4. Checking collection details:"
docker-compose exec munirag python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='qdrant', port=6333)
try:
    info = client.get_collection('documents')
    print(f'Collection: documents')
    print(f'Vectors count: {info.vectors_count}')
    print(f'Vector size: {info.config.params.vectors.size}')
except Exception as e:
    print(f'Error: {e}')
"