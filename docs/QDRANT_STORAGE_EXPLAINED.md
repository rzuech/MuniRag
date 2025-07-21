# Qdrant Storage Architecture in MuniRAG

## Overview

MuniRAG uses Qdrant as its vector database to store document embeddings and metadata. This document explains how data is organized and stored within Qdrant collections.

## Current Collections

### 1. Main Document Collection: `munirag_baai_bge_large_en_v1.5`

This is the primary collection storing all ingested documents.

**Structure:**
- **Collection Name**: Dynamically generated based on embedding model
  - Format: `munirag_{sanitized_model_name}`
  - Example: `munirag_baai_bge_large_en_v1.5`

**Vector Configuration:**
- **Dimension**: 1024 (BGE model output)
- **Distance Metric**: Cosine similarity
- **On-disk Storage**: Enabled for large datasets

**Point (Document) Structure:**
Each point in the collection represents a text chunk with:

```json
{
  "id": "UUID (auto-generated)",
  "vector": [1024-dimensional float array],
  "payload": {
    "content": "The actual text chunk content...",
    "metadata": {
      "file_path": "/app/Test-PDFs/weston-code-of-ordinances.pdf",
      "title": "Weston Code of Ordinances",
      "author": "City of Weston",
      "subject": "Municipal Code",
      "keywords": "ordinances, regulations, municipal law",
      "creator": "Adobe Acrobat",
      "producer": "Adobe PDF Library",
      "pages": 825,
      "processing_time": 45.23,
      "chunk_index": 142,
      "tokens": 487,
      "sentences": 12,
      "semantic_coherence": 0.92,
      "embedding_model": "BAAI/bge-large-en-v1.5",
      "embedding_dimension": 1024
    }
  }
}
```

**Current Storage Stats:**
- Total Points: 14,966
- Storage Size: ~60MB (vectors) + ~20MB (payloads)
- Index Type: HNSW (Hierarchical Navigable Small World)

### 2. Test Collections (Planned)

For accuracy testing and monitoring:

#### `munirag_test_results`
- Stores test run summaries
- Dimension: 384 (smaller model for metadata search)
- Used for tracking accuracy trends over time

#### `munirag_accuracy_metrics`
- Aggregated metrics from multiple test runs
- Used for performance monitoring

#### `munirag_baseline_scores`
- Baseline performance benchmarks
- Used for regression detection

## Data Flow

### 1. Document Ingestion
```
PDF File → PyMuPDF Extraction → Text Chunks → Embeddings → Qdrant Storage
```

1. **PDF Processing**: Extract text with metadata
2. **Chunking**: Split into semantic chunks (~500 tokens)
3. **Embedding**: Convert to 1024-dim vectors via BGE
4. **Storage**: Insert into Qdrant with metadata

### 2. Query Processing
```
User Query → Embedding → Vector Search → Retrieved Chunks → LLM Response
```

1. **Query Embedding**: Same model as documents
2. **Similarity Search**: Find top-k nearest vectors
3. **Metadata Retrieval**: Get content + source info
4. **Context Building**: Format for LLM prompt

## Storage Organization

### File System
```
./qdrant_storage/
├── collections/
│   └── munirag_baai_bge_large_en_v1.5/
│       ├── segments/           # Vector index segments
│       ├── payload_index/      # Metadata indices
│       └── collection.json     # Collection config
└── storage.json               # Qdrant configuration
```

### Docker Volume
- Mounted at: `./qdrant_storage:/qdrant/storage`
- Persistent across container restarts
- Backup-friendly directory structure

## Key Features

### 1. Multi-Model Support
Collections are model-specific to ensure dimension consistency:
- `munirag_jinaai_jina_embeddings_v3` (1024 dim)
- `munirag_baai_bge_large_en_v1.5` (1024 dim)
- `munirag_sentence_transformers_all_minilm_l6_v2` (384 dim)

### 2. Metadata Filtering
Qdrant supports filtering by metadata fields:
```python
# Example: Find chunks from specific PDF
filter_dict = {"file_path": "/app/Test-PDFs/specific.pdf"}
```

### 3. Payload Indexing
Indexed fields for fast filtering:
- `metadata.file_path`
- `metadata.chunk_index`
- `metadata.embedding_model`

## Performance Characteristics

### Search Performance
- **Vector Search**: ~5-10ms for top-4 retrieval
- **With Filtering**: ~10-20ms depending on filter complexity
- **Batch Operations**: Optimized for 1000-point batches

### Storage Efficiency
- **Compression**: Scalar quantization available
- **Memory Mapping**: Efficient disk-based operations
- **Caching**: Hot vectors kept in memory

## Maintenance Operations

### 1. Collection Purging
```python
# Remove all munirag collections except test/metrics
manager.purge_munirag_collections()
```

### 2. Health Checks
```python
# Check Qdrant status
manager.health_check()
```

### 3. Backup
```bash
# Backup entire Qdrant storage
tar -czf qdrant_backup.tar.gz ./qdrant_storage/
```

### 4. Collection Recreation
```python
# If corrupted, recreate collection
vector_store._create_collection_if_not_exists()
```

## Monitoring

### Collection Info
```python
info = vector_store.get_collection_info()
# Returns: vectors_count, points_count, segments_count
```

### Storage Usage
```bash
du -sh ./qdrant_storage/
# Shows total disk usage
```

### Performance Metrics
- Monitor via Qdrant dashboard: http://localhost:6333/dashboard
- Metrics endpoint: http://localhost:6333/metrics

## Best Practices

1. **Consistent Models**: Always use same embedding model for ingest/query
2. **Batch Operations**: Insert/update in batches of 1000
3. **Regular Backups**: Backup qdrant_storage directory
4. **Monitor Growth**: Track collection size over time
5. **Index Optimization**: Rebuild indices after major changes

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**
   - Cause: Different embedding models
   - Fix: Ensure consistent MODEL environment variable

2. **Empty Results**
   - Cause: Collection doesn't exist or is empty
   - Fix: Check collection status, re-ingest if needed

3. **Slow Searches**
   - Cause: Large collection without optimization
   - Fix: Rebuild HNSW index, adjust ef_construct

4. **Storage Full**
   - Cause: Too many vectors/payloads
   - Fix: Increase disk space, enable compression

## Future Enhancements

1. **Hybrid Search**: Combine vector + keyword search
2. **Multi-tenancy**: Separate collections per department
3. **Version Control**: Track document versions
4. **Automatic Optimization**: Self-tuning indices
5. **Distributed Mode**: Multi-node Qdrant cluster