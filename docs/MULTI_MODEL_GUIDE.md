# Multi-Model Support Guide for MuniRAG

This guide explains how to use the enhanced multi-model support in MuniRAG, which handles different embedding models with varying dimensions automatically.

## Overview

The enhanced MuniRAG system now supports:

- **Multiple embedding models** with different dimensions (768D, 1024D)
- **Automatic dimension mismatch detection**
- **Separate collections per model** in Qdrant
- **Seamless model switching** without data loss
- **Migration tools** for moving data between models
- **Unified search interface** across models

## Supported Models

| Model | Dimensions | Max Tokens | GPU Optimized | Notes |
|-------|------------|------------|---------------|-------|
| BAAI/bge-large-en-v1.5 | 1024 | 512 | ✓ | Best GPU performance |
| thenlper/gte-large | 768 | 512 | ✓ | Lightweight alternative |
| hkunlp/instructor-xl | 768 | 512 | ✓ | Task-aware, needs 16GB+ GPU |
| intfloat/e5-large-v2 | 1024 | 512 | ✓ | Requires query/passage prefixes |
| jinaai/jina-embeddings-v3 | 1024 | 8192 | ✗ | Long context, CPU-optimized |

## Key Features

### 1. Automatic Collection Management

Each model gets its own Qdrant collection automatically:

```python
from src.vector_store_v2 import MultiModelVectorStore

# Automatically creates/uses collection for BGE model
vector_store = MultiModelVectorStore("BAAI/bge-large-en-v1.5")

# Automatically creates/uses collection for GTE model  
vector_store = MultiModelVectorStore("thenlper/gte-large")
```

Collections are named based on the model: `munirag_baai_bge_large_en_v1_5`

### 2. Dimension Mismatch Handling

The system automatically detects dimension mismatches:

```python
# Using BGE (1024D)
embedder = EmbeddingModel("BAAI/bge-large-en-v1.5")
embeddings = embedder.embed_documents(texts)  # 1024D vectors

# Switching to GTE (768D) - automatically uses different collection
embedder = EmbeddingModel("thenlper/gte-large")
embeddings = embedder.embed_documents(texts)  # 768D vectors
```

### 3. Model Migration

Migrate data between models with different dimensions:

```python
from src.model_migration import ModelMigrationManager

manager = ModelMigrationManager()

# Check compatibility
is_compatible, msg = manager.check_dimension_compatibility(
    "BAAI/bge-large-en-v1.5",  # 1024D
    "thenlper/gte-large"       # 768D
)
# Returns: (False, "Dimension mismatch: 1024D vs 768D")

# Migrate data (re-embeds with new model)
result = vector_store.migrate_collection(
    source_model="BAAI/bge-large-en-v1.5",
    target_model="thenlper/gte-large",
    embedder_func=lambda texts: embedder.embed_documents(texts),
    batch_size=100
)
```

### 4. Cross-Model Search

Search across multiple models (when dimensions match):

```python
# Search only models with matching dimensions
results = vector_store.search(
    query_embedding=query_emb,  # Must match target dimension
    top_k=10,
    search_all_models=True
)

# Results include model information
for result in results:
    print(f"Model: {result['model']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
```

## Usage Examples

### Basic Usage

```python
from src.vector_store_v2 import MultiModelVectorStore
from src.embedder import EmbeddingModel

# 1. Initialize with a specific model
model_name = "BAAI/bge-large-en-v1.5"
embedder = EmbeddingModel(model_name)
vector_store = MultiModelVectorStore(model_name)

# 2. Add documents
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = embedder.embed_documents(texts)

documents = [{"content": text, "metadata": {"source": "test"}} for text in texts]
vector_store.add_documents(documents, embeddings)

# 3. Search
query = "test query"
query_embedding = embedder.embed_query(query)
results = vector_store.search(query_embedding, top_k=5)
```

### Switching Models

```python
# Start with BGE model (1024D)
vector_store = MultiModelVectorStore("BAAI/bge-large-en-v1.5")
# ... add documents ...

# Switch to GTE model (768D) - data remains in separate collection
vector_store = MultiModelVectorStore("thenlper/gte-large")
# ... add new documents with GTE embeddings ...

# Both collections exist independently
stats = vector_store.get_collection_stats()
print(stats)
# {
#   "munirag_baai_bge_large_en_v1_5": {"points_count": 100, "dimension": 1024},
#   "munirag_thenlper_gte_large": {"points_count": 50, "dimension": 768}
# }
```

### Command-Line Migration

```bash
# List all collections
python src/migrate_models.py --list-collections

# Check model compatibility
python src/migrate_models.py --check-compatibility "BAAI/bge-large-en-v1.5" "thenlper/gte-large"

# Migrate between models
python src/migrate_models.py \
    --source "BAAI/bge-large-en-v1.5" \
    --target "thenlper/gte-large" \
    --batch-size 100

# Migrate and delete source
python src/migrate_models.py \
    --source "BAAI/bge-large-en-v1.5" \
    --target "thenlper/gte-large" \
    --delete-source

# Show migration history
python src/migrate_models.py --history
```

### Streamlit Integration

The enhanced app (`app_enhanced.py`) includes:

1. **Model selector with dimension info** in the sidebar
2. **Automatic dimension mismatch warnings**
3. **Migration center** for data migration
4. **Collection statistics** display
5. **Cross-model search** capabilities

## Configuration

### Environment Variables

```bash
# Enable multi-model support (default: true)
ENABLE_MULTI_MODEL=true

# Use legacy single collection (for backward compatibility)
USE_LEGACY_COLLECTION=false

# Enable cross-model search
ENABLE_CROSS_MODEL_SEARCH=false

# Cross-model search strategy: "all", "same_dimension", "specific_models"
CROSS_MODEL_SEARCH_STRATEGY=same_dimension

# Auto-detect dimension mismatches
AUTO_DETECT_DIMENSION_MISMATCH=true

# Auto-create new collections for dimension mismatches  
AUTO_CREATE_NEW_COLLECTIONS=true

# Show migration prompts in UI
SHOW_MIGRATION_PROMPTS=true

# Migration batch size
MIGRATION_BATCH_SIZE=100
```

### Custom Model Configuration

Add custom models by creating a JSON file:

```json
{
  "custom/my-model": {
    "dimension": 384,
    "max_tokens": 512,
    "gpu_optimized": true,
    "batch_size_gpu": 256,
    "batch_size_cpu": 32,
    "description": "Custom 384D model"
  }
}
```

Then set the environment variable:
```bash
CUSTOM_MODEL_CONFIGS=/path/to/custom_models.json
```

## Best Practices

### 1. Model Selection

- **For GPU systems**: Use BGE or E5 models for best performance
- **For CPU systems**: Consider GTE or smaller models
- **For long documents**: Use Jina (but expect slower performance)

### 2. Migration Strategy

- **Same dimension models** (e.g., BGE ↔ E5): Can switch without migration
- **Different dimensions**: Requires re-embedding all documents
- **Large datasets**: Migrate in off-peak hours
- **Always backup**: Keep source collection until migration is verified

### 3. Performance Optimization

```python
# Batch ingestion for better performance
texts = ["doc1", "doc2", "doc3", ...]  # Many documents
embeddings = embedder.embed_documents(texts)  # Batch embedding

# Add all at once
documents = [{"content": text} for text in texts]
vector_store.add_documents(documents, embeddings)
```

### 4. Error Handling

```python
try:
    # Attempt to use a model
    vector_store = MultiModelVectorStore("BAAI/bge-large-en-v1.5")
    vector_store.add_documents(documents, embeddings)
except ValueError as e:
    if "Dimension mismatch" in str(e):
        # Handle dimension mismatch
        print(f"Error: {e}")
        print("Consider migrating your data to the new model")
```

## Troubleshooting

### Common Issues

1. **"Dimension mismatch" error**
   - You're trying to add embeddings with wrong dimensions
   - Solution: Ensure embedder model matches vector store model

2. **"Collection exists with wrong dimension"**
   - Trying to use a model with existing data of different dimensions
   - Solution: Either migrate the data or use a different collection name

3. **Migration takes too long**
   - Large datasets can take time to re-embed
   - Solution: Use larger batch sizes, run during off-peak hours

4. **Out of memory during migration**
   - Batch size too large for available memory
   - Solution: Reduce batch size in migration

### Debug Commands

```python
# Check collection statistics
vector_store = MultiModelVectorStore()
stats = vector_store.get_collection_stats()
print(json.dumps(stats, indent=2))

# Verify model dimensions
from src.config_v2 import enhanced_settings
for model in enhanced_settings.get_available_models():
    dim = enhanced_settings.get_embedding_dimension(model)
    print(f"{model}: {dim}D")

# Test embedding compatibility
embedder = EmbeddingModel("your-model")
test_embedding = embedder.embed_query("test")
print(f"Embedding dimension: {len(test_embedding)}")
```

## Migration Scenarios

### Scenario 1: Upgrading to Better Model (Same Dimension)

```python
# BGE (1024D) to E5 (1024D) - No migration needed!
# Just switch the model in settings
enhanced_settings.EMBEDDING_MODEL = "intfloat/e5-large-v2"

# Both models can access the same collection if needed
# But recommended to keep separate for clarity
```

### Scenario 2: Changing Dimensions for Performance

```python
# From Jina (1024D) to GTE (768D) for better GPU performance
# Requires full migration

# 1. Check current data
stats = vector_store.get_collection_stats()
print(f"Current documents: {stats['munirag_jinaai_jina_embeddings_v3']['points_count']}")

# 2. Run migration
python src/migrate_models.py \
    --source "jinaai/jina-embeddings-v3" \
    --target "thenlper/gte-large" \
    --delete-source  # Remove old data after migration
```

### Scenario 3: Testing Multiple Models

```python
# Keep multiple models active for A/B testing
models = ["BAAI/bge-large-en-v1.5", "thenlper/gte-large", "intfloat/e5-large-v2"]

results = {}
for model in models:
    embedder = EmbeddingModel(model)
    vector_store = MultiModelVectorStore(model)
    
    # Test query
    query_emb = embedder.embed_query("test query")
    search_results = vector_store.search(query_emb, top_k=5)
    
    results[model] = {
        "top_score": search_results[0]["score"] if search_results else 0,
        "avg_score": sum(r["score"] for r in search_results) / len(search_results)
    }

# Compare results
print(json.dumps(results, indent=2))
```

## Future Enhancements

Planned improvements for the multi-model system:

1. **Automatic model selection** based on document characteristics
2. **Ensemble search** combining results from multiple models
3. **Model performance tracking** and analytics
4. **Automatic migration scheduling** for large datasets
5. **Model-specific optimization** profiles
6. **Cross-model reranking** strategies

## Conclusion

The multi-model support in MuniRAG provides flexibility to use different embedding models without losing data or requiring complex migrations. The system automatically handles dimension differences and provides tools for seamless transitions between models.

For questions or issues, please refer to the GitHub repository or raise an issue.