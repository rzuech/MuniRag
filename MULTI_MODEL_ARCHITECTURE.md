# Multi-Model Architecture for MuniRAG

## Overview

MuniRAG now supports multiple embedding models with different dimensions through an intelligent collection management system. Each embedding model gets its own Qdrant collection, preventing dimension mismatches and allowing seamless model switching.

## Architecture

### 1. Collection Naming Convention
Each model's vectors are stored in a separate collection:
- `munirag_baai_bge_large_en_v1_5` - BGE model (1024D)
- `munirag_thenlper_gte_large` - GTE model (768D)
- `munirag_hkunlp_instructor_xl` - Instructor model (768D)
- `munirag_intfloat_e5_large_v2` - E5 model (1024D)
- `munirag_jinaai_jina_embeddings_v3` - Jina model (1024D)

### 2. Key Components

#### MultiModelVectorStore (`src/vector_store_v2.py`)
- Automatically creates collections with correct dimensions
- Validates dimension compatibility on operations
- Provides unified interface for all models
- Includes backward compatibility wrapper

#### Enhanced Retriever (`src/retriever.py`)
- Automatically selects correct collection based on model
- Supports cross-model search (search multiple collections)
- Provides detailed error messages for dimension mismatches

#### Model Migration Manager (`src/model_migration.py`)
- Migrates documents between models
- Handles re-embedding when dimensions differ
- Provides progress tracking and time estimates

## Usage Examples

### 1. Basic Usage (Automatic Collection Selection)
```python
from src.embedder import EmbeddingModel
from src.retriever import retrieve

# Embed with current model (BGE)
embedder = EmbeddingModel()
query_embedding = embedder.embed_query("What are the zoning laws?")

# Retrieve automatically uses the same model's collection
results = retrieve(query_embedding)
```

### 2. Switching Models
```python
# Switch to GTE model (768D)
embedder_gte = EmbeddingModel("thenlper/gte-large")
vector_store_gte = MultiModelVectorStore("thenlper/gte-large")

# This creates a new collection for GTE automatically
# No dimension conflicts!
```

### 3. Migrating Between Models
```python
from src.model_migration import ModelMigrationManager

manager = ModelMigrationManager()

# Estimate migration time
estimate = manager.estimate_migration_time(
    source_model="BAAI/bge-large-en-v1.5",
    target_model="thenlper/gte-large"
)
print(f"Estimated time: {estimate['estimated_time_readable']}")

# Perform migration
result = manager.migrate_collection(
    source_model="BAAI/bge-large-en-v1.5",
    target_model="thenlper/gte-large",
    batch_size=1000
)
```

### 4. Cross-Model Search
```python
from src.retriever import retrieve_cross_model

# Search across multiple models
query_embeddings = {
    "BAAI/bge-large-en-v1.5": embedder_bge.embed_query(query),
    "thenlper/gte-large": embedder_gte.embed_query(query)
}

results = retrieve_cross_model(query_embeddings)
```

## Benefits

1. **No Dimension Conflicts**: Each model uses its own collection
2. **Easy Model Switching**: Change models without losing data
3. **Gradual Migration**: Test new models while keeping old data
4. **Performance Tracking**: Know which model performs best
5. **Future Proof**: Add new models without breaking existing data

## Current State

After applying the fix:
- Existing data in `munirag_docs` (1024D) remains intact
- New BGE embeddings go to `munirag_baai_bge_large_en_v1_5`
- Each model creates its own collection automatically
- The system prevents dimension mismatches

## Migration Path

If you have existing data in `munirag_docs`:
1. It's already 1024D (compatible with BGE/E5/Jina)
2. To use with 768D models (GTE/Instructor), migration is required
3. Use the migration manager to re-embed documents

## Configuration

Set the default model in `.env`:
```env
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

## Troubleshooting

### "Dimension mismatch" errors
- Ensure query and documents use the same model
- Check collection dimensions match the model
- Use migration tool if switching between incompatible models

### Performance considerations
- Same-dimension migrations are fast (direct copy)
- Different-dimension migrations require re-embedding
- BGE model provides best GPU performance (14,000 texts/sec)

## Future Enhancements

1. **Automatic model selection** based on query characteristics
2. **Hybrid search** combining multiple models
3. **Model performance analytics** to track accuracy
4. **Automatic migration scheduling** during low usage
5. **Model versioning** (e.g., bge-v1.5 vs bge-v2.0)