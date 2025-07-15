# Hybrid Multi-Model Approach for MuniRAG

## Overview

MuniRAG uses a **hybrid collection-based approach** to handle multiple embedding models with different dimensions. This design prevents dimension conflicts while maintaining flexibility and performance.

## Core Concepts

### 1. Collection Isolation
Each embedding model gets its own dedicated Qdrant collection:
- **Benefit**: Complete dimension isolation - no conflicts possible
- **Trade-off**: Separate storage per model (acceptable for most use cases)

### 2. Automatic Collection Naming
Collections are named based on the model:
```
munirag_{model_name_sanitized}
```
Examples:
- `munirag_baai_bge_large_en_v1_5` (1024D)
- `munirag_thenlper_gte_large` (768D)
- `munirag_hkunlp_instructor_xl` (768D)

### 3. Dimension Registry
Known models have pre-configured dimensions:
```python
MODEL_DIMENSIONS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "thenlper/gte-large": 768,
    "hkunlp/instructor-xl": 768,
    "intfloat/e5-large-v2": 1024,
    "jinaai/jina-embeddings-v3": 1024,
}
```

## Hybrid Scenarios

### Scenario 1: Same Model Throughout (Most Common)
```
User → Embed Query → Search Same Collection → Results
      (BGE 1024D)    (BGE collection)
```
- **Performance**: Optimal (no overhead)
- **Accuracy**: Best (same model characteristics)

### Scenario 2: Model Switching
```
Initial: PDFs → BGE Embeddings → BGE Collection
Switch:  User → GTE Query → ❌ Dimension Mismatch
Solution: Create new GTE collection, re-embed documents
```
- **Handled by**: Migration Manager
- **Options**: Keep both collections or migrate fully

### Scenario 3: Cross-Model Search (Advanced)
```
Query → Embed with Model A → Search Collection A ─┐
     └→ Embed with Model B → Search Collection B ─┼→ Merge Results
     └→ Embed with Model C → Search Collection C ─┘
```
- **Use case**: Compare model performance
- **Trade-off**: Multiple embeddings needed

### Scenario 4: Compatible Models (Same Dimension)
```
BGE (1024D) ←→ E5 (1024D) ←→ Jina (1024D)
```
- **Benefit**: Can share collections if needed
- **Reality**: Better to keep separate for clean model tracking

## Implementation Details

### Collection Creation
```python
# Automatic on first use
store = MultiModelVectorStore("BAAI/bge-large-en-v1.5")
# Creates: munirag_baai_bge_large_en_v1_5 if not exists
```

### Dimension Validation
```python
# On document addition
if len(embedding) != self.dimension:
    raise ValueError(f"Expected {self.dimension}D, got {len(embedding)}D")

# On search
if len(query_embedding) != self.dimension:
    raise ValueError(f"Query is {len(query_embedding)}D, collection is {self.dimension}D")
```

### Migration Between Models
```python
# Different dimensions = re-embedding required
migrate_collection(
    source_model="BAAI/bge-large-en-v1.5",  # 1024D
    target_model="thenlper/gte-large",      # 768D
    batch_size=1000  # GPU-optimized batching
)
```

## Benefits of This Approach

1. **Zero Dimension Conflicts**: Physical separation prevents all mismatch errors
2. **Model Flexibility**: Add new models without affecting existing data
3. **Clean Rollback**: Keep old model data while testing new ones
4. **Performance Tracking**: Compare models side-by-side
5. **Future Proof**: New models just create new collections

## When to Use Each Pattern

### Single Model (Default)
- Production deployments
- Consistent performance requirements
- Simplest architecture

### Multi-Model with Migration
- Upgrading to better models
- A/B testing different models
- Gradual transition periods

### Cross-Model Search
- Research and experimentation
- Model performance comparison
- Special accuracy requirements

## Current Implementation Status

✅ **Implemented**:
- Multi-model vector store with automatic collection management
- Dimension validation and error handling
- Model migration with re-embedding support
- Backward compatibility wrapper

🔄 **In Progress**:
- Streamlit UI for model selection
- Cross-model search interface

📋 **Planned**:
- Model performance analytics
- Automatic model recommendation
- Hybrid scoring algorithms

## Best Practices

1. **Stick to One Model**: Unless you have specific needs
2. **Plan Migrations**: Re-embedding is GPU-intensive
3. **Monitor Collection Sizes**: Each model stores data separately
4. **Document Model Choices**: Track why you chose specific models
5. **Test Before Migrating**: Verify new model performance first

## Example Workflows

### Production Deployment
```python
# .env
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5  # Fast, accurate, GPU-optimized

# All operations use BGE automatically
```

### Model Experimentation
```python
# Test different models
models = ["BAAI/bge-large-en-v1.5", "thenlper/gte-large", "intfloat/e5-large-v2"]
for model in models:
    store = MultiModelVectorStore(model)
    # Ingest same documents
    # Compare retrieval quality
```

### Planned Model Upgrade
```python
# 1. Keep using current model
# 2. Ingest new documents to new model
# 3. A/B test quality
# 4. Migrate if better
# 5. Deprecate old collection
```

## Troubleshooting

### "Dimension mismatch" Error
- Check query and document models match
- Verify collection has correct dimension
- Use migration tool if switching models

### Empty Results
- Ensure documents were ingested to correct collection
- Check embedder and retriever use same model
- Verify collection has documents

### Performance Issues
- Batch size affects GPU utilization
- Re-embedding is slower than direct copy
- Consider keeping compatible models separate

## Summary

The hybrid approach provides **maximum flexibility** with **zero dimension conflicts** by using **isolated collections per model**. This is slightly less storage-efficient than a single collection, but the benefits of clean model separation and conflict prevention far outweigh the minimal storage overhead.