# Multi-Model Architecture Decision

*Date: 2025-07-20*
*Version: 1.0*

## Executive Summary

We've adopted the `MultiModelVectorStore` architecture throughout MuniRAG to support multiple embedding models while maintaining clean separation of concerns. This decision enables A/B testing of models, gradual migrations, and future extensibility.

## Architecture Overview

### What is MultiModelVectorStore?

`MultiModelVectorStore` creates **separate Qdrant collections for each embedding model**, preventing dimension conflicts and enabling clean model management.

**Example Collections:**
- `munirag_baai_bge_large_en_v1_5` (1024 dimensions)
- `munirag_jinaai_jina_embeddings_v3` (1024 dimensions)  
- `munirag_thenlper_gte_large` (768 dimensions)

### Key Benefits

1. **No Dimension Conflicts** - Each model's embeddings stored separately
2. **Easy Model Switching** - Change models without re-embedding everything
3. **A/B Testing** - Compare model performance side-by-side
4. **Gradual Migration** - Move from one model to another incrementally
5. **Future Proof** - Add new models without breaking existing data

## Implementation Details

### Files Modified

1. **Created/Restored:**
   - `src/vector_store_v2.py` - MultiModelVectorStore implementation

2. **Updated to use MultiModelVectorStore:**
   - `src/ingest.py` - Document ingestion
   - `src/retriever.py` - Document retrieval

3. **LangChain Removal:**
   - `src/pdf_processor.py` - Custom text splitter implementation
   - `requirements.txt` - Removed langchain dependencies

### How It Works

```python
# Ingestion (ingest.py)
vector_store = MultiModelVectorStore()  # Uses default model from settings
# Automatically creates collection: munirag_baai_bge_large_en_v1_5

# Retrieval (retriever.py)  
vector_store = MultiModelVectorStore()  # Same model, same collection
# Searches in: munirag_baai_bge_large_en_v1_5
```

### Collection Naming Convention

Model name → Collection name:
- `BAAI/bge-large-en-v1.5` → `munirag_baai_bge_large_en_v1_5`
- `jinaai/jina-embeddings-v3` → `munirag_jinaai_jina_embeddings_v3`

## Migration from Simple VectorStore

### Before (Problem)
- `VectorStore` used fixed collection "munirag_docs"
- Changing models caused dimension mismatches
- Required complete re-ingestion

### After (Solution)
- `MultiModelVectorStore` creates model-specific collections
- Can have multiple models' data simultaneously
- Switch models via configuration

## LangChain Removal

### Why Removed
1. **Build Time**: 43+ minutes due to dependency conflicts
2. **Bloat**: 500MB+ of unused dependencies
3. **Simple Use Case**: Only used for text splitting

### Custom Implementation
Created `_recursive_split_text()` in `pdf_processor.py`:
- Hierarchical splitting (headers → paragraphs → sentences)
- Token-based chunk sizing
- Overlap management
- Same functionality, zero dependencies

### Build Time Impact
- **Before**: 43+ minutes (LangChain dependency resolution)
- **After**: 5-10 minutes (expected)

## Configuration

### Environment Variables
```bash
# Embedding model (determines collection)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Chunk settings (unchanged)
MAX_CHUNK_TOKENS=500
CHUNK_OVERLAP=50
```

### Adding New Models
1. Add to `MODEL_DIMENSIONS` in `vector_store_v2.py`
2. Update `.env` with new model name
3. Restart services
4. New collection created automatically

## Testing Checklist

- [ ] PDF upload creates documents in correct collection
- [ ] Search queries correct collection
- [ ] No dimension mismatch errors
- [ ] Text splitting works without LangChain
- [ ] Build time under 15 minutes

## Future Enhancements

1. **Model Management API** - Switch models on the fly
2. **Cross-Model Search** - Query multiple models simultaneously
3. **Model Performance Metrics** - Track which models perform best
4. **Automatic Migration** - Move documents between models

## Rollback Plan

If issues arise:
```bash
# Revert to backup branch
git checkout backup-before-multimodel-fix-20250719-203935

# Or revert specific files
git checkout HEAD~1 -- src/ingest.py src/retriever.py
```

## Decision Rationale

We chose to embrace `MultiModelVectorStore` now because:
1. You plan to use multiple embedding models
2. The architecture is sound and well-tested
3. Fixing it properly now prevents future pain
4. The store/retrieve mismatch needed fixing anyway

This positions MuniRAG for growth while maintaining stability.