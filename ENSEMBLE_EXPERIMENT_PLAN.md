# Ensemble Embedding Experiment Plan

## Current Understanding

### Chunking Strategy
**IMPORTANT**: Based on code review, all models currently use the SAME chunking strategy:
- Text is chunked BEFORE embedding (in `pdf_processor.py`)
- All models receive the same pre-chunked text
- Chunking is based on token count, not model-specific

### Performance Considerations
1. **Vector DB Load**: Yes, 3x storage (each model gets its own collection)
2. **Query Overhead**: Minimal - 3x50ms = 150ms total (can be parallelized)
3. **Memory**: Need ~6GB GPU memory for 3 models loaded simultaneously

## Ensemble Architecture Design

### Option 1: Simple Ensemble (Recommended Start)
```python
# All 3 models embed the same chunks
models = ["BAAI/bge-large-en-v1.5", "jinaai/jina-embeddings-v3", "intfloat/e5-large-v2"]
results = []

# Parallel query embedding
for model in models:
    embedding = embed_with_model(model, query)
    docs = search_collection(model_collection, embedding)
    results.extend(docs)

# Merge and rank
final_results = merge_by_score(results)
```

### Option 2: Model-Specific Chunking (Future Enhancement)
```python
# Different chunking strategies per model
chunking_strategies = {
    "bge": {"method": "semantic", "size": 512},
    "jina": {"method": "sliding_window", "size": 1024},  # Jina handles long context
    "e5": {"method": "sentence", "size": 768}
}
```

## Implementation Steps

1. **Modify `rag_pipeline.py`**:
   - Add ensemble query method
   - Implement parallel embedding
   - Create result fusion algorithm

2. **Extend `vector_store.py`**:
   - Support multi-collection queries
   - Add ensemble search method

3. **Update configuration**:
   - Add `ENSEMBLE_MODELS` list
   - Add `ENSEMBLE_FUSION_METHOD` (max, average, weighted)

## Expected Results

### Performance Impact
- **Ingestion**: 3x slower (one-time cost)
- **Query**: ~150ms vs 50ms (3x but parallelizable)
- **Storage**: 3x increase
- **Memory**: ~6GB for 3 models

### Quality Benefits
- **Robustness**: Less sensitive to model weaknesses
- **Coverage**: Different models capture different aspects
- **Flexibility**: Can weight models based on document type

## Experiment Metrics to Track
1. **Retrieval Quality**: Precision@K, Recall@K
2. **Response Time**: Query latency
3. **Resource Usage**: GPU memory, CPU usage
4. **Result Diversity**: How different are the results from each model?

## Quick Test Command
```bash
# After implementation, test ensemble query
docker-compose exec munirag python -c "
from src.rag_pipeline import RAGPipeline
pipeline = RAGPipeline(ensemble_mode=True)
results = pipeline.ensemble_query('What is the municipal budget?')
print(f'Ensemble results: {len(results)} documents')
"
```

## Next Session Action Items
1. Implement parallel query embedding
2. Create result fusion algorithm
3. Add configuration for ensemble mode
4. Test performance impact
5. Compare quality vs single model

## Notes for Implementation
- Start with same chunking for all models (simpler)
- Use asyncio for parallel embedding
- Consider weighted voting based on model strengths
- Monitor GPU memory carefully with 3 models loaded