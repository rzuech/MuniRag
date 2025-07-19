# Ensemble Architecture Research & Scaling Analysis

## Critical Findings

### 1. Jina Licensing Concerns ⚠️
After reviewing Jina's licensing:
- **Jina embeddings v3 requires API key** for commercial use
- The model weights are not fully open (unlike BGE/E5)
- Terms: https://jina.ai/embeddings/ - "Free for non-commercial use"
- **Recommendation**: Use BGE + E5 + GTE for truly open ensemble

### 2. Inference Speed Scaling Analysis 🚨

#### Current Understanding (CORRECTED)
The "100 documents" in research refers to **embedding generation speed**, NOT search speed.

**Two Different Operations:**
1. **Embedding Generation** (one-time, during upload)
   - Speed: ~3,000 docs/sec (BGE on GPU)
   - Scales linearly: 2x docs = 2x time
   - Not a concern for queries (only embed the question)

2. **Vector Search** (every query) - THE REAL CONCERN
   - Speed depends on collection size
   - Does NOT scale linearly
   - This is where ensemble multiplies problems

#### Vector Search Scaling (Qdrant)

| Collection Size | Single Model | 3-Model Ensemble |
|----------------|--------------|------------------|
| 1K docs | ~5ms | ~15ms |
| 10K docs | ~15ms | ~45ms |
| 100K docs | ~50ms | ~150ms |
| 1M docs | ~200ms | ~600ms |
| 10M docs | ~1s | ~3s |

**Why it scales this way:**
- Qdrant uses HNSW (Hierarchical Navigable Small World) index
- Search complexity: O(log N) where N = number of vectors
- Memory usage: ~100-200 bytes per vector + overhead

### 3. Ensemble Architecture Options

#### Option A: Separate Collections (Current Design)
```
Query → Embed with 3 models → Search 3 collections → Merge
```
**Pros:**
- Clean separation
- Can optimize each collection independently
- Easy to add/remove models

**Cons:**
- 3x search time
- 3x memory usage
- Complex result merging

#### Option B: Unified Multi-Vector Storage
```
Document → 3 embeddings → Store as single entry with 3 vectors
```
**Pros:**
- Single search operation
- Atomic updates
- Simpler architecture

**Cons:**
- Not supported by Qdrant directly
- Would need custom implementation
- Less flexible

#### Option C: Primary + Secondary Models
```
Query → Primary model → If low confidence → Secondary models
```
**Pros:**
- Fast common case (single model)
- Ensemble only when needed
- Adaptive performance

**Cons:**
- Complexity in confidence scoring
- May miss relevant results

### 4. Performance Optimizations for Scale

#### 4.1 Index Optimization
```python
# Qdrant collection config for large scale
{
    "vectors": {
        "size": 1024,
        "distance": "Cosine"
    },
    "optimizers_config": {
        "indexing_threshold": 20000,  # Build index after 20K vectors
        "default_segment_number": 4,   # Parallel segments
        "max_segment_size": 200000     # Segment size for performance
    },
    "hnsw_config": {
        "m": 16,                # Connections per node
        "ef_construct": 200,     # Build-time accuracy
        "ef": 100               # Search-time accuracy/speed trade-off
    }
}
```

#### 4.2 Caching Strategy
```python
# Cache frequent queries
query_cache = {}  # Redis in production

def cached_ensemble_search(query):
    cache_key = hashlib.md5(query.encode()).hexdigest()
    if cache_key in query_cache:
        return query_cache[cache_key]
    
    results = ensemble_search(query)
    query_cache[cache_key] = results
    return results
```

#### 4.3 Async Parallel Search
```python
async def ensemble_search_async(query):
    # Parallel execution instead of sequential
    tasks = [
        search_bge_async(query),
        search_e5_async(query),
        search_gte_async(query)
    ]
    results = await asyncio.gather(*tasks)
    return merge_results(results)
```

### 5. Recommended Architecture for Scale

#### For <100K documents:
- **Use ensemble with 3 open models** (BGE, E5, GTE-large)
- Separate collections with parallel search
- Accept 3x search time (still <200ms)

#### For 100K-1M documents:
- **Primary model + fallback**
- BGE as primary (fastest)
- E5 as fallback for low-confidence results
- Cache frequent queries

#### For >1M documents:
- **Single optimized model**
- Consider fine-tuning BGE on your data
- Use Qdrant's filtering to reduce search space
- Implement sharding by document type/date

### 6. Storage Calculation

For 1 million chunks with ensemble:

| Component | Single Model | 3-Model Ensemble |
|-----------|--------------|------------------|
| Vectors | 4GB (1M × 1024 × 4 bytes) | 12GB |
| Metadata | ~1GB | ~1GB (shared) |
| Indexes | ~2GB | ~6GB |
| **Total** | **~7GB** | **~19GB** |

### 7. Decision Framework

Choose ensemble if:
- ✅ <100K documents
- ✅ Accuracy is critical
- ✅ 200ms query latency acceptable
- ✅ Have 20GB+ RAM available

Choose single model if:
- ✅ >1M documents
- ✅ Need <50ms query latency
- ✅ Resource constrained
- ✅ Can fine-tune on your data

### 8. Implementation Recommendations

1. **Start with single model** (BGE)
2. **Measure baseline** performance
3. **A/B test ensemble** on subset
4. **Monitor query latency** P50, P95, P99
5. **Scale decision** based on real metrics

### 9. Alternative: Reranking Approach

Instead of ensemble embeddings:
```
Query → Single embedding → Retrieve 20 candidates → Rerank with 3 models → Top 5
```

This gives ensemble benefits with single-model search speed!

## Conclusion

The ensemble approach is architecturally sound for municipal use cases (typically <100K documents) but requires careful consideration of:
- Licensing (avoid Jina for commercial)
- Search latency at scale
- Resource requirements
- Actual accuracy improvements

Start simple, measure everything, scale based on data.