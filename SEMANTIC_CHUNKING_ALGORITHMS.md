# Semantic Chunking Algorithms - Future Experiments

## Current Implementation
The system currently uses **cosine similarity** with sentence embeddings (threshold: 0.7) to group semantically related sentences.

## Alternative Algorithms to Experiment With

### 1. **Sliding Window with Overlap Score**
- Uses overlapping windows to find natural break points
- Scores each position based on semantic coherence before/after
- **Pros**: More context-aware boundaries
- **Cons**: Computationally expensive

### 2. **Topic Modeling Based Chunking**
- Uses LDA or BERTopic to identify topic shifts
- Creates chunks when topic changes significantly
- **Pros**: Great for documents with clear topic sections
- **Cons**: May miss subtle transitions

### 3. **Hierarchical Clustering**
- Groups sentences into hierarchical clusters
- Cuts dendrogram at optimal height for chunk size
- **Pros**: Naturally handles nested topics
- **Cons**: Memory intensive for large documents

### 4. **Graph-Based Segmentation**
- Builds sentence similarity graph
- Uses community detection algorithms
- **Pros**: Captures complex relationships
- **Cons**: Slower than current approach

### 5. **Neural Text Segmentation**
- Fine-tuned models (like TextTiling-BERT)
- Learns document structure patterns
- **Pros**: Most accurate for specific domains
- **Cons**: Requires training data

### 6. **Markdown/Structure-Aware Chunking**
- Respects document formatting (headers, lists)
- Never splits logical sections
- **Pros**: Preserves document intent
- **Cons**: Only works for structured documents

### 7. **Dynamic Programming Optimization**
- Optimizes chunk boundaries for maximum coherence
- Uses DP to find globally optimal splits
- **Pros**: Theoretically optimal chunks
- **Cons**: O(n²) complexity

### 8. **Ensemble Chunking**
- Combines multiple algorithms
- Votes on best boundaries
- **Pros**: More robust
- **Cons**: Slower, complex to tune

## Implementation Considerations

### Performance Metrics
- **Coherence Score**: Avg similarity within chunks
- **Distinction Score**: Avg dissimilarity between chunks
- **Retrieval Quality**: Precision/Recall on test queries

### Tunable Parameters
- Similarity thresholds
- Min/max chunk sizes
- Overlap allowance
- Algorithm weights (for ensemble)

### A/B Testing Framework
```python
def compare_chunking_algorithms(pdf_path, algorithms):
    results = {}
    for algo_name, algo_func in algorithms.items():
        chunks = algo_func(pdf_path)
        results[algo_name] = {
            "num_chunks": len(chunks),
            "avg_size": np.mean([len(c) for c in chunks]),
            "coherence": calculate_coherence(chunks),
            "time": measure_time(algo_func, pdf_path)
        }
    return results
```

## Recommended Experiments

1. **Short Term**: Test markdown-aware chunking for structured docs
2. **Medium Term**: Implement sliding window algorithm
3. **Long Term**: Train domain-specific segmentation model

## Integration Points
- Keep current algorithm as baseline
- Add `CHUNKING_ALGORITHM` config option
- Allow per-document algorithm selection
- Maintain backward compatibility