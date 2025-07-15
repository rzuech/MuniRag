# Experimental Code Archive

This directory contains experimental implementations that were explored but not ultimately used in production.

## Files

### embedder_universal.py
- **Purpose**: Universal embedder supporting multiple models
- **Status**: Failed with Jina due to SentenceTransformers incompatibility
- **Issue**: Jina's custom XLMRobertaLoRA architecture not fully supported
- **Learning**: Keep it simple - model-specific handling in main embedder.py works better

### embedder_v2.py  
- **Purpose**: Second iteration of embedder design
- **Status**: Superseded by current embedder.py
- **Notes**: Contains early GPU optimization attempts

### parallel_embedder.py
- **Purpose**: First attempt at parallel embedding
- **Status**: Had thread safety issues in Streamlit
- **Learning**: Need proper thread-safe initialization

### parallel_embedder_v2.py
- **Purpose**: Thread-safe parallel embedder
- **Status**: Works but complex, disabled due to Jina GPU issues
- **Notes**: Could be re-enabled for CPU-only parallel processing

## Key Learnings

1. **Simplicity wins**: Complex abstractions made debugging harder
2. **Jina's architecture**: Fundamentally incompatible with GPU optimization in SentenceTransformers
3. **Thread safety**: Critical for Streamlit - must initialize models in main thread
4. **Fallback complexity**: Too many fallback paths made the code unpredictable

## Preserved for Future Reference

These implementations contain valuable patterns that might be useful:
- Thread-safe model initialization (parallel_embedder_v2.py)
- Multi-process CPU optimization (parallel_embedder.py)
- Model abstraction patterns (embedder_universal.py)

To re-enable any of these, copy back to src/ and update imports in ingest.py.