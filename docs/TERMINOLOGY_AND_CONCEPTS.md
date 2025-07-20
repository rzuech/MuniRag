# MuniRAG Terminology and Concepts Guide

*Created: 2025-07-20*

This document clarifies the key concepts and terminology used throughout the MuniRAG codebase, especially focusing on the document processing pipeline.

## 🔤 Core Concepts

### 1. Text Splitting (Character-based Chunking)
- **What**: Breaks large documents into smaller, manageable pieces
- **Why**: LLMs have token limits; can't process entire 50-page PDFs at once
- **How**: Splits text by character count with overlap between chunks
- **When**: BEFORE embedding, during document ingestion
- **Controlled by**: `MAX_CHUNK_TOKENS` setting (default: 500 tokens ≈ 2000 characters)

**Example**:
```
Original: "The city council meets every Tuesday at 7 PM. Meetings are open to the public. Citizens may speak during the comment period."

Chunk 1: "The city council meets every Tuesday at 7 PM. Meetings are open to the public."
Chunk 2: "Meetings are open to the public. Citizens may speak during the comment period."
         ↑ Notice the overlap for context preservation
```

### 2. Semantic Chunking (Meaning-based Chunking)
- **What**: Groups text by topic/meaning changes rather than fixed character counts
- **Why**: Keeps related content together (e.g., entire policy section)
- **How**: Uses embeddings to detect topic boundaries
- **When**: Alternative to simple text splitting
- **Controlled by**: `SEMANTIC_CHUNKING` setting (true/false)

**Comparison**:
- Simple splitting: Every 1000 characters
- Semantic chunking: Breaks at topic changes (could be 500 or 2000 characters)

### 3. Embeddings (Vector Representations)
- **What**: Converts text chunks into numerical vectors (arrays of numbers)
- **Why**: Computers can't understand text but can compare numbers
- **How**: ML models (like BGE) transform text → vectors
- **Output**: Array of floats (e.g., [0.123, -0.456, 0.789, ...])

### 4. Embedding Dimensions
- **What**: The size/length of embedding vectors (how many numbers)
- **BGE Model**: 1024 dimensions (1024 numbers per chunk)
- **Jina Model**: 1024 dimensions
- **GTE Model**: 768 dimensions
- **Critical**: Dimensions MUST match between storage and retrieval!

**Example Vector**:
```python
# BGE embedding (1024 dimensions)
[0.0234, -0.1455, 0.7823, ..., 0.0012]  # 1024 numbers total

# GTE embedding (768 dimensions)  
[0.0412, -0.2341, 0.6234, ..., 0.0089]  # 768 numbers total
```

### 5. Vector Store (Qdrant)
- **What**: Database optimized for storing and searching vectors
- **Why**: Regular databases can't efficiently search high-dimensional vectors
- **How**: Uses cosine similarity to find similar vectors
- **Collections**: Like tables, each stores vectors of same dimension

## 📊 Data Flow Pipeline

```
1. PDF Upload
   ↓
2. Text Extraction (PyMuPDF)
   ↓
3. Text Splitting (Character-based, ~1000 chars per chunk)
   ↓
4. [Optional] Semantic Re-chunking (Group by meaning)
   ↓
5. Embedding Generation (Text → 1024D vectors)
   ↓
6. Vector Storage (Qdrant collection)
   ↓
7. Ready for Search!
```

## 🔍 Search Process

```
1. User Query: "What are the parking regulations?"
   ↓
2. Query Embedding (Same model as documents!)
   ↓
3. Vector Search (Find similar vectors in Qdrant)
   ↓
4. Retrieve Chunks (Get original text of matches)
   ↓
5. Send to LLM (With context for answer generation)
```

## ⚠️ Common Confusion Points

### Chunks vs Embeddings
- **Chunk**: Piece of text (human readable)
- **Embedding**: Vector representation of chunk (numbers)
- One chunk → One embedding

### Dimensions vs Tokens
- **Dimensions**: Size of embedding vector (e.g., 1024)
- **Tokens**: Word pieces in text (~4 chars = 1 token)
- These are DIFFERENT measurements!

### Character Count vs Token Count
- **Characters**: Letters, spaces, punctuation
- **Tokens**: How LLMs break up text
- Rough conversion: 4 characters ≈ 1 token

### Text Splitter vs Semantic Chunker
- **Text Splitter**: Simple, splits every N characters
- **Semantic Chunker**: Smart, splits at topic boundaries
- Both produce chunks, but different strategies

## 🔧 Configuration Impact

| Setting | What it Controls | Impact |
|---------|-----------------|---------|
| `MAX_CHUNK_TOKENS` | Size of text chunks | Larger = more context but fewer chunks |
| `SEMANTIC_CHUNKING` | Use smart chunking | Better context but slower |
| `EMBEDDING_MODEL` | Which ML model | Different dimensions & quality |
| `TOP_K` | Results to retrieve | More = better recall but more noise |

## 💡 Best Practices

1. **Consistency**: Always use same embedding model for storage & retrieval
2. **Chunk Size**: 500-1000 tokens works well for most documents
3. **Overlap**: 10-20% overlap preserves context across boundaries
4. **Semantic Chunking**: Enable for complex documents with clear sections

## 🐛 Common Errors

### "Dimension Mismatch"
- **Cause**: Trying to search 768D vector in 1024D collection
- **Fix**: Use same embedding model throughout

### "No Results Found"
- **Cause**: Often dimension mismatch or empty collection
- **Fix**: Check logs for embedding errors

### "Out of Memory"
- **Cause**: Batch size too large for GPU
- **Fix**: Reduce batch size in embedder

---

This document will be updated as the project evolves. For implementation details, see the corresponding source files.