# MuniRAG Master Priority List - STABILITY FIRST

## 🔴 SESSION RESTORE POINT (2025-07-15 00:23)

### Current Issue: Documents Not Being Retrieved
**Status**: PDF uploads successfully but search returns 0 documents
**Error**: No error - Qdrant returns 200 OK but finds 0 matching documents

### What We Did This Session:
1. ✅ Fixed GPU progress bar slowdown (33x improvement)
2. ✅ Connected parallel PDF processor (10-50x faster)
3. ✅ Added pytesseract and Pillow dependencies
4. ❌ Documents being stored but not retrieved

### Debug Info:
- Collection `munirag_baai_bge_large_en_v1.5` exists (was corrupted, we cleared it)
- Old collection `munirag_docs` has 31,098 documents
- Search returns 200 OK but 0 results
- Dimensions match (1024)

### Next Steps to Debug:
1. **Check if documents are actually being stored after upload**:
   ```bash
   docker-compose exec munirag python -c "
   from qdrant_client import QdrantClient
   client = QdrantClient(host='qdrant', port=6333)
   count = client.count('munirag_baai_bge_large_en_v1.5').count
   print(f'Documents in BGE collection: {count}')
   "
   ```

2. **Verify storage location mismatch**:
   - `ingest_parallel.py` uses `VectorStore()` (stores in 'munirag_docs')
   - `retriever.py` uses `MultiModelVectorStore()` (searches in 'munirag_baai_bge_large_en_v1.5')
   - **THIS IS THE PROBLEM!**

3. **Quick Fix**: Update `ingest_parallel.py` to use `MultiModelVectorStore` instead of `VectorStore`

### Windows Line Ending Issues:
Scripts had `\r` errors - need to run: `dos2unix *.sh` or save with LF endings

---

## 🚨 CRITICAL: Current State Summary
- **PDF Processing**: Connected but store/retrieve mismatch
- **GPU Performance**: Fixed! Progress bars disabled
- **Major Components**: Parallel processor connected, but vector store mismatch

## Priority 1: IMMEDIATE STABILITY FIXES (Today)

### 1.1 Fix GPU Progress Bar Slowdown ⚡
**Impact**: 33x performance improvement for Instructor/Jina models  
**Effort**: 2 minutes  
**File**: `src/embedder.py` lines 236, 243  
```python
# Change from:
show_progress_bar=True
# To:
show_progress_bar=False
```

### 1.2 Connect Existing Parallel PDF Processor 🔌
**Impact**: 10-50x faster PDF processing  
**Effort**: 30 minutes  
**Current**: App uses slow PyPDF2 via ingest.py  
**Fix**: Import and use existing pdf_parallel_processor.py  
```python
# In app.py, replace:
from src.ingest_optimized import ingest_pdfs_optimized
# With:
from src.pdf_parallel_processor import ParallelPDFProcessor
```

### 1.3 Add Missing Dependencies 📦
**Impact**: Enable OCR and other features  
**Effort**: 5 minutes  
```bash
# Add to requirements.txt:
pytesseract>=0.3.10  # For OCR support
Pillow>=10.0.0      # For image processing
PyMuPDF>=1.23.0     # Already there but verify
```

### 1.4 Enable Semantic Chunking ✂️
**Impact**: Better retrieval quality  
**Effort**: 1 minute  
```bash
# In .env:
SEMANTIC_CHUNKING=true
PDF_WORKERS=4  # Or number of CPU cores
```

## Priority 2: PERFORMANCE OPTIMIZATION (This Week)

### 2.1 Model-Specific Batch Size Fix
**Current**: Some models use suboptimal batch sizes  
**Fix**: Review and optimize batch sizes per model in embedder.py

### 2.2 Memory Management
**Issue**: Large PDFs might OOM  
**Fix**: Implement streaming/chunked processing for huge files

### 2.3 Collection Metadata
**Issue**: Qdrant doesn't track embedding model used  
**Fix**: Add model name to collection metadata
```python
collection_config = {
    "metadata": {
        "embedding_model": settings.EMBEDDING_MODEL,
        "created_at": datetime.now()
    }
}
```

## Priority 3: RELIABILITY IMPROVEMENTS (Next Week)

### 3.1 Error Recovery
- Add retry logic for embedding failures
- Graceful degradation when models unavailable
- Better error messages for users

### 3.2 Configuration Validation
- Validate settings on startup
- Check model compatibility
- Verify GPU availability matches config

### 3.3 Health Monitoring
- Add /health endpoint
- Monitor Qdrant connectivity
- Track GPU memory usage

## Priority 4: FUTURE FEATURES (Later)

### 4.1 Ensemble Embeddings (Your Request)
- Implement 3-model ensemble
- Parallel query embedding
- Result fusion strategies

### 4.2 Advanced PDF Features
- Table extraction
- Image captioning
- Multi-column layout handling

### 4.3 API Enhancements
- Authentication implementation
- Rate limiting enforcement
- Caching layer

## 🧪 Test Commands After Fixes

### Step 1: Apply Critical Fixes
```bash
# 1. Fix progress bar in embedder.py (manual edit required)
# 2. Restart services
docker-compose down && docker-compose up --build

# 3. Verify GPU fix
docker-compose exec munirag python -c "
from src.embedder import EmbeddingModel
model = EmbeddingModel('hkunlp/instructor-xl')
import time
texts = ['test'] * 1000
start = time.time()
embeddings = model.embed_documents(texts)
print(f'Speed: {len(texts)/(time.time()-start):.0f} texts/sec')
print('Expected: ~1500-3500 texts/sec')
"
```

### Step 2: Test PDF Processing
```bash
# Create a test script to compare processors
docker-compose exec munirag python -c "
import time
from pypdf import PdfReader

# Test current slow method
start = time.time()
reader = PdfReader('/path/to/test.pdf')
for page in reader.pages[:10]:
    text = page.extract_text()
print(f'PyPDF2: {time.time()-start:.2f}s for 10 pages')

# Test fast method (after connecting parallel processor)
from src.pdf_parallel_processor import ParallelPDFProcessor
processor = ParallelPDFProcessor()
start = time.time()
chunks = processor.process('/path/to/test.pdf', max_pages=10)
print(f'Parallel: {time.time()-start:.2f}s for 10 pages')
print(f'Chunks created: {len(chunks)}')
"
```

### Step 3: Verify Semantic Chunking
```bash
docker-compose exec munirag python -c "
from src.config import settings
print(f'Semantic chunking enabled: {settings.SEMANTIC_CHUNKING}')
print(f'PDF workers: {settings.PDF_WORKERS}')
print(f'OCR enabled: {settings.USE_OCR}')
"
```

### Step 4: Full Integration Test
```bash
# Upload a test PDF through Streamlit
# Monitor logs for:
# - "Parallel PDF Processor initialized"
# - "Using semantic chunking"
# - Embedding speeds > 1000/sec
# - Total processing time < 1 minute for 100 pages
```

## 📊 Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| 100-page PDF | 10+ min | <1 min | Time the upload |
| Instructor embedding | ~117/sec | >1500/sec | Run test script |
| CPU utilization | 10-20% | >80% | Monitor during PDF processing |
| Semantic chunks | Disabled | Enabled | Check config & logs |

## 🎯 Quick Win Summary

**Just connecting the existing parallel processor will give you:**
- 10-50x faster PDF processing
- Semantic chunking for better quality
- OCR support for scanned documents
- Progress tracking that actually works

**Total effort: ~1 hour of work for massive improvements!**