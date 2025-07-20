# MuniRAG Master Priority List v2.0
*Last Updated: 2025-07-19*

## 🚨 Project Status Overview

### Current State
- **Version**: 0.90 (clean commit history achieved)
- **Core Functionality**: Working but with critical bugs
- **Performance**: GPU fixed, PDF processing connected but not integrated
- **Architecture**: Mix of v1 and v2 components causing confusion

### Critical Issues
1. **Store/Retrieve Mismatch**: Documents stored in 'munirag_docs' but searched in 'munirag_baai_bge_large_en_v1.5'
2. **Multiple Redundant Files**: 4 ingest versions, 2 app versions, 2 docker files
3. **OCR Not Working**: Dependencies added but not configured
4. **Semantic Chunking**: Implemented but disabled

---

## 📋 Prioritized Task List

### Priority 0: CRITICAL BUG FIXES (Today - 2 hours)

#### 0.1 Fix Store/Retrieve Mismatch 🚨
**Status**: Not started  
**Impact**: Documents not retrievable after upload  
**Fix**: Update `src/ingest_parallel.py` line 30-31
```python
# Replace:
from src.vector_store import VectorStore
vector_store = VectorStore()

# With:
from src.vector_store_v2 import MultiModelVectorStore
vector_store = MultiModelVectorStore()
```

#### 0.2 Enable Semantic Chunking
**Status**: Code exists, config disabled  
**Impact**: Better retrieval quality  
**Fix**: Already in .env.example, need to ensure it's in .env
```bash
SEMANTIC_CHUNKING=true
PDF_WORKERS=4
```

---

### Priority 1: CODE CLEANUP & CONSOLIDATION (Tomorrow - 4 hours)

#### 1.1 Consolidate Ingestion Files
**Current State**: 4 versions causing confusion
- `ingest_backup.py` - Original slow version
- `ingest.py` - Current but uses slow PyPDF2
- `ingest_optimized.py` - GPU batching but still PyPDF2
- `ingest_parallel.py` - Best version with PyMuPDF

**Action Plan**:
1. Rename `ingest_parallel.py` → `ingest.py`
2. Archive others to `src/deprecated/`
3. Update all imports

#### 1.2 Consolidate App Files
**Current State**: 2 versions
- `app.py` - Current working version
- `app_enhanced.py` - Has API endpoints

**Action Plan**:
1. Merge API features from enhanced into main
2. Archive `app_enhanced.py`

#### 1.3 Clean Docker Files
**Current State**: 2 Dockerfiles
- `Dockerfile` - Current working version
- `Dockerfile.optimized` - Has better caching

**Action Plan**:
1. Merge optimizations into main Dockerfile
2. Remove `Dockerfile.optimized`

#### 1.4 Consolidate Requirements
**Current State**: 5 requirement files
**Action Plan**:
1. Keep only `requirements.txt` for simplicity
2. Move others to `docs/deployment/`

#### 1.5 Clean Vector Store Implementations
**Current State**: 
- `vector_store.py` - Simple, uses single collection "munirag_docs"
- `vector_store_v2.py` - Complex, creates separate collection per embedding model
  - Example: "munirag_baai_bge_large_en_v1_5", "munirag_jinaai_jina_embeddings_v3"
  - Benefit: Can switch models without dimension conflicts
  - Drawback: More complex, harder to debug, more storage

**Decision**: Keep MultiModelVectorStore for multi-model support
**Action Plan**: ✅ COMPLETED
1. Fixed store/retrieve mismatch - both now use MultiModelVectorStore
2. Kept `vector_store_v2.py` for clean multi-model architecture
3. Removed LangChain to fix 43-minute build times
4. Fixed UploadedFile error in PDF processing

---

### Priority 2: MUNICIPALITY INTEGRATION (This Week - 16 hours)

#### 2.1 JavaScript Widget for Websites 🌐
**Status**: Not started (was mentioned in README)  
**Impact**: Easy integration for municipality websites  
**Tasks**:
1. Create embeddable widget (`munirag-widget.js`)
2. Implement chat interface in JavaScript
3. Add customization options (colors, size, position)
4. Create demo page with integration examples
5. Add CORS configuration for municipalities

**Reference**: README.md mentions this as 1-2 day task

#### 2.2 API Endpoints for External Integration
**Status**: Partially exists in `app_enhanced.py`  
**Tasks**:
1. Merge API endpoints from app_enhanced.py
2. Add authentication (Bearer tokens/API keys)
3. Implement rate limiting
4. Add CORS for specific domains
5. Create OpenAPI/Swagger documentation
6. Add /health endpoint

#### 2.3 Front-end Integration Guide
**Tasks**:
1. Document iframe embedding method
2. Create JavaScript SDK
3. Add WordPress plugin example
4. Create React/Vue components
5. Security best practices guide

### Priority 3: FEATURE COMPLETION (Following Week - 8 hours)

#### 3.1 Complete OCR Integration
**Status**: Dependencies added, code exists in `pdf_parallel_processor.py`  
**Tasks**:
1. Install Tesseract in Docker image
2. Test OCR detection logic
3. Add OCR language packs
4. Performance benchmarking

**Reference**: `/docs/OCR_IMAGE_HANDLING.md`

#### 3.2 Implement Advanced Chunking Options
**Status**: Basic semantic chunking implemented  
**Tasks**:
1. Add markdown-aware chunking
2. Implement chunking algorithm selector
3. A/B testing framework
4. Performance comparison

**Reference**: `/docs/SEMANTIC_CHUNKING_ALGORITHMS.md`

#### 3.3 Add Document Management UI
**Tasks**:
1. List uploaded documents
2. Delete individual documents
3. Show document metadata
4. Search within specific documents

---

### Priority 4: ARCHITECTURE IMPROVEMENTS (Next Week)

#### 4.1 Implement Proper Error Handling
- Add retry logic for embeddings and LLM calls
- Better error messages with context
- Graceful degradation when services unavailable
- User-friendly error display
- Network timeout handling

#### 4.2 Add Monitoring & Health Checks
- `/health` endpoint for all services
- Qdrant connectivity check
- Ollama status check
- GPU memory monitoring
- Performance metrics collection
- Query response time tracking

#### 4.3 Implement Caching Layer
- Cache frequent queries
- Cache embedding results
- Redis integration
- Cache invalidation strategy
- LLM response caching for common questions

#### 4.4 Add Progress Tracking
- Real-time upload progress
- Chunking progress
- Embedding progress
- ETA calculations
- Time remaining for large documents

#### 4.5 Code Organization (from FUTURE_ENHANCEMENTS.md)
- Standardize imports (absolute vs relative)
- Remove legacy ChromaDB code
- Consistent configuration usage
- Clean up experimental code

---

### Priority 5: TESTING & DOCUMENTATION (Following Week)

#### 5.1 Create Test Suite
- Unit tests for all modules
- Integration tests for full pipeline
- Performance benchmarks
- Load testing for concurrent users
- Test with real municipal PDFs

#### 5.2 Update Documentation
- API documentation (OpenAPI/Swagger)
- Production deployment guide
- Configuration reference
- Architecture diagrams
- Scaling recommendations

#### 5.3 Security Enhancements (from FUTURE_ENHANCEMENTS.md)
- Input validation for uploaded files
- Rate limiting for API endpoints
- User authentication for multi-tenant
- API key management
- File type validation

---

### Priority 6: ENSEMBLE EXPERIMENT (Future)

#### 6.1 Multi-Model Implementation
**Status**: Research complete, architecture designed  
**Concerns**: Jina licensing for commercial use  
**Recommendation**: Use BGE + E5 + GTE instead  

**Tasks**:
1. Implement parallel embedding
2. Result fusion algorithm
3. Performance testing
4. A/B testing framework

**Reference**: `/docs/ENSEMBLE_ARCHITECTURE_RESEARCH.md`

---

## 🧹 Cleanup Checklist

### Immediate Cleanup (Do First)
- [ ] Fix store/retrieve mismatch
- [ ] Enable semantic chunking
- [ ] Test basic functionality

### File Consolidation
- [ ] Merge ingest files → single `ingest.py`
- [ ] Merge app files → single `app.py`
- [ ] Merge docker files → single `Dockerfile`
- [ ] Consolidate requirements → single `requirements.txt`
- [ ] Merge vector stores → single `vector_store.py`

### Archive Structure
```
src/deprecated/
├── ingest_backup.py
├── ingest_optimized.py
├── app_enhanced.py
├── vector_store_v1.py
└── config_v1.py

docs/deployment/
├── requirements-minimal.txt
├── requirements-core.txt
├── requirements-ml.txt
└── requirements-app.txt
```

### Remove Experimental Files
- [ ] Move `src/experimental/` to `docs/research/`
- [ ] Archive test scripts in `scripts/deprecated/`
- [ ] Clean up root directory files

---

## 📊 Success Metrics

| Metric | Current | Week 1 Target | Month Target |
|--------|---------|---------------|--------------|
| Code files | ~20 duplicates | 0 duplicates | Clean architecture |
| PDF processing | 10+ min/100pg | <1 min/100pg | <30s/100pg |
| Search accuracy | 0% (broken) | 80% | 95% |
| OCR support | No | Basic | Advanced |
| Test coverage | 0% | 30% | 80% |

---

## 🚀 Quick Start Commands

### After Priority 0 Fixes:
```bash
# Test the fixes
docker-compose down
docker-compose up --build -d
docker-compose logs -f munirag

# Verify collections
docker-compose exec munirag python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='qdrant', port=6333)
for c in client.get_collections().collections:
    print(f'{c.name}: {client.count(c.name).count} docs')
"
```

### After Cleanup:
```bash
# Archive old files
mkdir -p src/deprecated docs/deployment
mv src/ingest_backup.py src/ingest_optimized.py src/deprecated/
mv requirements-*.txt docs/deployment/

# Test everything still works
docker-compose up --build
```

---

## 📝 Notes

1. **Breaking Changes**: Moving to single vector_store.py will require data migration
2. **Backup First**: Create git branch before major cleanup
3. **Test Continuously**: Run tests after each consolidation
4. **Document Changes**: Update README with new structure

This is our roadmap to a clean, stable, production-ready MuniRAG!