# MuniRAG Master Priority List v2.1
*Last Updated: 2025-07-21*

## 🚨 Project Status Overview

### Current State
- **Version**: 0.90+ (multiple fixes applied)
- **Core Functionality**: ✅ WORKING - Upload, embed, store, retrieve all functioning
- **Performance**: ✅ GPU working (9 texts/sec), builds take ~2 min (was 43+)
- **Architecture**: Multi-model architecture implemented and tested

### Recently Fixed Issues ✅
1. **Store/Retrieve Mismatch**: FIXED - Using MultiModelVectorStore consistently
2. **LangChain Removal**: FIXED - Custom text splitter, no more dependency conflicts
3. **Docker Build**: FIXED - Deprecated files excluded, clean builds
4. **PDF Processing**: FIXED - TableFinder error resolved
5. **Qdrant Purge**: IMPLEMENTED - Configurable data persistence
6. **Accuracy Testing**: IMPLEMENTED - 77.7% baseline accuracy achieved (2025-07-21)
7. **Full Response Logging**: IMPLEMENTED - Complete LLM responses now captured
8. **Project Organization**: COMPLETED - All test files reorganized into /accuracy_testing/

### Remaining Issues
1. **Multiple Redundant Files**: Still need cleanup (see Priority 1)
2. **OCR Configuration**: Dependencies added but needs Docker setup
3. **Config.py Review**: Many settings may be outdated (see new Priority 5)

---

## 📋 Prioritized Task List

### Priority 0: CRITICAL ACCURACY IMPROVEMENTS (Immediate)

#### 0.1 Fix Semantic Chunking Configuration
**Status**: In Progress  
**Impact**: Major accuracy improvement  
**Fix**: Enable SEMANTIC_CHUNKING=true in .env
```bash
echo "SEMANTIC_CHUNKING=true" >> .env
echo "CHUNK_SIZE=500" >> .env  
echo "CHUNK_OVERLAP=100" >> .env
docker-compose restart munirag
```

#### 0.2 Improve 512 Token Limitation in Semantic Chunking
**Status**: Not started  
**Impact**: Better handling of long text chunks  
**Tasks**:
1. Research alternatives to all-MiniLM-L6-v2 with higher token limits
2. Implement sliding window approach for long chunks
3. Consider using BGE model for semantic similarity (1024 tokens)
4. Add fallback for chunks exceeding token limits
5. Test with long document sections

#### 0.3 Document Complete Chunking Pipeline
**Status**: Not started  
**Impact**: Understanding and optimization  
**Tasks**:
1. Map entire chunking flow from PDF → storage
2. Document where chunking happens (PDF processor)
3. Document where re-chunking happens (if any)
4. Analyze chunk size distribution
5. Identify optimization opportunities
6. Create visual diagram of data flow

#### 0.4 Implement Automated Web-Based Accuracy Testing
**Status**: ✅ COMPLETED (2025-07-21)  
**Impact**: Continuous quality assurance  
**Results**: 
- Built comprehensive test suite with 20+ questions
- Multi-dimensional scoring (40% factual, 30% completeness, 20% relevance, 10% coherence)
- **Achieved 77.7% baseline accuracy**
- Full LLM response logging implemented
- Created `/accuracy_testing/` directory structure
- Can run tests without user interaction

### Priority 1: CRITICAL BUG FIXES (Today - 2 hours)

#### 0.1 Fix Store/Retrieve Mismatch 🚨
**Status**: ✅ COMPLETED  
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
**Status**: ✅ COMPLETED (enabled in config)  
**Impact**: Better retrieval quality  
**Fix**: Already in .env.example, need to ensure it's in .env
```bash
SEMANTIC_CHUNKING=true
PDF_WORKERS=4
```

---

### Priority 0.5: GIT CLEANUP BEFORE PUSH (Immediate - 30 minutes)

#### 0.5.1 Remove Development Artifacts
**Status**: Not started  
**Impact**: Clean repository for public viewing  
**Tasks**:
1. Delete entire `archive_cleanup/` directory (30+ temp files)
2. Remove test/debug scripts from root: `check_*.py`, `verify_*.py`
3. Remove backup files: `.env.backup`, `src/pdf_processor.langchain_backup.py`
4. Delete `scripts/` directory (old test scripts)
5. Clean some scripts from `accuracy_testing/scripts/` (debug_test.py, simple_test.py, etc.)

**Commands**:
```bash
rm -rf archive_cleanup/
rm -f check_*.py verify_*.py
rm -f .env.backup
rm -f src/pdf_processor.langchain_backup.py
rm -rf scripts/
# Keep main test runners in accuracy_testing/scripts/
```

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
**Status**: ✅ COMPLETED (2025-07-20)
**Current State**: Single optimized Dockerfile
- Renamed `Dockerfile.optimized` → `Dockerfile`
- Removed old Dockerfile
- Docker build times reduced from 43+ min to ~2 min

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

### Priority 5: CONFIG.PY REVIEW & CLEANUP (Next Week)

#### 5.1 Review All Configuration Settings
**Status**: Not started
**Why**: Many settings in config.py may be outdated or unused
**Tasks**:
1. Audit each setting to verify if it's actually used
2. Remove deprecated settings (e.g., CHROMA_DIR, old ChromaDB configs)
3. Document what each setting actually controls
4. Identify settings that have no effect
5. Create migration guide for breaking config changes

#### 5.2 Settings to Review
- `COLLECTION_NAME` - Still says "munirag_docs" but we use model-specific names
- `CHROMA_DIR` - Legacy ChromaDB setting, not used
- `USE_HYBRID_SEARCH` - Verify if implemented
- `RERANK_TOP_K` - Check if reranking is implemented
- `GPU_MEMORY_THRESHOLD` - Check if GPU switching logic exists
- `MAX_PAGES_CRAWL` - Website ingestion not yet implemented
- Many others that may be placeholders

### Priority 5.5: EXPAND TEST DATA CORPUS (Medium Priority)

#### 5.5.1 Gather Municipality Documents
**Status**: Not started  
**Impact**: Better testing coverage and real-world validation  
**Tasks**:
1. Research other municipality websites
2. Download 30-50 diverse municipal PDFs:
   - City codes and ordinances
   - Permit guides and applications
   - Zoning regulations
   - Budget documents
   - Meeting minutes
   - Policy documents
3. Organize by document type and complexity
4. Create test questions for each document type
5. Never commit PDFs to Git (keep in Test-PDFs/)

### Priority 6: TESTING & DOCUMENTATION (Following Week)

#### 6.1 Create Test Suite
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

### Priority 7: VALIDATION & MONITORING (Next Week)

#### 7.1 PDF Ingestion Validation
**Status**: Not started  
**Impact**: Ensure complete document ingestion  
**Tasks**:
1. Create automated validation for PDF completeness
2. Add page count verification after ingestion
3. Implement chunk distribution analysis
4. Add error reporting for failed pages
5. Create ingestion quality metrics

#### 7.2 Review all-MiniLM-L6-v2 Usage
**Status**: Not started  
**Impact**: Understanding and documenting model usage  
**Tasks**:
1. Document why all-MiniLM-L6-v2 is used for semantic chunking
2. Review if this is the best model for the task
3. Consider making it configurable
4. Add to model registry if keeping
5. Document token limit implications (512 tokens)

#### 7.3 PDF Ingestion Error Handling
**Status**: Not started  
**Impact**: Better error visibility and recovery  
**Tasks**:
1. Add comprehensive error logging for each page
2. Create error summary report after ingestion
3. Implement retry logic for failed pages
4. Add OCR failure recovery
5. Create ingestion health dashboard

### Priority 8: DOCUMENTATION REORGANIZATION (Low Priority)

#### 8.1 Consolidate Documentation
**Status**: Not started  
**Impact**: Better developer experience and maintainability  
**Tasks**:
1. Create docs/INDEX.md as master documentation hub
2. Consolidate related docs into subdirectories:
   - docs/operations/ - Scripts, commands, troubleshooting
   - docs/architecture/ - System design, data flow
   - docs/testing/ - Testing strategies, benchmarks
   - docs/deployment/ - Production setup, configuration
3. Remove duplicate information
4. Create clear navigation structure
5. Add "Quick Start" vs "Deep Dive" sections

#### 8.2 Script Documentation
**Status**: Not started  
**Impact**: Understanding of utility scripts  
**Tasks**:
1. Document all scripts in scripts/ directory
2. Create SCRIPTS_REFERENCE.md with:
   - Script name and purpose
   - Usage examples
   - Expected outputs
3. Add docstrings to all Python scripts
4. Remove obsolete scripts

### Priority 9: ENSEMBLE EXPERIMENT (Future)

#### 8.1 Multi-Model Implementation
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
- [x] Fix store/retrieve mismatch ✅
- [x] Enable semantic chunking ✅
- [x] Test basic functionality ✅
- [x] Implement accuracy testing ✅

### File Consolidation
- [x] Merge docker files → single `Dockerfile` ✅
- [x] Merge vector stores → using `vector_store_v2.py` as MultiModelVectorStore ✅
- [x] Remove deprecated directory ✅
- [ ] Merge ingest files → single `ingest.py`
- [ ] Merge app files → single `app.py`
- [ ] Consolidate requirements → single `requirements.txt`

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
| Code files | ~10 duplicates | 0 duplicates | Clean architecture |
| PDF processing | <1 min/100pg ✅ | <1 min/100pg | <30s/100pg |
| Search accuracy | 77.7% ✅ | 80% | 95% |
| OCR support | No | Basic | Advanced |
| Test coverage | 20% ✅ | 30% | 80% |
| Response logging | Full ✅ | Full | Full + Analytics |

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