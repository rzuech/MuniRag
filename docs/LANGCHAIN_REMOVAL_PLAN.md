# LangChain Removal Plan

*Created: 2025-07-20*
*Build Time Issue: 35+ minutes stuck on LangChain dependency resolution*

## 📊 Current Status

### Build Issue
- Docker build spending 35+ minutes resolving LangChain dependencies
- Cycling through versions: 0.3.21 → 0.3.20 → ... → 0.3.3
- Downloading hundreds of unnecessary dependencies

### Usage Analysis
LangChain is used ONLY for text splitting in 2 active files:
1. `src/pdf_processor.py` - RecursiveCharacterTextSplitter
2. `scripts/deep_gpu_analysis.py` - Same text splitter

The main ingestion system (`ingest_parallel.py` + `pdf_parallel_processor.py`) already has custom chunking!

## 🎯 Removal Strategy

### Option 1: Complete Removal (Recommended)
**Effort**: 30 minutes
**Benefit**: Remove 500MB+ of dependencies, fix build times

Steps:
1. Backup `pdf_processor.py`
2. Replace LangChain splitter with custom implementation
3. Remove from requirements.txt
4. Update deprecated imports in app.py
5. Test thoroughly

### Option 2: Pin Specific Version
**Effort**: 5 minutes
**Benefit**: Faster resolution, but keeps bloat

```
langchain==0.1.0
langchain-text-splitters==0.1.0
# Remove langchain-community entirely
```

### Option 3: Use Only Text Splitters
**Effort**: 10 minutes  
**Benefit**: Minimal dependencies

```
# Just the text splitter package
langchain-text-splitters==0.1.0
# Remove langchain and langchain-community
```

## 🔧 Implementation Details

### Files to Modify
1. **src/pdf_processor.py**
   - Replace: `from langchain.text_splitter import RecursiveCharacterTextSplitter`
   - With: Custom text splitter function

2. **requirements.txt**
   - Remove all langchain lines

3. **src/app.py**
   - Remove deprecated fallback imports
   - Use only `ingest.py` (renamed from ingest_parallel.py)

### Custom Text Splitter
Already implemented in `pdf_parallel_processor.py`:
- `_fixed_chunk_text()` - Simple chunking
- `_semantic_chunk_text()` - Smart chunking

## ⚠️ Risk Assessment

### Low Risk
- Custom chunking already tested in parallel processor
- Only affects 2 files
- Main ingestion doesn't use LangChain

### Testing Required
1. PDF upload and chunking
2. Verify chunk sizes are similar
3. Test retrieval quality

## 📝 Documentation

### Where LangChain Was Removed
- `src/pdf_processor.py` - Replaced text splitter
- `requirements.txt` - Removed 3 langchain packages
- `scripts/deep_gpu_analysis.py` - Made optional or updated

### Why Removed
1. 35+ minute build times due to dependency resolution
2. 500MB+ of unused dependencies
3. Only used for basic text splitting
4. Custom implementation already exists

## 🚀 Expected Results

### Before
- Build time: 35+ minutes
- Docker image size: Larger
- Dependencies: 100+ packages

### After  
- Build time: 5-10 minutes
- Docker image size: ~500MB smaller
- Dependencies: Only what's needed

## 🔄 Rollback Plan

If issues arise:
1. `git checkout HEAD -- requirements.txt src/pdf_processor.py`
2. Or use backup: `cp src/pdf_processor.langchain_backup.py src/pdf_processor.py`
3. Rebuild with LangChain

---

This plan ensures safe removal of LangChain while maintaining all functionality.