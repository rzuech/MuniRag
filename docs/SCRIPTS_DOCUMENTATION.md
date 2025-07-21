# MuniRAG Scripts Documentation

## Maintenance Scripts

### fix_qdrant.py
**Purpose**: Fixes Qdrant database issues by purging all collections and starting fresh  
**When to use**: When Qdrant collections are corrupted or showing errors  
**Usage**: 
```bash
docker cp fix_qdrant.py munirag-munirag-1:/app/
docker exec munirag-munirag-1 python fix_qdrant.py
```
**What it does**:
1. Connects to Qdrant manager
2. Purges ALL collections (data loss!)
3. Verifies database is clean
4. Prompts user to re-upload PDFs

### check_qdrant.py
**Purpose**: Checks Qdrant database health and shows collection status  
**When to use**: To verify database state or debug issues  
**Usage**:
```bash
docker cp check_qdrant.py munirag-munirag-1:/app/
docker exec munirag-munirag-1 python check_qdrant.py
```
**Output**: Shows all collections, document counts, and health status

### purge_qdrant.py
**Purpose**: Manual purge tool with safety options  
**When to use**: For selective data cleanup  
**Features**:
- Can purge all or only munirag_* collections
- Interactive confirmation
- Detailed logging

### test_retrieval_accuracy.py
**Purpose**: Tests retrieval accuracy with predefined questions  
**When to use**: After changes to verify system still works correctly  
**Usage**:
```bash
docker cp test_retrieval_accuracy.py munirag-munirag-1:/app/
docker exec munirag-munirag-1 python test_retrieval_accuracy.py
```
**What it tests**:
- Document retrieval for common questions
- Shows if chunks are found
- Displays current configuration
- Helps diagnose accuracy issues

## Legacy/Testing Scripts

### verify_pdf_*.py scripts
Various PDF verification scripts created during debugging:
- `verify_pdf_ingestion.py` - Comprehensive PDF verification
- `verify_pdf_search.py` - Search-based verification
- `verify_metadata.py` - Metadata structure analysis
- `verify_pdf_complete.py` - Chunk completeness check

### Other Utilities
- `setup-local-git-filters.sh` - Removes Claude attribution from commits
- `test_system_integration.py` - Full system integration test
- `migrate_data.py` - Migrates data between collections

## Best Practices

1. **Always backup before destructive operations**
   ```bash
   docker exec munirag-munirag-1 python -c "from src.qdrant_manager import get_qdrant_manager; m = get_qdrant_manager(); m.health_check()" > backup_status.txt
   ```

2. **Copy scripts to container before running**
   ```bash
   docker cp script.py munirag-munirag-1:/app/
   ```

3. **Check logs after operations**
   ```bash
   docker logs munirag-munirag-1 --tail 50
   ```