# MuniRAG Fallback Plan & Disaster Recovery

## Quick Recovery Commands

### 1. Emergency Rollback (If Something Goes Wrong)
```bash
# Immediate rollback to backup branch
git checkout backup-before-rearchitecture

# Restore backup source files
cp -r src_backup_* src/

# Restart with old configuration
docker-compose down
docker-compose up --build
```

### 2. Model Loading Failures

#### Symptom: Jina v3 FileNotFoundError
```bash
# Quick fix: Switch to BGE model
export DEFAULT_EMBEDDING_MODEL=bge-large-en
# or
sed -i 's/jina-v3/bge-large-en/g' .env
```

#### Symptom: Out of Memory
```bash
# Use smaller model
export DEFAULT_EMBEDDING_MODEL=bge-base-en
# Reduce batch size
export EMBEDDING_BATCH_SIZE=16
```

### 3. API Failures

#### Symptom: FastAPI won't start
```bash
# Use fallback main.py
cp main.py main_backup.py
cp main_v2.py main.py

# If that fails, minimal mode:
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

#### Symptom: CORS errors
```python
# Emergency CORS fix in main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TEMPORARY - restrict after fixing
    allow_methods=["*"],
    allow_headers=["*"]
)
```

### 4. PDF Processing Failures

#### Symptom: Parallel processing crashes
```python
# Fallback to sequential processing
processor = ParallelPDFProcessor(num_workers=1, enable_ocr=False)
```

#### Symptom: OCR failures
```python
# Disable OCR
processor = ParallelPDFProcessor(enable_ocr=False)
```

### 5. Database Issues

#### Symptom: Qdrant connection failed
```bash
# Restart Qdrant
docker-compose restart qdrant

# Check Qdrant health
curl http://localhost:6333/health

# Use in-memory fallback (temporary)
export VECTOR_STORE_TYPE=memory
```

## Component-Specific Fallbacks

### Embedder Fallbacks
```python
# Priority order (fastest to most capable)
FALLBACK_MODELS = [
    "bge-large-en",      # Fast, reliable
    "e5-large",          # Good multilingual
    "bge-m3",            # Multilingual backup
    "jina-v3"            # Most capable but problematic
]

def get_working_embedder():
    for model in FALLBACK_MODELS:
        try:
            return MultiModelEmbedder(model)
        except:
            continue
    raise Exception("No working embedding model found")
```

### PDF Processor Fallbacks
```python
def process_pdf_with_fallback(pdf_path):
    strategies = [
        lambda: ParallelPDFProcessor(num_workers=4, enable_ocr=True),
        lambda: ParallelPDFProcessor(num_workers=2, enable_ocr=False),
        lambda: PDFProcessor(),  # Old single-threaded
    ]
    
    for strategy in strategies:
        try:
            processor = strategy()
            return processor.process_pdf(pdf_path)
        except:
            continue
    raise Exception("All PDF processing strategies failed")
```

## Environment Variable Overrides

Create `.env.emergency` for quick recovery:
```bash
# Minimal working configuration
DEFAULT_EMBEDDING_MODEL=bge-large-en
ENABLE_OCR=false
PDF_WORKERS=1
ENABLE_PLUGINS=false
ENABLE_RERANKING=false
MAX_CHUNK_SIZE=300
BATCH_SIZE=16
```

Apply emergency config:
```bash
cp .env .env.backup
cp .env.emergency .env
docker-compose restart
```

## Data Recovery

### Backup Critical Data
```bash
# Before any major change
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    qdrant_storage/ \
    data/ \
    .env \
    docker-compose.yml
```

### Restore Data
```bash
# List backups
ls -la backup_*.tar.gz

# Restore specific backup
tar -xzf backup_20240115_143022.tar.gz
```

## Health Checks

### Quick System Check
```bash
#!/bin/bash
# save as check_health.sh

echo "Checking services..."

# API Health
curl -s http://localhost:8000/health || echo "API Failed"

# Qdrant Health  
curl -s http://localhost:6333/health || echo "Qdrant Failed"

# Ollama Health
curl -s http://localhost:11434/api/tags || echo "Ollama Failed"

# Test Query
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}' || echo "Query Failed"
```

## Monitoring Commands

```bash
# Watch logs
docker-compose logs -f munirag

# Check memory usage
docker stats

# Monitor GPU
nvidia-smi -l 1

# Check disk space
df -h
```

## Common Issues & Solutions

### Issue: "torch.cuda.OutOfMemoryError"
```python
# Solution 1: Clear cache
import torch
torch.cuda.empty_cache()

# Solution 2: Reduce batch size
embedder = MultiModelEmbedder("bge-large-en")
embeddings = embedder.encode(texts, batch_size=8)  # Smaller batch

# Solution 3: Use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### Issue: "Connection refused" errors
```bash
# Check all services are running
docker-compose ps

# Restart everything
docker-compose down
docker-compose up -d

# Check ports
netstat -tlnp | grep -E "8000|8501|6333|11434"
```

### Issue: Slow performance
```bash
# Quick performance fixes
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Disable debug logging
export LOG_LEVEL=WARNING
```

## Emergency Contacts & Resources

- GitHub Issues: https://github.com/anthropics/munirag/issues
- Docker Hub Status: https://status.docker.com/
- PyTorch Forums: https://discuss.pytorch.org/
- Qdrant Docs: https://qdrant.tech/documentation/

## Rollback Procedure

1. **Stop Current Services**
   ```bash
   docker-compose down
   ```

2. **Checkout Backup Branch**
   ```bash
   git checkout backup-before-rearchitecture
   ```

3. **Restore Original Files**
   ```bash
   git reset --hard
   ```

4. **Rebuild and Start**
   ```bash
   docker-compose build --no-cache
   docker-compose up
   ```

5. **Verify Working**
   ```bash
   curl http://localhost:8000/health
   ```

## Testing After Recovery

Run minimal test suite:
```python
# test_recovery.py
import requests

def test_basic_functionality():
    # Test health
    r = requests.get("http://localhost:8000/health")
    assert r.status_code == 200
    
    # Test query
    r = requests.post("http://localhost:8000/api/query", 
                     json={"query": "test"})
    assert r.status_code == 200
    
    print("✓ Basic functionality restored")

if __name__ == "__main__":
    test_basic_functionality()
```

Remember: **When in doubt, use the simplest configuration that works!**