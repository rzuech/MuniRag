# 🔄 MuniRAG Complete Rebuild Instructions

After the major embedding architecture changes, follow these steps for a clean rebuild.

## 1. Stop Everything
```bash
docker-compose down -v
```
The `-v` flag removes volumes, giving us a fresh start.

## 2. Clean Docker System (Optional but Recommended)
```bash
# Remove old images to force fresh builds
docker image prune -a

# Or just remove MuniRAG images
docker images | grep munirag | awk '{print $3}' | xargs docker rmi -f
```

## 3. Clear Local Caches
```bash
# Remove model cache (they'll be re-downloaded)
rm -rf models/huggingface/*
rm -rf models/sentence_transformers/*

# Clear Qdrant data
rm -rf qdrant_storage/*

# Clear any uploaded test data
rm -rf data/*
```

## 4. Update Configuration
```bash
# Ensure .env has BGE as default
# Check that EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
cat .env | grep EMBEDDING_MODEL
```

## 5. Rebuild Everything
```bash
# Build with no cache to ensure all changes are applied
docker-compose build --no-cache

# Start services
docker-compose up
```

## 6. Wait for Model Downloads
First run will download:
- BGE model (~1.5GB) - should take 2-3 minutes
- Watch logs for "Model loaded" message

## 7. Verify GPU Setup
```bash
# Check GPU is recognized
docker-compose exec munirag nvidia-smi

# Run GPU benchmark
docker-compose exec munirag python3 /app/scripts/benchmark_embeddings.py
```

## 8. Reset Qdrant for New Model
```bash
# Since we switched from Jina (1024) to BGE (1024), dimensions match
# But good practice to reset anyway
docker-compose exec munirag python3 /app/scripts/reset_qdrant.py
```

## 9. Test PDF Upload
1. Navigate to http://localhost:8501
2. Upload a small test PDF
3. Watch console - should see:
   - "Using batch size: 256" (for good GPUs)
   - "~3000+ texts/second" 
   - GPU utilization 70-90%
   - Completion in 2-3 minutes (not 1 hour!)

## 10. Troubleshooting

### If GPU not working:
```bash
# Check CUDA in container
docker-compose exec munirag python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify nvidia runtime
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

### If models won't download:
```bash
# Check Hugging Face connectivity
docker-compose exec munirag python3 -c "from huggingface_hub import scan_cache_dir; print(scan_cache_dir())"

# Force re-download
docker-compose exec munirag python3 /app/scripts/download_models.py
```

### If Qdrant has issues:
```bash
# Full Qdrant reset
docker-compose down
rm -rf qdrant_storage/*
docker-compose up qdrant
# Wait 30 seconds
docker-compose up munirag
```

## Expected Performance After Rebuild

| Metric | Before (Jina) | After (BGE) |
|--------|---------------|-------------|
| PDF Ingestion (50 pages) | 60+ minutes | 2-3 minutes |
| GPU Utilization | 3% | 70-90% |
| Texts/Second | ~117 | ~3,743 |
| Memory Usage | 5-7GB | 1.5-2GB |

## Quick Verification Commands
```bash
# Check embedding model
docker-compose exec munirag python3 -c "from src.config import settings; print(f'Model: {settings.EMBEDDING_MODEL}')"

# Check dimension
docker-compose exec munirag python3 -c "from src.config import settings; print(f'Dimension: {settings.get_embedding_dimension()}')"

# Count documents
docker-compose exec munirag python3 -c "from src.vector_store import VectorStore; print(f'Docs: {VectorStore().client.count(\"munirag_docs\")}')"
```

## Notes
- First PDF will be slower due to model loading
- Subsequent PDFs should process at full speed
- Monitor GPU memory - reduce batch size if OOM
- BGE is 32x faster than Jina - enjoy the speed!