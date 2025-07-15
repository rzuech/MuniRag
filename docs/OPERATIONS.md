# MuniRAG Operations Guide

## 🚨 Emergency Procedures

### Quick Rollback
If something goes wrong:
```bash
# Stop everything
docker-compose down

# Revert code
git checkout main

# Rebuild
docker-compose up --build
```

### Common Issues & Fixes

#### Progress Bar Stuck
**Symptom**: Upload progress bar full but not completing

**Fix**: This is normal during embedding phase. Check logs:
```bash
docker-compose logs -f munirag | grep -i progress
```

#### No Logs Visible
**Symptom**: Console shows no processing information

**Fix**: Already fixed. If still an issue:
```bash
# Check environment
docker-compose exec munirag env | grep VERBOSITY
# Should show: TRANSFORMERS_VERBOSITY=warning
```

#### Slow Processing
**Symptom**: PDFs processing slowly, few CPUs active

**Fix**: Verify both parallel systems are active:
```bash
# Check parallel PDF extraction
docker-compose exec munirag python3 -c "from src.ingest import PARALLEL_AVAILABLE; print(f'Parallel PDF extraction: {PARALLEL_AVAILABLE}')"

# Check parallel embedding
docker-compose exec munirag python3 -c "from src.ingest import PARALLEL_EMBEDDER_AVAILABLE; print(f'Parallel embedding: {PARALLEL_EMBEDDER_AVAILABLE}')"

# Test the parallel embedder
docker-compose exec munirag python3 test_parallel_embedder.py
```

**NEW**: Two-Phase Parallel Processing
1. **Extraction Phase** (20 seconds for 12MB PDF)
   - All CPUs active extracting text from pages
   - Progress bar shows "Extracting page X/Y"
   
2. **Embedding Phase** (now <1 minute, was 10+ minutes)
   - Multiple CPUs creating embeddings in parallel
   - Progress bar shows "Creating embeddings: X/Y"
   - Uses thread pool for concurrent processing

## 🔧 Configuration

### Environment Variables
```bash
# Force specific worker count
PDF_WORKERS=4

# Control logging
LOG_LEVEL=INFO
TRANSFORMERS_VERBOSITY=warning

# Embedding model
DEFAULT_EMBEDDING_MODEL=bge-large-en
```

### CPU Worker Configuration
- Auto-detected based on CPU count
- Override with `PDF_WORKERS` if needed
- Safe defaults for all systems

## 📊 Monitoring

### Check System Status
```bash
# Container status
docker-compose ps

# Resource usage
docker stats

# Application logs
docker-compose logs -f munirag
```


## Commands for logs and GPU monitoring:

Docker Logs Commands

### View all logs from startup
docker-compose logs

### View logs for specific service with timestamps
docker-compose logs -t munirag
docker-compose logs -t ollama

### Follow logs in real-time (like tail -f)
docker-compose logs -f

### Follow logs for specific service
docker-compose logs -f munirag

### Show last 100 lines and follow
docker-compose logs --tail=100 -f munirag

### Show logs since specific time
docker-compose logs --since 30m  # last 30 minutes
docker-compose logs --since 2h   # last 2 hours

### Save logs to file
docker-compose logs > munirag_logs.txt

## NVIDIA GPU Usage Commands

### Basic GPU monitoring (updates every 2 seconds)
nvidia-smi

### Continuous monitoring (updates every 0.5 seconds)
watch -n 0.5 nvidia-smi

### Compact view with just utilization
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1

### Even more compact (percentage only)
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv -l 1

### One-liner for monitoring during operations
nvidia-smi -l 1 | grep -E "MiB|%"

## Combined Monitoring Setup

### Open two terminals:

#### Terminal 1 - Logs:
docker-compose logs -f munirag | grep -E "embed|PDF|GPU|Error"

#### Terminal 2 - GPU:
watch -n 0.5 'nvidia-smi | grep -A 3 "RTX\|MiB"'

## Debugging Specific Issues

### Check if container is running
docker-compose ps

### See container resource usage
docker stats

### Enter container for debugging
docker-compose exec munirag bash

### Check Python output directly
docker-compose exec munirag python -c "print('Container is responsive')"

### View Docker daemon logs (if needed)
sudo journalctl -u docker.service -f

The most useful for you will likely be:
docker-compose logs -f munirag

This shows real-time logs from your MuniRAG container where you'll see embedding speeds, errors, and
processing status.

--------------------------------------------------------------------------------
  
### Performance Metrics

#### PDF Processing Performance (v2.0)
- **Extraction Phase**: ~200+ pages/minute (all CPUs active)
- **Embedding Phase**: ~1000+ chunks/minute (multiple threads)
- **Overall**: 12MB PDF (~250 pages) processes in <90 seconds

#### Old vs New Comparison
| Phase | Old Time | New Time | Speedup |
|-------|----------|----------|---------|
| Text Extraction | 3-5 minutes | 20 seconds | 10x |
| Embedding Creation | 10+ minutes | <1 minute | 15x |
| **Total** | **15+ minutes** | **<90 seconds** | **10x+** |

#### Resource Usage
- **CPU**: All cores during extraction, 50% during embedding
- **Memory**: 2-4GB typical, up to 6GB for large PDFs
- **Disk I/O**: Minimal (all processing in memory)

## 🔄 Maintenance

### Daily Operations
1. Monitor disk space (vector DB grows with documents)
2. Check logs for errors
3. Verify API health: `curl http://localhost:8000/health`

### Weekly Tasks
1. Backup vector database
2. Review usage statistics
3. Update documents as needed

### Cleanup Commands
```bash
# Remove old logs
docker-compose exec munirag find /app -name "*.log" -mtime +7 -delete

# Clean Python cache
docker-compose exec munirag find /app -type d -name "__pycache__" -exec rm -rf {} +

# Prune Docker
docker system prune -f
```

## 🛡️ Security

### API Security
- Default: No authentication on local deployment
- Production: Enable API keys
- CORS configured for municipality domains

### Data Protection
- PDFs never committed to Git
- Test outputs sanitized
- Vector DB persisted in Docker volume

### Network Security
- Ports 8000 (API) and 8501 (Streamlit)
- Internal Docker network for services
- No external dependencies during runtime

## 📞 Support Procedures

### Diagnostics Package
Generate a diagnostics report:
```bash
# Create diagnostics
docker-compose exec munirag python3 -c "
import platform
import multiprocessing
import torch
print(f'System: {platform.system()}')
print(f'CPUs: {multiprocessing.cpu_count()}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Workers: {get_optimal_workers()}')
"
```

### Common Support Scenarios

1. **"It's slow"**
   - Check CPU worker count
   - Verify parallel processing active
   - Review PDF size and complexity

2. **"It's stuck"**
   - Check embedding phase in logs
   - Large PDFs take time for embeddings
   - Progress bar updates during extraction, not embedding

3. **"No results"**
   - Verify PDFs were ingested
   - Check vector DB has data
   - Test with simple queries first