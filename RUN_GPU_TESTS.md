# 🧪 GPU Testing Commands - Run These Now!

## Step 1: Copy All Test Scripts
```bash
# Copy all testing scripts to container
docker cp test_speed.py munirag-munirag-1:/app/
docker cp scripts/test_optimization_impact.py munirag-munirag-1:/app/scripts/
docker cp scripts/gpu_testing_framework.py munirag-munirag-1:/app/scripts/
docker cp scripts/deep_gpu_analysis.py munirag-munirag-1:/app/scripts/
```

## Step 2: Quick Performance Check (30 seconds)
```bash
# See current embedding speed
docker-compose exec munirag python3 /app/test_speed.py
```
**Expected**: Should see ~3000+ texts/sec (up from 300)

## Step 3: Optimization Impact Test (1 minute)
```bash
# Compare page-by-page vs batched processing
docker-compose exec munirag python3 /app/scripts/test_optimization_impact.py
```
**Expected**: Should show 5-10x speedup potential

## Step 4: Comprehensive GPU Tests (3 minutes)
```bash
# Run full GPU testing framework
docker-compose exec munirag python3 /app/scripts/gpu_testing_framework.py
```
**Expected**: All tests should pass for production readiness

## Step 5: Deep Analysis (Optional, 5 minutes)
```bash
# Detailed performance analysis
docker-compose exec munirag python3 /app/scripts/deep_gpu_analysis.py
```

## Step 6: Monitor GPU During PDF Upload
In a separate terminal while uploading a PDF:
```bash
# Watch GPU utilization
docker-compose exec munirag watch -n 1 nvidia-smi
```
**Current**: GPU usage ~3-5%
**After fixes**: Should see 70-90%

## 🔥 Quick Fix to Test

To immediately test the optimized version:
```bash
# Backup current ingest.py
docker-compose exec munirag cp /app/src/ingest.py /app/src/ingest_backup.py

# Copy optimized version
docker cp src/ingest_optimized.py munirag-munirag-1:/app/src/ingest.py

# Restart to load changes
docker-compose restart munirag
```

Then upload a PDF and watch the dramatic speedup!

## 📊 What You Should See

### Current Performance:
- Embedding speed: ~3000 texts/sec ✅ (fixed)
- PDF processing: Still slow due to page-by-page
- GPU utilization: <5%

### After Batching Fix:
- Embedding speed: ~3500-4000 texts/sec
- PDF processing: 5-10x faster
- GPU utilization: 70-90%

### Time Estimates:
- 50-page PDF: 10 min → 30 seconds
- 500-page PDF: 60 min → 2-3 minutes

## 🚨 If Tests Fail

1. **Low embedding speed (<2000 texts/sec)**
   - GPU not being used properly
   - Check CUDA availability
   
2. **Optimization test shows no speedup**
   - Batching might already be implemented
   - Check current ingest.py code

3. **GPU framework tests fail**
   - Memory issues
   - Driver problems
   - Model loading errors

## 📝 Results to Share

After running tests, please share:
1. Output of `test_speed.py`
2. Speedup ratio from `test_optimization_impact.py`
3. Pass/fail summary from `gpu_testing_framework.py`
4. GPU utilization % during PDF upload

This will confirm if the fixes are working and what else needs attention before launch!