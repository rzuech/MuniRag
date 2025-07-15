# MuniRAG Testing Guide

## 🧪 Overview

This guide covers all testing procedures for MuniRAG v2.0, including performance testing, security verification, and system validation.

## 🔒 Security First

### PDF Protection
All test PDFs are protected from Git exposure:
- PDFs stored in `tests/test_data/pdfs/` (gitignored)
- Test outputs in `tests/test_results/` (gitignored)
- File names sanitized in logs

### Git Filter for Claude Attribution
Remove Claude attribution from commits:
```bash
python3 setup_git_filters.py
```

## 📊 Performance Testing

### Quick Performance Check
```bash
python3 simple_pdf_test.py
```
Shows processing time estimates without needing full environment.

### Full System Test
```bash
python3 test_complete_system.py
```
Checks:
- Docker status
- API health
- PDF availability
- Runs appropriate tests based on environment

### Safe Test Runner
```bash
python3 run_all_tests_safely.py
```
Runs all tests with sanitized output to prevent metadata leaks.

## 🏛️ Municipal Document Testing

### Your Test PDFs
1. **Procurement Ethics Policy** (89KB) - Ethics guidelines
2. **Certificate of Use Guide** (667KB) - Business certificates
3. **Permit Application Guide** (1.2MB) - Building permits
4. **ACA Homeowner Guide** (869KB) - Homeowner portal
5. **Vacation Rental Guide** (2.3MB) - Short-term rentals
6. **Weston Code of Ordinances** (12MB) - Complete city laws

### Test Queries
After ingesting PDFs, try:
- "How do I apply for a vacation rental permit?"
- "What are the noise ordinance quiet hours?"
- "What documents do I need for a building permit?"
- "Can city employees accept gifts from contractors?"
- "What are the pool fence requirements?"

## 🚀 Quick Testing Workflow

1. **Check System**
   ```bash
   docker-compose ps
   ```

2. **Ingest Test PDFs**
   ```bash
   docker-compose exec munirag python3 ingest_test_pdfs.py
   ```

3. **Test via Web**
   - Streamlit: http://localhost:8501
   - Widget: http://localhost:8000/widget

4. **Monitor Performance**
   - Watch CPU usage during upload
   - Check logs: `docker-compose logs -f munirag`

## 📈 Expected Performance

### Processing Speed (with 4 CPU cores)
- Small PDFs (<1MB): 2-5 seconds
- Medium PDFs (1-3MB): 5-15 seconds
- Large PDFs (10MB+): 30-60 seconds

### Old vs New
- **Old**: ~10 pages/minute (single CPU)
- **New**: ~200+ pages/minute (all CPUs)

## 🔍 Troubleshooting

### Progress Bar Stuck
- Normal during embedding phase
- Check logs for actual progress
- Large files take time for embeddings

### No Logs Visible
- Fixed in latest version
- Check: `docker-compose logs munirag`

### PDFs Not Processing Fast
- Verify parallel processing active
- Check CPU count detection
- Monitor with `htop` or Task Manager

## ✅ Verification Commands

```bash
# Check CPU detection
docker-compose exec munirag python3 -c "from src.pdf_parallel_adapter import get_optimal_workers; print(f'Workers: {get_optimal_workers()}')"

# Check parallel processing available
docker-compose exec munirag python3 -c "from src.ingest import PARALLEL_AVAILABLE; print(f'Parallel: {PARALLEL_AVAILABLE}')"

# Test PDF processing
docker-compose exec munirag python3 simple_pdf_test.py
```