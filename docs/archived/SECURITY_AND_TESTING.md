# 🔒 MuniRAG Security & Testing Guide

## Overview

This document explains the security measures and testing procedures implemented to protect sensitive municipal data while enabling comprehensive testing of MuniRAG v2.0.

## 🛡️ Security Measures Implemented

### 1. **Comprehensive .gitignore**
All sensitive data is excluded from Git tracking:
```
# Critical exclusions
*.pdf, *.PDF              # ALL PDF files
Test-PDFs/                 # Test PDF directory
tests/test_data/           # Test data directory
tests/test_results/        # Test output directory
*_test_report*.json        # Test reports
```

**Verification**: Run `git status` - no PDFs should appear

### 2. **Local Git Filters (Claude Attribution)**
Remove Claude attribution from commits automatically:

```bash
# One-time setup (already done)
./setup-local-git-filters.sh

# Test it works
./test_git_filters.sh
```

This removes:
- "🤖 Generated with [Claude Code]" lines
- "Co-Authored-By: Claude" lines

**Note**: This is LOCAL only - doesn't affect other developers

### 3. **Safe Test Runner**
The `run_all_tests_safely.py` script:
- Sanitizes all output to remove PDF names
- Stores results only in gitignored directories
- Provides summary without exposing metadata
- Verifies security before running

## 📊 Your Test PDFs

Located in `tests/test_data/pdfs/` (gitignored):

1. **Procurement Ethics Policy** (89KB)
2. **Create Certificate of Use** (667KB)
3. **How to Register and Create Permit** (1.2MB)
4. **HowtoUseACA Homeowner** (869KB)
5. **How to Submit Vacation Rental** (2.4MB)
6. **Weston Code of Ordinances** (12.2MB) ⭐

## 🧪 Testing Procedures

### Quick Performance Demo
```bash
# See performance improvements without full environment
python3 demo_pdf_performance.py
```

**Results**: Shows 48x speedup for your PDFs!

### Full Test Suite (Safe)
```bash
# Run all tests with security measures
python3 run_all_tests_safely.py
```

**What it does**:
- Verifies environment security
- Runs performance tests
- Tests embedding models
- Saves sanitized results

### PDF Ingestion
```bash
# Process all your PDFs into the system
python3 ingest_test_pdfs.py
```

### Municipal RAG Test
```bash
# Test complete pipeline with your documents
python3 test_municipal_rag.py
```

## 🔍 Example Queries for Your PDFs

After ingestion, try these queries:

**Vacation Rentals:**
- "How do I apply for a vacation rental permit?"
- "What insurance is required for vacation rentals?"

**Building Permits:**
- "What are the steps to get a building permit?"
- "How do I register for a permit application?"

**Ethics:**
- "Can city employees accept gifts?"
- "What are the procurement ethics rules?"

**Ordinances:**
- "What are the noise ordinance hours?"
- "What are residential zoning requirements?"
- "What are the pool fence requirements?"

## 🚀 Performance Highlights

| Document | Size | Old Time | New Time | Speedup |
|----------|------|----------|----------|---------|
| Ethics Policy | 89KB | 0.2 min | 0.2 sec | 48x |
| Certificate Guide | 667KB | 1.3 min | 1.6 sec | 48x |
| Permit Guide | 1.2MB | 2.4 min | 3.0 sec | 48x |
| ACA Guide | 869KB | 1.7 min | 2.1 sec | 48x |
| Vacation Rental | 2.4MB | 4.6 min | 5.8 sec | 48x |
| **Ordinances** | **12MB** | **24 min** | **30 sec** | **48x** |

**Total time saved**: 33 minutes per full update!

## 🌐 Widget Integration

After processing PDFs, test the widget:

1. Visit: http://localhost:8000/widget
2. Or open: `demo_municipal_widget.html` in browser
3. Try the example queries

## ⚠️ Security Reminders

1. **NEVER** commit PDFs to the repository
2. **ALWAYS** use the safe test scripts
3. **CHECK** `git status` before committing
4. **REVIEW** test outputs before sharing
5. **KEEP** sensitive data in gitignored directories

## 🔧 Troubleshooting

**PDFs not found?**
```bash
ls -la tests/test_data/pdfs/
# Should show 6 PDF files
```

**Git tracking PDFs?**
```bash
git status | grep -i pdf
# Should return nothing
```

**Tests failing?**
```bash
# Check Docker is running
docker ps

# Check API is up
curl http://localhost:8000/health
```

## 📝 Quick Reference

```bash
# Performance demo (no setup needed)
./demo_pdf_performance.py

# Run all tests safely
./run_all_tests_safely.py

# Test git filters
./test_git_filters.sh

# Process your PDFs
python3 ingest_test_pdfs.py

# Widget demo
open demo_municipal_widget.html
```

---

**Remember**: All security measures are in place. Your PDFs and their metadata are protected from Git exposure.