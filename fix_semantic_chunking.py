#!/usr/bin/env python3
"""
Fix semantic chunking configuration
Ensures SEMANTIC_CHUNKING is enabled for better accuracy
"""

import os
import subprocess

print("=== Fixing Semantic Chunking Configuration ===\n")

# Check current setting
current = os.getenv('SEMANTIC_CHUNKING', 'NOT_SET')
print(f"Current SEMANTIC_CHUNKING: {current}")

if current == 'true':
    print("✓ Semantic chunking is already enabled!")
    print("\nIf you're still having accuracy issues:")
    print("1. Make sure you've re-ingested PDFs after enabling")
    print("2. Check other settings with check_accuracy_issue.py")
else:
    print("\n⚠️  SEMANTIC_CHUNKING is not enabled!")
    print("This is likely causing the accuracy issues.")
    
    print("\nTo fix this, add these lines to your .env file:")
    print("-" * 50)
    print("SEMANTIC_CHUNKING=true")
    print("CHUNK_SIZE=500")
    print("CHUNK_OVERLAP=100")
    print("RETRIEVAL_TOP_K=10")
    print("RERANK_TOP_K=4")
    print("-" * 50)
    
    print("\nThen run these commands:")
    print("1. docker-compose restart munirag")
    print("2. Re-upload all PDFs through the web interface")
    print("3. Test with your questions again")
    
print("\n=== Checking Related Settings ===")
print(f"CHUNK_SIZE: {os.getenv('CHUNK_SIZE', 'NOT_SET')}")
print(f"CHUNK_OVERLAP: {os.getenv('CHUNK_OVERLAP', 'NOT_SET')}")
print(f"TOP_K: {os.getenv('TOP_K', 'NOT_SET')}")
print(f"RETRIEVAL_TOP_K: {os.getenv('RETRIEVAL_TOP_K', 'NOT_SET')}")
print(f"RERANK_TOP_K: {os.getenv('RERANK_TOP_K', 'NOT_SET')}")

print("\nNote: These settings are read from the container's environment.")
print("Make sure your .env file is properly configured and restart the container.")