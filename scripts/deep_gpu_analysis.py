#!/usr/bin/env python3
"""
Deep GPU Performance Analysis - Entire Pipeline
"""

import sys
sys.path.append('/app')

import torch
import time
import numpy as np
from pypdf import PdfReader
import io
import gc
import traceback
from datetime import datetime

print("=== DEEP GPU PERFORMANCE ANALYSIS ===")
print(f"Time: {datetime.now()}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Import all components
from src.embedder import EmbeddingModel
from src.vector_store import VectorStore
from src.ingest import _splitter, _add_chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("\n" + "="*60)
print("PHASE 1: Component Initialization")
print("="*60)

# 1. Test embedder initialization
print("\n1.1 Embedder Initialization:")
start = time.time()
embedder = EmbeddingModel()
init_time = time.time() - start
print(f"   Initialization time: {init_time:.2f}s")
print(f"   Device: {embedder.device}")
print(f"   Batch size: {embedder.batch_size}")

# Check if model is really on GPU
if hasattr(embedder.model, '_modules'):
    for name, module in embedder.model._modules.items():
        if hasattr(module, 'parameters'):
            try:
                param = next(module.parameters())
                print(f"   Module '{name}' device: {param.device}")
                break
            except:
                pass

# 2. Test vector store
print("\n1.2 Vector Store Initialization:")
start = time.time()
vector_store = VectorStore()
vs_time = time.time() - start
print(f"   Initialization time: {vs_time:.2f}s")

print("\n" + "="*60)
print("PHASE 2: Embedding Performance Tests")
print("="*60)

# Test different batch sizes
test_sizes = [1, 10, 50, 100, 256, 512, 1000]
results = {}

for size in test_sizes:
    texts = ["This is a test document for municipal services analysis."] * size
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Time the embedding
    start = time.time()
    embeddings = embedder.embed_documents(texts)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    texts_per_sec = size / elapsed if elapsed > 0 else 0
    results[size] = {
        'time': elapsed,
        'texts_per_sec': texts_per_sec,
        'ms_per_text': (elapsed * 1000) / size
    }
    
    print(f"\nBatch size {size:>4}: {texts_per_sec:>6.0f} texts/sec ({results[size]['ms_per_text']:.2f}ms per text)")

# Find optimal batch size
optimal_size = max(results.items(), key=lambda x: x[1]['texts_per_sec'])[0]
print(f"\n✨ Optimal batch size: {optimal_size}")

print("\n" + "="*60)
print("PHASE 3: Pipeline Bottleneck Analysis")
print("="*60)

# Simulate full pipeline
print("\n3.1 Full Pipeline Test (100 chunks):")

# Generate test data
test_chunks = ["This is a test chunk of municipal document content. " * 20] * 100
test_metadata = {"source": "test.pdf", "page": 1}

pipeline_times = {}

# Step 1: Text preparation
start = time.time()
prepared_chunks = test_chunks  # In real pipeline, this might involve processing
pipeline_times['preparation'] = time.time() - start

# Step 2: Embedding generation
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
embeddings = embedder.embed_documents(test_chunks)
torch.cuda.synchronize() if torch.cuda.is_available() else None
pipeline_times['embedding'] = time.time() - start

# Step 3: Document preparation for vector store
start = time.time()
documents = []
for i, chunk in enumerate(test_chunks):
    doc = {
        "id": None,
        "content": chunk,
        "metadata": test_metadata.copy()
    }
    documents.append(doc)
pipeline_times['doc_prep'] = time.time() - start

# Step 4: Vector store insertion
start = time.time()
try:
    vector_store.add_documents(documents, embeddings)
    pipeline_times['storage'] = time.time() - start
except Exception as e:
    pipeline_times['storage'] = 0
    print(f"   Storage error: {e}")

# Report pipeline times
total_time = sum(pipeline_times.values())
print(f"\nPipeline breakdown:")
for step, step_time in pipeline_times.items():
    percentage = (step_time / total_time * 100) if total_time > 0 else 0
    print(f"   {step:<12}: {step_time:.3f}s ({percentage:>5.1f}%)")
print(f"   {'Total':<12}: {total_time:.3f}s")

print("\n" + "="*60)
print("PHASE 4: GPU Memory & Synchronization Analysis")
print("="*60)

if torch.cuda.is_available():
    # Test for unnecessary synchronizations
    print("\n4.1 Testing GPU synchronization overhead:")
    
    # Test 1: Multiple small batches (bad pattern)
    small_texts = ["test"] * 10
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = embedder.embed_documents(small_texts)
    torch.cuda.synchronize()
    multi_small_time = time.time() - start
    
    # Test 2: One large batch (good pattern)
    large_texts = ["test"] * 1000
    torch.cuda.synchronize()
    start = time.time()
    _ = embedder.embed_documents(large_texts)
    torch.cuda.synchronize()
    one_large_time = time.time() - start
    
    print(f"   100x10 texts: {multi_small_time:.2f}s ({1000/multi_small_time:.0f} texts/sec)")
    print(f"   1x1000 texts: {one_large_time:.2f}s ({1000/one_large_time:.0f} texts/sec)")
    print(f"   Overhead ratio: {multi_small_time/one_large_time:.1f}x slower with small batches")
    
    # Memory analysis
    print("\n4.2 GPU Memory Usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    print(f"   Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

print("\n" + "="*60)
print("PHASE 5: Common Performance Killers")
print("="*60)

# Check for common issues
issues_found = []

# 1. Check if progress bar is causing issues
print("\n5.1 Progress Bar Overhead Test:")
# Test with progress bar
texts = ["test"] * 500
start = time.time()
_ = embedder.model.encode(texts, show_progress_bar=True, batch_size=256)
with_progress = time.time() - start

# Test without progress bar
start = time.time()
_ = embedder.model.encode(texts, show_progress_bar=False, batch_size=256)
without_progress = time.time() - start

progress_overhead = (with_progress - without_progress) / without_progress * 100
print(f"   With progress: {500/with_progress:.0f} texts/sec")
print(f"   Without progress: {500/without_progress:.0f} texts/sec")
print(f"   Overhead: {progress_overhead:.1f}%")

if progress_overhead > 10:
    issues_found.append("Progress bar causing >10% overhead")

# 2. Check tensor conversion
print("\n5.2 Tensor Conversion Test:")
test_tensor = torch.randn(1000, 1024).cuda()

# GPU to CPU
torch.cuda.synchronize()
start = time.time()
cpu_array = test_tensor.cpu().numpy()
torch.cuda.synchronize()
gpu_to_cpu_time = time.time() - start

# Direct numpy operation (for comparison)
cpu_tensor = torch.randn(1000, 1024)
start = time.time()
cpu_array2 = cpu_tensor.numpy()
cpu_to_cpu_time = time.time() - start

print(f"   GPU→CPU conversion: {gpu_to_cpu_time*1000:.2f}ms")
print(f"   CPU→CPU conversion: {cpu_to_cpu_time*1000:.2f}ms")
print(f"   Overhead: {gpu_to_cpu_time/cpu_to_cpu_time:.1f}x")

# 3. Check for blocking operations
print("\n5.3 Checking for blocking operations in pipeline...")
# This would require inspecting the actual ingest code

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print("\n🎯 Performance Optimizations:")

# Based on results
if optimal_size > embedder.batch_size:
    print(f"1. Increase batch size from {embedder.batch_size} to {optimal_size}")

if 'multi_small_time' in locals() and multi_small_time/one_large_time > 2:
    print("2. Batch chunks before embedding (avoid many small calls)")

if progress_overhead > 10:
    print("3. Disable progress bars or update less frequently")

print("4. Ensure all tensor operations stay on GPU until final conversion")
print("5. Use larger PDF chunk sizes to reduce total chunks")
print("6. Consider streaming pipeline to overlap I/O and computation")

print("\n📊 Expected Performance:")
print(f"   Current: ~{results[256]['texts_per_sec']:.0f} texts/sec")
print(f"   Optimal: ~{max(r['texts_per_sec'] for r in results.values()):.0f} texts/sec")
print(f"   Target: 4000+ texts/sec for RTX 4090")

if issues_found:
    print(f"\n⚠️  Issues Found:")
    for issue in issues_found:
        print(f"   - {issue}")