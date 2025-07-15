#!/usr/bin/env python3
"""
Deep diagnosis of why embeddings are still slow
Testing every possible bottleneck
"""

import sys
sys.path.append('/app')

import torch
from sentence_transformers import SentenceTransformer
import time
import numpy as np

print("=== DEEP EMBEDDING BOTTLENECK DIAGNOSIS ===\n")

# Test configuration
test_texts = ["This is a test document for municipal services analysis."] * 1000
model_name = "BAAI/bge-large-en-v1.5"

print("1. Testing raw SentenceTransformers performance...")
print("-" * 50)

# Test 1: Absolute minimal - no wrapper, no progress bar
model = SentenceTransformer(model_name, device='cuda')
print(f"Model loaded on: {model.device}")

# Warmup
_ = model.encode(['warmup'], show_progress_bar=False)

# Test with different parameters
tests = [
    {
        'name': 'Minimal (no progress, no conversion)',
        'params': {
            'show_progress_bar': False,
            'convert_to_numpy': False,
            'convert_to_tensor': 'cuda',
            'batch_size': 256
        }
    },
    {
        'name': 'No progress, numpy=True (current)',
        'params': {
            'show_progress_bar': False,
            'convert_to_numpy': True,
            'batch_size': 256
        }
    },
    {
        'name': 'With progress bar (current)',
        'params': {
            'show_progress_bar': True,
            'convert_to_numpy': False,
            'convert_to_tensor': 'cuda',
            'batch_size': 256
        }
    },
    {
        'name': 'Larger batch size (512)',
        'params': {
            'show_progress_bar': False,
            'convert_to_numpy': False,
            'convert_to_tensor': 'cuda',
            'batch_size': 512
        }
    },
    {
        'name': 'Even larger batch (1024)',
        'params': {
            'show_progress_bar': False,
            'convert_to_numpy': False,
            'convert_to_tensor': 'cuda',
            'batch_size': 1024
        }
    }
]

results = {}
for test in tests:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    start = time.time()
    embeddings = model.encode(test_texts, **test['params'])
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    speed = len(test_texts) / elapsed
    results[test['name']] = speed
    
    print(f"\n{test['name']}:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {speed:.0f} texts/sec")
    print(f"  Output type: {type(embeddings)}")

# Find the fastest configuration
fastest = max(results.items(), key=lambda x: x[1])
print(f"\n✨ Fastest: {fastest[0]} at {fastest[1]:.0f} texts/sec")

# Test 2: Check if normalization is the issue
print("\n\n2. Testing normalization impact...")
print("-" * 50)

# Disable normalization
model.normalize_embeddings = False

torch.cuda.synchronize()
start = time.time()
_ = model.encode(test_texts, show_progress_bar=False, batch_size=256, convert_to_tensor='cuda')
torch.cuda.synchronize()
no_norm_time = time.time() - start
no_norm_speed = len(test_texts) / no_norm_time

print(f"Without normalization: {no_norm_speed:.0f} texts/sec")

# Test 3: Direct transformer call (bypass SentenceTransformers wrapper)
print("\n\n3. Testing direct transformer performance...")
print("-" * 50)

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name).cuda()
transformer.eval()

# Process in batches
batch_size = 256
total_time = 0

with torch.no_grad():
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        torch.cuda.synchronize()
        start = time.time()
        
        # Forward pass
        outputs = transformer(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        torch.cuda.synchronize()
        total_time += time.time() - start

direct_speed = len(test_texts) / total_time
print(f"Direct transformer: {direct_speed:.0f} texts/sec")

# Test 4: Profile the current embedder
print("\n\n4. Testing our current embedder implementation...")
print("-" * 50)

from src.embedder import EmbeddingModel

embedder = EmbeddingModel()
print(f"Our embedder device: {embedder.device}")
print(f"Our batch size: {embedder.batch_size}")

# Test our implementation
torch.cuda.synchronize()
start = time.time()
_ = embedder.embed_documents(test_texts)
torch.cuda.synchronize()
our_time = time.time() - start
our_speed = len(test_texts) / our_time

print(f"Our implementation: {our_speed:.0f} texts/sec")

# Diagnose the issue
print("\n\n5. DIAGNOSIS:")
print("=" * 50)

if fastest[1] > 3000:
    print("✅ GPU CAN achieve 3000+ texts/sec with optimal settings")
    print(f"   Optimal config: {fastest[0]}")
    
    if our_speed < fastest[1] * 0.8:
        print(f"\n❌ Our implementation is {fastest[1]/our_speed:.1f}x slower than optimal")
        print("   Issues to fix:")
        
        if 'progress' in fastest[0].lower() and 'no progress' in fastest[0].lower():
            print("   - Remove progress bar from encode()")
        
        if 'numpy' in fastest[0].lower():
            print("   - Check numpy conversion settings")
            
        if results.get('Larger batch size (512)', 0) > results.get('With progress bar (current)', 0):
            print("   - Increase batch size to 512 or 1024")
else:
    print("❌ Even optimal settings can't reach 3000 texts/sec")
    print("   Possible issues:")
    print("   - Model is too large for GPU")
    print("   - GPU memory bandwidth limitation")
    print("   - CPU bottleneck in data preparation")

# Check for specific bottlenecks
print("\n\n6. Specific bottleneck checks:")
print("-" * 50)

# Memory bandwidth test
print("\nMemory bandwidth test:")
size = 1024
x = torch.randn(size, size).cuda()
y = torch.randn(size, size).cuda()

torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    z = x @ y
torch.cuda.synchronize()
bandwidth_time = time.time() - start
print(f"  1000 matrix multiplies: {bandwidth_time:.3f}s")

if bandwidth_time > 1.0:
    print("  ⚠️  GPU might be bandwidth limited")

# CPU preparation overhead
print("\nCPU tokenization overhead:")
start = time.time()
for i in range(0, 100, 10):
    batch = test_texts[i:i+10]
    _ = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
tokenize_time = time.time() - start
print(f"  100 texts tokenization: {tokenize_time:.3f}s")

if tokenize_time > 0.1:
    print("  ⚠️  Tokenization might be a bottleneck")