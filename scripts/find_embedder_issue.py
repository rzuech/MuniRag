#!/usr/bin/env python3
"""
Find the exact issue in our embedder vs raw SentenceTransformers
"""

import sys
sys.path.append('/app')

import torch
from sentence_transformers import SentenceTransformer
import time
import nvidia_ml_py3 as nvml

print("=== EMBEDDER ISSUE FINDER ===\n")

# Initialize GPU monitoring
try:
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_gpu_util():
        return nvml.nvmlDeviceGetUtilizationRates(handle).gpu
except:
    def get_gpu_util():
        return -1

texts = ["Test document for municipal services."] * 1000

print("1. Raw SentenceTransformers (should be fast)")
print("-" * 50)

# Load model directly
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')

# Test optimal settings
torch.cuda.synchronize()
gpu_before = get_gpu_util()
start = time.time()

embeddings = model.encode(
    texts,
    batch_size=512,  # Optimal from diagnostic
    show_progress_bar=False,
    convert_to_numpy=False,
    convert_to_tensor='cuda'
)

torch.cuda.synchronize()
elapsed = time.time() - start
gpu_after = get_gpu_util()

print(f"Speed: {1000/elapsed:.0f} texts/sec")
print(f"GPU util: {gpu_before}% → {gpu_after}%")
print(f"Output type: {type(embeddings)}")
print(f"Output device: {embeddings.device if hasattr(embeddings, 'device') else 'N/A'}")

print("\n2. Our embedder implementation")
print("-" * 50)

from src.embedder import EmbeddingModel

# Check what's happening in our embedder
embedder = EmbeddingModel()

print(f"Model name: {embedder.model_name}")
print(f"Device: {embedder.device}")
print(f"Batch size: {embedder.batch_size}")

# Check model location
if hasattr(embedder.model, '_modules'):
    for name, module in embedder.model._modules.items():
        if hasattr(module, 'parameters'):
            try:
                param = next(module.parameters())
                print(f"Module '{name}' on: {param.device}")
                break
            except:
                pass

# Test our implementation
torch.cuda.synchronize()
gpu_before = get_gpu_util()
start = time.time()

embeddings = embedder.embed_documents(texts)

torch.cuda.synchronize()
elapsed = time.time() - start
gpu_after = get_gpu_util()

print(f"\nSpeed: {1000/elapsed:.0f} texts/sec")
print(f"GPU util: {gpu_before}% → {gpu_after}%")
print(f"Output type: {type(embeddings)}")

print("\n3. Testing step by step what's different")
print("-" * 50)

# Test our model directly
print("\nDirect test of embedder.model.encode():")

torch.cuda.synchronize()
start = time.time()

# Call encode directly on our model
embeddings_direct = embedder.model.encode(
    texts,
    batch_size=512,  # Use optimal
    show_progress_bar=False,
    convert_to_numpy=False,
    convert_to_tensor='cuda'
)

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Speed: {1000/elapsed:.0f} texts/sec")

# Check if it's the prepare_texts function
print("\n4. Testing text preparation overhead")
print("-" * 50)

prepared_texts = embedder._prepare_texts_for_embedding(texts, is_query=False)
print(f"Original texts: {len(texts)}")
print(f"Prepared texts: {len(prepared_texts)}")
print(f"Text changed: {texts[0] != prepared_texts[0]}")

# Test with prepared texts
torch.cuda.synchronize()
start = time.time()

embeddings = embedder.model.encode(
    prepared_texts,
    batch_size=embedder.batch_size,
    show_progress_bar=False,
    convert_to_numpy=False,
    convert_to_tensor='cuda'
)

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Speed with prepared texts: {1000/elapsed:.0f} texts/sec")

print("\n5. GPU Utilization Test")
print("-" * 50)

# Monitor GPU during large batch
print("Processing 5000 texts to check GPU usage...")
large_texts = texts * 5

# Sample GPU usage during encoding
gpu_samples = []
import threading

def monitor_gpu():
    for _ in range(50):
        gpu_samples.append(get_gpu_util())
        time.sleep(0.1)

monitor_thread = threading.Thread(target=monitor_gpu)
monitor_thread.start()

# Process large batch
torch.cuda.synchronize()
start = time.time()

_ = model.encode(
    large_texts,
    batch_size=512,
    show_progress_bar=False,
    convert_to_tensor='cuda'
)

torch.cuda.synchronize()
elapsed = time.time() - start

monitor_thread.join()

avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
max_gpu = max(gpu_samples) if gpu_samples else 0

print(f"Speed: {5000/elapsed:.0f} texts/sec")
print(f"Average GPU: {avg_gpu:.0f}%")
print(f"Peak GPU: {max_gpu:.0f}%")

if max_gpu < 50:
    print("\n❌ GPU is not being utilized properly!")
    print("   The computation is likely happening on CPU")

# Cleanup
try:
    nvml.nvmlShutdown()
except:
    pass