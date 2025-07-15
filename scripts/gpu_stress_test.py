#!/usr/bin/env python3
"""
GPU Stress Test - Force high GPU utilization
"""

import sys
sys.path.append('/app')

import torch
import time
import nvidia_ml_py3 as nvml
from sentence_transformers import SentenceTransformer

print("=== GPU STRESS TEST ===\n")

# Initialize NVML for GPU monitoring
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_stats():
    """Get current GPU utilization"""
    util = nvml.nvmlDeviceGetUtilizationRates(handle)
    mem = nvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        'gpu_util': util.gpu,
        'mem_util': util.memory,
        'mem_used_gb': mem.used / 1e9,
        'mem_total_gb': mem.total / 1e9
    }

print("1. Testing raw GPU compute (should spike to 90%+)")
print("-" * 50)

# Create large tensors
size = 4096
x = torch.randn(size, size).cuda()
y = torch.randn(size, size).cuda()

print("Starting matrix multiplication stress test...")
print("Watch GPU utilization - should spike to 90%+\n")

# Monitor GPU while computing
for i in range(5):
    start_stats = get_gpu_stats()
    
    # Heavy computation
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(50):
        z = torch.matmul(x, y)
        x = z
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    end_stats = get_gpu_stats()
    
    print(f"Iteration {i+1}:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  GPU util: {start_stats['gpu_util']}% → {end_stats['gpu_util']}%")
    print(f"  Memory: {end_stats['mem_used_gb']:.1f}/{end_stats['mem_total_gb']:.1f} GB")

print("\n2. Testing SentenceTransformers GPU usage")
print("-" * 50)

# Load model
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')

# Generate large batch
texts = ["This is a test document for GPU stress testing."] * 5000

print("Encoding 5000 texts with different settings...\n")

# Test 1: Optimal settings
print("Test 1: Optimal (batch_size=512, no progress)")
start_stats = get_gpu_stats()

torch.cuda.synchronize()
start = time.time()
_ = model.encode(texts, batch_size=512, show_progress_bar=False, convert_to_tensor='cuda')
torch.cuda.synchronize()
elapsed = time.time() - start

end_stats = get_gpu_stats()
print(f"  Time: {elapsed:.2f}s ({5000/elapsed:.0f} texts/sec)")
print(f"  GPU util during encoding: {end_stats['gpu_util']}%")
print(f"  Memory: {end_stats['mem_used_gb']:.1f} GB")

# Test 2: With progress bar
print("\nTest 2: With progress bar (like our implementation)")
from tqdm import tqdm

start_stats = get_gpu_stats()

# Monitor during encoding
gpu_samples = []
def monitor_gpu():
    """Sample GPU usage during encoding"""
    import threading
    def sample():
        for _ in range(20):
            stats = get_gpu_stats()
            gpu_samples.append(stats['gpu_util'])
            time.sleep(0.1)
    
    thread = threading.Thread(target=sample)
    thread.start()
    return thread

monitor_thread = monitor_gpu()

torch.cuda.synchronize()
start = time.time()
_ = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_tensor='cuda')
torch.cuda.synchronize()
elapsed = time.time() - start

monitor_thread.join()

avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
max_gpu = max(gpu_samples) if gpu_samples else 0

print(f"  Time: {elapsed:.2f}s ({5000/elapsed:.0f} texts/sec)")
print(f"  Average GPU util: {avg_gpu:.0f}%")
print(f"  Peak GPU util: {max_gpu:.0f}%")

print("\n3. Testing our embedder implementation")
print("-" * 50)

from src.embedder import EmbeddingModel

embedder = EmbeddingModel()
texts_1000 = texts[:1000]

print("Encoding 1000 texts with our embedder...")

# Clear GPU samples
gpu_samples = []
monitor_thread = monitor_gpu()

torch.cuda.synchronize()
start = time.time()
_ = embedder.embed_documents(texts_1000)
torch.cuda.synchronize()
elapsed = time.time() - start

monitor_thread.join()

avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
max_gpu = max(gpu_samples) if gpu_samples else 0

print(f"  Time: {elapsed:.2f}s ({1000/elapsed:.0f} texts/sec)")
print(f"  Average GPU util: {avg_gpu:.0f}%")
print(f"  Peak GPU util: {max_gpu:.0f}%")

print("\n4. DIAGNOSIS")
print("=" * 50)

if max_gpu < 50:
    print("❌ GPU is severely underutilized!")
    print("   Something is blocking GPU computation")
    print("   Likely causes:")
    print("   - Progress bar forcing synchronization")
    print("   - Small batch processing")
    print("   - CPU bottleneck in data preparation")
else:
    print("✅ GPU utilization is good")

# Cleanup
nvml.nvmlShutdown()