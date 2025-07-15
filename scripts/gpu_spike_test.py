#!/usr/bin/env python3
"""
Force GPU to spike - prove it can work
"""

import torch
import time
import subprocess

print("=== GPU SPIKE TEST ===")
print("This WILL make your GPU spike to prove it's working\n")

def get_gpu_usage():
    """Get GPU usage from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return -1

print("1. Heavy matrix multiplication (should spike to 90%+)")
print("-" * 50)

# Create large matrices
size = 8192
print(f"Creating {size}x{size} matrices on GPU...")
x = torch.randn(size, size, device='cuda', dtype=torch.float32)
y = torch.randn(size, size, device='cuda', dtype=torch.float32)

print("Starting computation...\n")
print("GPU Usage:")

# Do heavy computation
for i in range(10):
    gpu_before = get_gpu_usage()
    
    torch.cuda.synchronize()
    start = time.time()
    
    # Heavy computation
    z = torch.matmul(x, y)
    x = z  # Reuse result
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    gpu_after = get_gpu_usage()
    
    print(f"  Iteration {i+1}: {gpu_before}% → {gpu_after}% (took {elapsed:.2f}s)")
    
    if gpu_after > 80:
        print("  ✅ GPU is spiking properly!")

print("\n2. Testing embeddings with forced GPU computation")
print("-" * 50)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# FORCE model to GPU
model = model.cuda()
model.eval()

# Generate very large batch
texts = ["This is a test document."] * 10000
print(f"Encoding {len(texts)} texts...")

# Monitor GPU
gpu_samples = []
import threading

def monitor():
    for _ in range(20):
        gpu_samples.append(get_gpu_usage())
        time.sleep(0.5)

thread = threading.Thread(target=monitor)
thread.start()

# Force GPU computation
with torch.cuda.amp.autocast():  # Force GPU mode
    torch.cuda.synchronize()
    start = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=1024,  # Large batch
        show_progress_bar=False,
        convert_to_tensor='cuda',  # Force CUDA
        device='cuda'  # Explicit device
    )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start

thread.join()

print(f"\nResults:")
print(f"  Time: {elapsed:.2f}s")
print(f"  Speed: {len(texts)/elapsed:.0f} texts/sec")
print(f"  GPU usage during encoding: {max(gpu_samples) if gpu_samples else 0}%")

if max(gpu_samples) < 50:
    print("\n❌ Even forced GPU mode isn't using GPU properly!")
    print("   This suggests a driver or CUDA issue")
else:
    print("\n✅ GPU can be utilized when forced")

print("\n3. Direct CUDA kernel test")
print("-" * 50)

# Custom CUDA operation
def gpu_stress():
    x = torch.randn(10000, 10000, device='cuda')
    for _ in range(100):
        x = torch.nn.functional.relu(x)
        x = x * 2.0 + 1.0
        x = torch.tanh(x)
    return x

print("Running CUDA kernels...")
gpu_before = get_gpu_usage()

torch.cuda.synchronize()
start = time.time()
result = gpu_stress()
torch.cuda.synchronize()
elapsed = time.time() - start

gpu_after = get_gpu_usage()

print(f"GPU usage: {gpu_before}% → {gpu_after}%")
print(f"Time: {elapsed:.2f}s")

print("\n" + "="*50)
print("SUMMARY:")

if gpu_after > 50:
    print("✅ Your GPU CAN spike - the issue is in the embedding pipeline")
else:
    print("❌ GPU never spikes - possible driver/CUDA issue")