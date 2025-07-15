#!/usr/bin/env python3
"""
Comprehensive GPU diagnosis
"""

import sys
sys.path.append('/app')

import torch
import os
from sentence_transformers import SentenceTransformer
import time

print("=== COMPREHENSIVE GPU DIAGNOSIS ===\n")

# 1. Environment check
print("1. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"   EMBEDDING_MODEL: {os.environ.get('EMBEDDING_MODEL', 'not set')}")

# 2. PyTorch CUDA check
print("\n2. PyTorch CUDA Status:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    
# 3. Test direct SentenceTransformers loading
print("\n3. Testing Direct SentenceTransformers Load:")
model_name = "BAAI/bge-large-en-v1.5"

# Method 1: Let ST handle device
print(f"\n   Method 1: SentenceTransformer('{model_name}')...")
try:
    start = time.time()
    model1 = SentenceTransformer(model_name)
    load_time = time.time() - start
    print(f"   ✓ Loaded in {load_time:.1f}s")
    print(f"   Device: {model1.device}")
    
    # Test embedding
    test_texts = ["test"] * 100
    start = time.time()
    emb1 = model1.encode(test_texts, show_progress_bar=False)
    speed1 = 100 / (time.time() - start)
    print(f"   Speed: {speed1:.0f} texts/sec")
    
    # Check actual device
    for name, module in model1._modules.items():
        if hasattr(module, 'parameters'):
            try:
                param = next(module.parameters())
                print(f"   Module '{name}' on: {param.device}")
                break
            except:
                pass
    del model1
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ✗ Error: {e}")

# Method 2: Explicit device
print(f"\n   Method 2: SentenceTransformer('{model_name}', device='cuda')...")
try:
    start = time.time()
    model2 = SentenceTransformer(model_name, device='cuda')
    load_time = time.time() - start
    print(f"   ✓ Loaded in {load_time:.1f}s")
    print(f"   Device: {model2.device}")
    
    # Test embedding
    start = time.time()
    emb2 = model2.encode(test_texts, show_progress_bar=False)
    speed2 = 100 / (time.time() - start)
    print(f"   Speed: {speed2:.0f} texts/sec")
    del model2
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ✗ Error: {e}")

# Method 3: Manual move to CUDA
print(f"\n   Method 3: Manual .to('cuda') after loading...")
try:
    model3 = SentenceTransformer(model_name)
    model3 = model3.to('cuda')
    print(f"   ✓ Moved to cuda")
    
    # Test embedding
    start = time.time()
    emb3 = model3.encode(test_texts, show_progress_bar=False)
    speed3 = 100 / (time.time() - start)
    print(f"   Speed: {speed3:.0f} texts/sec")
    del model3
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ✗ Error: {e}")

# 4. Check for common issues
print("\n4. Common Issues Check:")

# Check if model files exist
model_cache = "/app/models/huggingface"
print(f"\n   Model cache directory: {model_cache}")
if os.path.exists(model_cache):
    print(f"   Contents: {os.listdir(model_cache)[:5]}...")  # First 5 items

# Check torch compile
print(f"\n   torch.compiled: {torch.cuda.is_available() and torch.version.cuda}")

# Memory check
if torch.cuda.is_available():
    print(f"\n   GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print("\n5. Recommendations:")
if 'speed1' in locals() and speed1 < 1000:
    print("   ⚠️  BGE is NOT using GPU properly!")
    print("   - Expected: 3000+ texts/sec")
    print(f"   - Actual: {speed1:.0f} texts/sec")
    print("\n   Possible fixes:")
    print("   1. Ensure NVIDIA drivers are properly installed")
    print("   2. Check if another process is using the GPU")
    print("   3. Try restarting Docker")
    print("   4. Verify CUDA version compatibility")
else:
    print("   ✅ GPU is working correctly!")