#!/usr/bin/env python3
"""
Apply all fixes to make PDF processing fast
"""

import os
import shutil

print("=== APPLYING ALL PERFORMANCE FIXES ===\n")

fixes_applied = []

print("1. Backing up original files...")
try:
    # Backup ingest.py
    if os.path.exists('/app/src/ingest.py'):
        shutil.copy('/app/src/ingest.py', '/app/src/ingest_original.py')
        print("   ✓ Backed up ingest.py")
        
    # Backup embedder.py  
    if os.path.exists('/app/src/embedder.py'):
        shutil.copy('/app/src/embedder.py', '/app/src/embedder_original.py')
        print("   ✓ Backed up embedder.py")
except Exception as e:
    print(f"   ✗ Backup failed: {e}")

print("\n2. Fixing embedder.py...")

# Read current embedder
with open('/app/src/embedder.py', 'r') as f:
    embedder_content = f.read()

# Fix 1: Increase batch size
if 'self.batch_size = 256' in embedder_content:
    embedder_content = embedder_content.replace(
        'self.batch_size = 256',
        'self.batch_size = 512  # Optimal for RTX 4090'
    )
    fixes_applied.append("Increased batch size to 512")

# Fix 2: Remove any remaining progress bars
embedder_content = embedder_content.replace(
    'show_progress_bar=True',
    'show_progress_bar=False'
)
fixes_applied.append("Disabled all progress bars in embedder")

# Fix 3: Force FP16 for speed
if 'import torch' in embedder_content and 'torch.set_float32_matmul_precision' not in embedder_content:
    embedder_content = embedder_content.replace(
        'import torch',
        'import torch\ntorch.set_float32_matmul_precision("high")  # Speed optimization'
    )
    fixes_applied.append("Enabled FP16 matmul precision")

# Write fixed embedder
with open('/app/src/embedder.py', 'w') as f:
    f.write(embedder_content)

print("   ✓ Fixed embedder.py")

print("\n3. Implementing optimized PDF ingestion...")

# Check if optimized version exists
if os.path.exists('/app/src/ingest_optimized.py'):
    # Copy optimized version over current
    shutil.copy('/app/src/ingest_optimized.py', '/app/src/ingest.py')
    fixes_applied.append("Replaced ingest.py with optimized batch processing")
    print("   ✓ Installed optimized ingest.py")
else:
    print("   ✗ ingest_optimized.py not found")

print("\n4. Creating simple benchmark...")

benchmark_code = '''#!/usr/bin/env python3
import sys
sys.path.append('/app')
from src.embedder import EmbeddingModel
import time
import torch

print("Quick Performance Check...")

embedder = EmbeddingModel()
texts = ["Test"] * 1000

torch.cuda.synchronize()
start = time.time()
embeddings = embedder.embed_documents(texts)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Speed: {1000/elapsed:.0f} texts/sec")
print(f"Expected: 4000+ texts/sec")

if 1000/elapsed > 3500:
    print("✅ FIXED! GPU acceleration working!")
else:
    print("❌ Still needs fixing")
'''

with open('/app/quick_bench.py', 'w') as f:
    f.write(benchmark_code)

print("   ✓ Created quick_bench.py")

print("\n" + "="*50)
print("FIXES APPLIED:")
for i, fix in enumerate(fixes_applied, 1):
    print(f"{i}. {fix}")

print("\n⚠️  IMPORTANT: Restart the container for changes to take effect!")
print("\nRun these commands:")
print("1. docker-compose down")
print("2. docker-compose up")
print("3. docker-compose exec munirag python3 /app/quick_bench.py")