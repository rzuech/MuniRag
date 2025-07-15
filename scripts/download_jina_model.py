#!/usr/bin/env python3
"""
Download and cache Jina model properly
"""

import os
os.environ['HF_HOME'] = '/app/models/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/app/models/huggingface'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/models/sentence_transformers'

print("=== Downloading Jina Model ===")

try:
    from sentence_transformers import SentenceTransformer
    
    print("1. Setting up directories...")
    os.makedirs('/app/models/huggingface', exist_ok=True)
    os.makedirs('/app/models/sentence_transformers', exist_ok=True)
    
    print("2. Downloading Jina model (this may take a few minutes)...")
    model = SentenceTransformer(
        'jinaai/jina-embeddings-v3',
        trust_remote_code=True,
        device='cpu',  # Just for download
        cache_folder='/app/models/huggingface'
    )
    
    print("3. Testing model...")
    test_embedding = model.encode(['test'], show_progress_bar=False)
    print(f"   ✓ Model works! Dimension: {len(test_embedding[0])}")
    
    print("\n✓ Jina model downloaded successfully!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTrying alternative download method...")
    
    # Alternative: Download using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        
        repo_path = snapshot_download(
            repo_id="jinaai/jina-embeddings-v3",
            cache_dir="/app/models/huggingface",
            allow_patterns=["*.py", "*.json", "*.safetensors", "*.txt"]
        )
        print(f"✓ Downloaded to: {repo_path}")
        
        # Also download the flash implementation
        flash_repo = snapshot_download(
            repo_id="jinaai/xlm-roberta-flash-implementation",
            cache_dir="/app/models/huggingface"
        )
        print(f"✓ Downloaded flash implementation to: {flash_repo}")
        
    except Exception as e2:
        print(f"✗ Alternative download also failed: {e2}")