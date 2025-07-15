#!/usr/bin/env python3
"""Fix all vector store imports to use v2"""

import os
import re

files_to_fix = [
    '/app/src/ingest.py',
    '/app/src/rag_pipeline.py',
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        print(f"Fixing {filepath}...")
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace imports
        content = content.replace('from src.vector_store import VectorStore', 
                                'from src.vector_store_v2 import MultiModelVectorStore')
        
        # Replace instantiation
        content = re.sub(r'vector_store\s*=\s*VectorStore\(\)', 
                        'embedder = EmbeddingModel()\n    vector_store = MultiModelVectorStore(embedder.model_name)', 
                        content)
        
        # Fix any standalone VectorStore() calls
        content = content.replace('VectorStore()', 'MultiModelVectorStore()')
        
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Fixed")
    else:
        print(f"  ✗ File not found: {filepath}")

print("\nDone!")