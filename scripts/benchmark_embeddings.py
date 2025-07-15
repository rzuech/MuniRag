#!/usr/bin/env python3
"""
Embedding Model Benchmark Script
Tests performance of all supported embedding models
"""

import sys
sys.path.append('/app')

import torch
import time
import gc
import os
import psutil
import numpy as np
from typing import List, Dict, Any
import json
from datetime import datetime

# Suppress warnings during benchmark
import warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("=== MuniRAG Embedding Model Benchmark ===\n")

# Models to test
MODELS_TO_TEST = [
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-large", 
    "intfloat/e5-large-v2",
    "jinaai/jina-embeddings-v3",
    # "hkunlp/instructor-xl",  # Commented out - requires special handling
]

# Test configurations
TEST_SIZES = [10, 100, 500, 1000]
SAMPLE_TEXT = """The City Council met on Tuesday to discuss the proposed budget amendments 
for fiscal year 2024. Key topics included infrastructure improvements, public safety 
funding, and community development initiatives. The mayor presented a comprehensive 
plan to address citizen concerns while maintaining fiscal responsibility."""

def get_system_info():
    """Get system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "model": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
        },
        "gpu": {}
    }
    
    if torch.cuda.is_available():
        info["gpu"] = {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
        }
    else:
        info["gpu"]["available"] = False
    
    return info

def benchmark_model(model_name: str, test_sizes: List[int]) -> Dict[str, Any]:
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    results = {
        "model": model_name,
        "status": "success",
        "results": {},
        "errors": []
    }
    
    try:
        # Import embedder
        from src.embedder import EmbeddingModel
        
        # Initialize model
        print("Loading model...")
        start_load = time.time()
        embedder = EmbeddingModel(model_name=model_name)
        load_time = time.time() - start_load
        
        results["load_time"] = load_time
        results["dimension"] = embedder.get_dimension()
        results["max_tokens"] = embedder.get_max_tokens()
        results["device"] = str(embedder.device)
        results["batch_size"] = embedder.batch_size
        
        print(f"✓ Model loaded in {load_time:.1f}s")
        print(f"  Dimension: {results['dimension']}")
        print(f"  Max tokens: {results['max_tokens']}")
        print(f"  Device: {results['device']}")
        print(f"  Batch size: {results['batch_size']}")
        
        # Test different sizes
        for size in test_sizes:
            print(f"\nTesting {size} texts...")
            texts = [SAMPLE_TEXT] * size
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Measure memory before
            if embedder.device == "cuda":
                mem_before = torch.cuda.memory_allocated() / (1024**3)
            else:
                mem_before = psutil.Process().memory_info().rss / (1024**3)
            
            # Time the embedding
            start_time = time.time()
            embeddings = embedder.embed_documents(texts)
            end_time = time.time()
            
            # Measure memory after
            if embedder.device == "cuda":
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / (1024**3)
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                mem_after = psutil.Process().memory_info().rss / (1024**3)
                peak_mem = mem_after
            
            # Calculate metrics
            elapsed = end_time - start_time
            texts_per_sec = size / elapsed if elapsed > 0 else 0
            mem_used = mem_after - mem_before
            
            results["results"][str(size)] = {
                "time": elapsed,
                "texts_per_second": texts_per_sec,
                "memory_used_gb": mem_used,
                "peak_memory_gb": peak_mem,
                "embeddings_shape": f"{len(embeddings)}x{len(embeddings[0])}"
            }
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Speed: {texts_per_sec:.0f} texts/sec")
            print(f"  Memory used: {mem_used:.2f}GB")
            
        # Cleanup
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(str(e))
        print(f"✗ Error: {e}")
    
    return results

def run_benchmark():
    """Run full benchmark suite"""
    # System info
    print("\n=== System Information ===")
    system_info = get_system_info()
    print(f"CPU: {system_info['cpu']['cores']} cores, {system_info['cpu']['threads']} threads")
    print(f"Memory: {system_info['memory']['total_gb']:.1f}GB total, {system_info['memory']['available_gb']:.1f}GB available")
    if system_info['gpu']['available']:
        print(f"GPU: {system_info['gpu']['device']} ({system_info['gpu']['memory_gb']:.1f}GB)")
    else:
        print("GPU: Not available")
    
    # Run benchmarks
    all_results = {
        "system": system_info,
        "models": {}
    }
    
    for model in MODELS_TO_TEST:
        results = benchmark_model(model, TEST_SIZES)
        all_results["models"][model] = results
        
        # Small delay between models
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<30} {'Device':<6} {'1000 texts/sec':<15} {'Memory (GB)':<12} {'Status'}")
    print("-" * 80)
    
    for model, results in all_results["models"].items():
        if results["status"] == "success" and "1000" in results["results"]:
            speed = results["results"]["1000"]["texts_per_second"]
            memory = results["results"]["1000"]["peak_memory_gb"]
            device = results.get("device", "unknown")
            print(f"{model:<30} {device:<6} {speed:>14.0f} {memory:>11.2f} ✓")
        else:
            print(f"{model:<30} {'N/A':<6} {'N/A':>14} {'N/A':>11} ✗ Failed")
    
    # Save results
    output_file = f"/app/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Performance recommendations
    print("\n=== Performance Analysis ===")
    if system_info['gpu']['available']:
        # Find fastest GPU model
        gpu_models = [(m, r["results"]["1000"]["texts_per_second"]) 
                      for m, r in all_results["models"].items() 
                      if r["status"] == "success" and r.get("device") == "cuda" and "1000" in r["results"]]
        
        if gpu_models:
            fastest = max(gpu_models, key=lambda x: x[1])
            print(f"\n✨ Fastest GPU model: {fastest[0]}")
            print(f"   Speed: {fastest[1]:.0f} texts/second")
            
            # Compare with Jina
            jina_results = all_results["models"].get("jinaai/jina-embeddings-v3", {})
            if jina_results.get("status") == "success" and "1000" in jina_results["results"]:
                jina_speed = jina_results["results"]["1000"]["texts_per_second"]
                speedup = fastest[1] / jina_speed if jina_speed > 0 else 0
                print(f"\n📊 {fastest[0]} is {speedup:.1f}x faster than Jina")
    
    print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()