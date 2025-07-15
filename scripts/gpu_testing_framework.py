#!/usr/bin/env python3
"""
GPU Testing Framework for Production Deployment
Tests all critical paths and provides benchmarks
"""

import sys
sys.path.append('/app')

import torch
import time
import json
import os
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import gc

class GPUTestFramework:
    def __init__(self):
        self.results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def test_gpu_availability(self) -> bool:
        """Test 1: Basic GPU availability"""
        print("\n[TEST 1] GPU Availability")
        
        if not torch.cuda.is_available():
            self.log_fail("No CUDA device available")
            return False
            
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ Memory: {gpu_memory:.1f}GB")
        
        self.results['gpu_info'] = {
            'name': gpu_name,
            'memory_gb': gpu_memory,
            'cuda_version': torch.version.cuda
        }
        
        self.log_pass("GPU available and accessible")
        return True
        
    def test_embedding_performance(self) -> Dict:
        """Test 2: Embedding model performance"""
        print("\n[TEST 2] Embedding Performance")
        
        from src.embedder import EmbeddingModel
        
        try:
            # Initialize
            embedder = EmbeddingModel()
            
            # Test various batch sizes
            batch_sizes = [1, 10, 50, 100, 256, 512, 1000]
            perf_results = {}
            
            for size in batch_sizes:
                texts = ["Test document content"] * size
                
                torch.cuda.synchronize()
                start = time.time()
                _ = embedder.embed_documents(texts)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                texts_per_sec = size / elapsed
                perf_results[size] = texts_per_sec
                
                print(f"  Batch {size:>4}: {texts_per_sec:>6.0f} texts/sec")
            
            # Check if performance meets minimum
            max_perf = max(perf_results.values())
            if max_perf < 2000:
                self.log_fail(f"Performance too low: {max_perf:.0f} texts/sec (expected >2000)")
            else:
                self.log_pass(f"Performance acceptable: {max_perf:.0f} texts/sec")
                
            self.results['embedding_performance'] = perf_results
            
        except Exception as e:
            self.log_fail(f"Embedding test failed: {str(e)}")
            
    def test_memory_management(self) -> None:
        """Test 3: GPU memory management"""
        print("\n[TEST 3] Memory Management")
        
        if self.device != "cuda":
            self.log_skip("No GPU available")
            return
            
        from src.embedder import EmbeddingModel
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        initial_memory = torch.cuda.memory_allocated() / 1e9
        
        # Load model
        embedder = EmbeddingModel()
        model_memory = torch.cuda.memory_allocated() / 1e9 - initial_memory
        
        # Process large batch
        large_batch = ["Test text"] * 5000
        _ = embedder.embed_documents(large_batch)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"  Model size: {model_memory:.2f}GB")
        print(f"  Peak usage: {peak_memory:.2f}GB")
        
        if peak_memory > 10:
            self.log_fail(f"Excessive memory usage: {peak_memory:.2f}GB")
        else:
            self.log_pass(f"Memory usage acceptable: {peak_memory:.2f}GB")
            
        self.results['memory_usage'] = {
            'model_gb': model_memory,
            'peak_gb': peak_memory
        }
        
    def test_pipeline_bottlenecks(self) -> None:
        """Test 4: Full pipeline performance"""
        print("\n[TEST 4] Pipeline Bottlenecks")
        
        from src.embedder import EmbeddingModel
        from src.vector_store import VectorStore
        
        embedder = EmbeddingModel()
        vector_store = VectorStore()
        
        # Simulate different ingestion patterns
        patterns = {
            'single_page': (1, 10),      # 1 page, 10 chunks each
            'small_pdf': (10, 10),       # 10 pages, 10 chunks each
            'large_pdf': (100, 10),      # 100 pages, 10 chunks each
            'huge_pdf': (500, 10)        # 500 pages, 10 chunks each
        }
        
        pattern_results = {}
        
        for pattern_name, (pages, chunks_per_page) in patterns.items():
            print(f"\n  Testing {pattern_name}: {pages} pages, {chunks_per_page} chunks/page")
            
            # Method 1: Page-by-page (current implementation)
            torch.cuda.synchronize()
            start = time.time()
            
            for page in range(pages):
                chunks = ["Test chunk"] * chunks_per_page
                embeddings = embedder.embed_documents(chunks)
                # Simulate vector store add
                
            torch.cuda.synchronize()
            page_by_page_time = time.time() - start
            
            # Method 2: Batch all chunks (optimized)
            all_chunks = ["Test chunk"] * (pages * chunks_per_page)
            
            torch.cuda.synchronize()
            start = time.time()
            
            embeddings = embedder.embed_documents(all_chunks)
            
            torch.cuda.synchronize()
            batch_time = time.time() - start
            
            speedup = page_by_page_time / batch_time
            texts_per_sec_current = (pages * chunks_per_page) / page_by_page_time
            texts_per_sec_optimal = (pages * chunks_per_page) / batch_time
            
            print(f"    Current: {texts_per_sec_current:>6.0f} texts/sec ({page_by_page_time:.2f}s)")
            print(f"    Optimal: {texts_per_sec_optimal:>6.0f} texts/sec ({batch_time:.2f}s)")
            print(f"    Speedup: {speedup:.1f}x")
            
            pattern_results[pattern_name] = {
                'current_speed': texts_per_sec_current,
                'optimal_speed': texts_per_sec_optimal,
                'speedup_potential': speedup
            }
            
        self.results['pipeline_patterns'] = pattern_results
        
        # Check if current implementation is optimal
        avg_speedup = np.mean([r['speedup_potential'] for r in pattern_results.values()])
        if avg_speedup > 2:
            self.log_fail(f"Pipeline not optimized: {avg_speedup:.1f}x speedup possible")
        else:
            self.log_pass("Pipeline reasonably optimized")
            
    def test_concurrent_operations(self) -> None:
        """Test 5: Concurrent GPU operations"""
        print("\n[TEST 5] Concurrent Operations")
        
        if self.device != "cuda":
            self.log_skip("No GPU available")
            return
            
        # Test if other operations block GPU
        from src.embedder import EmbeddingModel
        import threading
        
        embedder = EmbeddingModel()
        
        def gpu_operation():
            """Simulate concurrent GPU usage"""
            x = torch.randn(1000, 1000).cuda()
            for _ in range(100):
                x = torch.matmul(x, x.T)
            return x
            
        # Test embedding during other GPU operations
        texts = ["Test"] * 1000
        
        # Baseline
        torch.cuda.synchronize()
        start = time.time()
        _ = embedder.embed_documents(texts)
        torch.cuda.synchronize()
        baseline_time = time.time() - start
        
        # With concurrent operation
        thread = threading.Thread(target=gpu_operation)
        thread.start()
        
        torch.cuda.synchronize()
        start = time.time()
        _ = embedder.embed_documents(texts)
        torch.cuda.synchronize()
        concurrent_time = time.time() - start
        
        thread.join()
        
        slowdown = concurrent_time / baseline_time
        
        print(f"  Baseline: {1000/baseline_time:.0f} texts/sec")
        print(f"  Concurrent: {1000/concurrent_time:.0f} texts/sec")
        print(f"  Slowdown: {slowdown:.2f}x")
        
        if slowdown > 2:
            self.log_fail(f"Severe performance degradation with concurrent ops: {slowdown:.2f}x")
        else:
            self.log_pass("Acceptable concurrent performance")
            
    def generate_report(self) -> None:
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("GPU PERFORMANCE TEST REPORT")
        print("="*60)
        
        print(f"\nTest Summary:")
        print(f"  Passed: {self.passed_tests}")
        print(f"  Failed: {self.failed_tests}")
        print(f"  Total: {self.passed_tests + self.failed_tests}")
        
        # Save detailed results
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'passed': self.passed_tests,
                'failed': self.failed_tests,
                'device': self.device
            },
            'results': self.results
        }
        
        with open('/app/gpu_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\nDetailed report saved to: /app/gpu_test_report.json")
        
        # Production readiness check
        print("\n🚀 PRODUCTION READINESS:")
        
        if self.failed_tests == 0:
            print("✅ System is ready for production deployment!")
        else:
            print("❌ System has issues that need fixing before deployment")
            
        # Performance recommendations
        if 'embedding_performance' in self.results:
            max_perf = max(self.results['embedding_performance'].values())
            print(f"\n📊 Expected PDF Processing Performance:")
            print(f"   50-page PDF: ~{50*10/max_perf*60:.1f} seconds")
            print(f"   500-page PDF: ~{500*10/max_perf*60:.1f} seconds")
            
    def log_pass(self, message: str):
        print(f"  ✅ PASS: {message}")
        self.passed_tests += 1
        
    def log_fail(self, message: str):
        print(f"  ❌ FAIL: {message}")
        self.failed_tests += 1
        
    def log_skip(self, message: str):
        print(f"  ⏭️  SKIP: {message}")
        

def run_all_tests():
    """Run comprehensive GPU testing suite"""
    tester = GPUTestFramework()
    
    # Run all tests
    tester.test_gpu_availability()
    tester.test_embedding_performance()
    tester.test_memory_management()
    tester.test_pipeline_bottlenecks()
    tester.test_concurrent_operations()
    
    # Generate report
    tester.generate_report()
    

if __name__ == "__main__":
    print("Starting GPU Performance Testing Framework...")
    print("This will take 2-3 minutes to complete.\n")
    
    run_all_tests()