#!/usr/bin/env python3
"""
Main script to run automated accuracy tests
Can be executed without user intervention
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, '/app')

from src.automated_test_runner import AutomatedTestRunner
from src.logger import get_logger

logger = get_logger("run_accuracy_tests")


def check_prerequisites():
    """Check if system is ready for testing"""
    issues = []
    
    # Check if test questions exist
    if not os.path.exists("test_questions.json"):
        issues.append("test_questions.json not found")
    
    # Check if PDFs exist
    pdf_dir = "/app/Test-PDFs"
    if not os.path.exists(pdf_dir):
        issues.append(f"Test PDF directory {pdf_dir} not found")
    else:
        pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        if not pdfs:
            issues.append(f"No PDFs found in {pdf_dir}")
        else:
            logger.info(f"Found {len(pdfs)} test PDFs: {', '.join(pdfs)}")
    
    # Check configuration
    critical_configs = ["SEMANTIC_CHUNKING", "EMBEDDING_MODEL"]
    for config in critical_configs:
        if not os.getenv(config):
            issues.append(f"{config} not set in environment")
    
    if issues:
        logger.error("Prerequisites not met:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    return True


def print_configuration():
    """Print current configuration"""
    print("\n=== CURRENT CONFIGURATION ===")
    configs = [
        "EMBEDDING_MODEL",
        "SEMANTIC_CHUNKING", 
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "RETRIEVAL_TOP_K",
        "RERANK_TOP_K"
    ]
    
    for config in configs:
        value = os.getenv(config, "NOT SET")
        print(f"{config}: {value}")
    print()


def main():
    """Run automated accuracy tests"""
    print("=== MuniRAG Automated Accuracy Testing ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Print configuration
    print_configuration()
    
    try:
        # Initialize test runner
        print("Initializing test runner...")
        runner = AutomatedTestRunner()
        
        # Phase 1: Ingest PDFs
        print("\n=== PHASE 1: Ingesting Test PDFs ===")
        chunk_counts = runner.ingest_test_pdfs()
        
        if not chunk_counts:
            print("❌ No PDFs were ingested. Cannot proceed with testing.")
            sys.exit(1)
        
        print("\nIngestion Summary:")
        total_chunks = 0
        for pdf, count in chunk_counts.items():
            print(f"  {pdf}: {count} chunks")
            total_chunks += count
        print(f"  Total: {total_chunks} chunks")
        
        # Phase 2: Run tests
        print("\n=== PHASE 2: Running Test Suite ===")
        results = runner.run_test_suite("automated_test")
        
        # Phase 3: Display results
        print("\n=== PHASE 3: Test Results ===")
        summary = results["summary"]
        
        # Overall results
        print(f"\nOVERALL PERFORMANCE: {summary['overall_grade'].upper()}")
        print(f"Average Score: {summary['overall_average']:.1%}")
        print(f"Total Questions: {summary['total_questions']}")
        
        # Category breakdown
        print("\nPerformance by Question Category:")
        for category, score in summary['category_averages'].items():
            grade = "PASS" if score >= 0.6 else "FAIL"
            print(f"  {category:12} {score:.1%} [{grade}]")
        
        # Difficulty breakdown
        print("\nPerformance by Difficulty:")
        for difficulty, score in summary['difficulty_averages'].items():
            print(f"  {difficulty:8} {score:.1%}")
        
        # PDF breakdown
        print("\nPerformance by Document:")
        for pdf, score in summary['pdf_averages'].items():
            print(f"  {pdf[:40]:40} {score:.1%}")
        
        # Detailed results for failures
        failures = [r for r in results["test_results"] if r["score_data"]["overall_score"] < 0.6]
        if failures:
            print(f"\n⚠️  FAILED QUESTIONS ({len(failures)}):")
            for result in failures[:5]:  # Show first 5 failures
                print(f"\nQuestion: {result['question']}")
                print(f"Score: {result['score_data']['overall_score']:.1%}")
                print(f"Issue: ", end="")
                if not result['score_data']['details']['required_elements_found']:
                    print("Missing required information")
                elif result['score_data']['sub_scores']['relevance'] < 0.5:
                    print("Response not relevant to question")
                else:
                    print("Incomplete or inaccurate response")
        
        # Save detailed report
        report_file = f"test_results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("test_results", exist_ok=True)
        with open(report_file, 'w') as f:
            f.write("MuniRAG Accuracy Test Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {results['timestamp']}\n")
            f.write(f"Overall Grade: {summary['overall_grade'].upper()}\n")
            f.write(f"Overall Score: {summary['overall_average']:.1%}\n\n")
            f.write(json.dumps(results, indent=2))
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Exit code based on performance
        if summary['overall_average'] >= 0.7:
            print("\n✅ Tests PASSED - Accuracy meets threshold")
            sys.exit(0)
        else:
            print("\n❌ Tests FAILED - Accuracy below threshold")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()