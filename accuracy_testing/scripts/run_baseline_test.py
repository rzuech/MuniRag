#!/usr/bin/env python3
"""
Run baseline accuracy tests on ingested PDFs
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, '/app')

from src.embedder import EmbeddingModel
from src.vector_store import MultiModelVectorStore
from src.retriever import retrieve
from src.llm import stream_answer
from src.utils import build_prompt
from src.accuracy_scorer import AccuracyScorer
from src.logger import get_logger

logger = get_logger("baseline_test")


def main():
    """Run baseline accuracy tests"""
    print("=== MuniRAG Baseline Accuracy Test ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize components
    embedder = EmbeddingModel()
    vector_store = MultiModelVectorStore()
    scorer = AccuracyScorer()
    
    # Check collection status
    info = vector_store.get_collection_info()
    print(f"Collection: {info['name']}")
    print(f"Documents: {info.get('vectors_count', 'Unknown')}")
    print()
    
    # Load test questions
    with open("/app/accuracy_testing/config/test_questions.json", 'r') as f:
        test_data = json.load(f)
    
    all_results = []
    
    # Run a subset of tests
    test_count = 0
    max_tests = 10  # Limit for baseline
    
    for pdf_name, pdf_data in test_data["test_suites"].items():
        if test_count >= max_tests:
            break
            
        print(f"\nTesting {pdf_name}")
        print("-" * 50)
        
        for question_data in pdf_data["questions"][:2]:  # 2 questions per PDF
            if test_count >= max_tests:
                break
                
            test_count += 1
            print(f"\n[{test_count}] {question_data['id']}: {question_data['question']}")
            
            try:
                # Embed query
                query_embedding = embedder.embed_query(question_data["question"])
                
                # Retrieve chunks
                retrieved_chunks = retrieve(query_embedding, top_k=4)
                
                if not retrieved_chunks:
                    print("  ❌ No chunks retrieved!")
                    all_results.append({
                        "question_id": question_data["id"],
                        "question": question_data["question"],
                        "score": 0,
                        "grade": "no_retrieval",
                        "error": "No chunks retrieved",
                        "llm_response": None,
                        "response_preview": "No retrieval"
                    })
                    continue
                
                print(f"  ✓ Retrieved {len(retrieved_chunks)} chunks")
                
                # Generate response
                prompt = build_prompt(retrieved_chunks, question_data["question"])
                response_text = ""
                
                for chunk in stream_answer(prompt):
                    response_text += chunk
                
                print(f"  ✓ Generated response ({len(response_text)} chars)")
                
                # Extract sources safely
                sources = []
                for i, chunk_data in enumerate(retrieved_chunks[:3]):
                    if isinstance(chunk_data, tuple) and len(chunk_data) >= 2:
                        metadata = chunk_data[1]
                        if isinstance(metadata, dict):
                            sources.append(metadata.get("source", f"chunk_{i}"))
                        else:
                            sources.append(f"chunk_{i}")
                    else:
                        sources.append(f"chunk_{i}")
                
                # Score response
                try:
                    score_data = scorer.score_response(response_text, question_data, sources)
                    
                    print(f"  ✓ Score: {score_data['overall_score']:.1%} ({score_data['grade']})")
                    
                    # Find required elements
                    required_found = score_data['details'].get('required_elements_found', [])
                    if required_found:
                        print(f"  ✓ Found elements: {required_found}")
                    
                    all_results.append({
                        "question_id": question_data["id"],
                        "question": question_data["question"],
                        "score": score_data["overall_score"],
                        "grade": score_data["grade"],
                        "llm_response": response_text,  # Full response for audit
                        "response_preview": response_text[:200] + "...",
                        "score_details": score_data,
                        "retrieved_chunks": [
                            {
                                "content": chunk[0][:200] + "..." if len(chunk[0]) > 200 else chunk[0],
                                "metadata": chunk[1]
                            } for chunk in retrieved_chunks[:2]  # First 2 chunks for reference
                        ]
                    })
                    
                except Exception as e:
                    print(f"  ❌ Scoring error: {e}")
                    all_results.append({
                        "question_id": question_data["id"],
                        "question": question_data["question"],
                        "score": 0,
                        "grade": "scoring_error",
                        "error": str(e),
                        "llm_response": response_text,  # Still log the response
                        "response_preview": response_text[:200] + "..." if response_text else "No response"
                    })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_results.append({
                    "question_id": question_data["id"],
                    "question": question_data["question"],
                    "score": 0,
                    "grade": "error",
                    "error": str(e),
                    "llm_response": None,
                    "response_preview": "Error occurred"
                })
    
    # Calculate summary
    if all_results:
        # Filter out errors for scoring
        scored_results = [r for r in all_results if r["score"] > 0 or r["grade"] not in ["error", "scoring_error", "no_retrieval"]]
        
        if scored_results:
            avg_score = sum(r["score"] for r in scored_results) / len(scored_results)
            passing = sum(1 for r in scored_results if r["score"] >= 0.6)
        else:
            avg_score = 0
            passing = 0
        
        print("\n" + "=" * 50)
        print("BASELINE TEST SUMMARY")
        print("=" * 50)
        print(f"Total Questions Tested: {len(all_results)}")
        print(f"Successfully Scored: {len(scored_results)}")
        print(f"Average Score: {avg_score:.1%}")
        print(f"Passing (≥60%): {passing}/{len(scored_results)}")
        
        # Grade distribution
        grades = {}
        for r in all_results:
            grade = r.get("grade", "unknown")
            grades[grade] = grades.get(grade, 0) + 1
        
        print("\nGrade Distribution:")
        for grade, count in sorted(grades.items()):
            print(f"  {grade}: {count}")
        
        # Save results
        results_file = f"/app/accuracy_testing/results/baseline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("/app/accuracy_testing/results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_questions": len(all_results),
                    "scored_questions": len(scored_results),
                    "average_score": avg_score,
                    "passing_count": passing
                },
                "results": all_results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Overall assessment
        print("\nOverall Assessment:")
        if avg_score >= 0.75:
            print("✅ GOOD - System is performing well")
        elif avg_score >= 0.6:
            print("⚠️  ACCEPTABLE - System needs improvement")  
        else:
            print("❌ POOR - System has significant accuracy issues")
            
        # Show some example responses
        print("\nExample Responses (First 3):")
        print("-" * 80)
        for i, r in enumerate(all_results[:3]):
            if r.get("llm_response"):
                print(f"\n[{i+1}] Question: {r['question']}")
                print(f"Score: {r['score']:.1%} ({r['grade']})")
                print(f"Required elements found: {r.get('score_details', {}).get('details', {}).get('required_elements_found', [])}")
                print(f"\nFull LLM Response:")
                print("-" * 40)
                print(r['llm_response'])
                print("-" * 40)


if __name__ == "__main__":
    main()