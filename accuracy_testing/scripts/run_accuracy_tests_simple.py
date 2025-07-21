#!/usr/bin/env python3
"""
Run accuracy tests on already-ingested PDFs
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

logger = get_logger("accuracy_tests")


def run_tests():
    """Run accuracy tests on ingested PDFs"""
    print("=== MuniRAG Accuracy Testing ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize components
    embedder = EmbeddingModel()
    vector_store = MultiModelVectorStore()
    scorer = AccuracyScorer()
    
    # Check collection status
    info = vector_store.get_collection_info()
    print(f"Collection: {info['name']}")
    print(f"Documents: {info['vectors_count']}")
    print()
    
    if info['vectors_count'] == 0:
        print("❌ No documents in collection! Please ingest PDFs first.")
        return
    
    # Load test questions
    with open("test_questions.json", 'r') as f:
        test_data = json.load(f)
    
    all_results = []
    
    # Run tests for each PDF
    for pdf_name, pdf_data in test_data["test_suites"].items():
        print(f"\nTesting {pdf_name}")
        print("-" * 50)
        
        for question_data in pdf_data["questions"][:3]:  # Test first 3 questions per PDF
            print(f"\n{question_data['id']}: {question_data['question']}")
            
            try:
                # Embed query
                query_embedding = embedder.embed_query(question_data["question"])
                
                # Retrieve chunks
                retrieved_chunks = retrieve(query_embedding, top_k=4)
                
                if not retrieved_chunks:
                    print("  ❌ No chunks retrieved!")
                    continue
                
                print(f"  ✓ Retrieved {len(retrieved_chunks)} chunks")
                
                # Generate response
                prompt = build_prompt(retrieved_chunks, question_data["question"])
                response_text = ""
                
                for chunk in stream_answer(prompt):
                    response_text += chunk
                
                # Score response
                sources = [chunk[1].get("source", "unknown") for chunk in retrieved_chunks[:3]]
                score_data = scorer.score_response(response_text, question_data, sources)
                
                print(f"  Score: {score_data['overall_score']:.1%} ({score_data['grade']})")
                print(f"  Required elements found: {score_data['details']['required_elements_found']}")
                
                all_results.append({
                    "question_id": question_data["id"],
                    "score": score_data["overall_score"],
                    "grade": score_data["grade"],
                    "question_metadata": {
                        "category": question_data["category"],
                        "difficulty": question_data["difficulty"],
                        "pdf": pdf_name
                    },
                    "score_data": score_data
                })
                
            except Exception as e:
                import traceback
                print(f"  ❌ Error: {e}")
                # print(f"  Traceback: {traceback.format_exc()}")
                all_results.append({
                    "question_id": question_data["id"],
                    "score": 0,
                    "grade": "error",
                    "error": str(e),
                    "question_metadata": {
                        "category": question_data["category"],
                        "difficulty": question_data["difficulty"],
                        "pdf": pdf_name
                    },
                    "score_data": {
                        "overall_score": 0,
                        "grade": "error",
                        "details": {}
                    }
                })
    
    # Calculate summary
    if all_results:
        avg_score = sum(r["score"] for r in all_results) / len(all_results)
        passing = sum(1 for r in all_results if r["score"] >= 0.6)
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total Questions: {len(all_results)}")
        print(f"Average Score: {avg_score:.1%}")
        print(f"Passing: {passing}/{len(all_results)}")
        
        # Grade distribution
        grades = {}
        for r in all_results:
            grade = r.get("grade", "error")
            grades[grade] = grades.get(grade, 0) + 1
        
        print("\nGrade Distribution:")
        for grade, count in sorted(grades.items()):
            print(f"  {grade}: {count}")
        
        # Save results
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_questions": len(all_results),
                    "average_score": avg_score,
                    "passing_count": passing
                },
                "results": all_results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Overall assessment
        if avg_score >= 0.75:
            print("\n✅ GOOD - System is performing well")
        elif avg_score >= 0.6:
            print("\n⚠️  ACCEPTABLE - System needs improvement")
        else:
            print("\n❌ POOR - System has significant issues")


if __name__ == "__main__":
    run_tests()