#!/usr/bin/env python3
"""
Automated accuracy testing for MuniRAG
Tests via FastAPI endpoints - no manual intervention needed
"""

import json
import requests
import time
from typing import Dict, List, Tuple
from datetime import datetime

class AutomatedAccuracyTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_questions = [
            # Weston Code of Ordinances
            {
                "id": "noise-1",
                "question": "What are the noise ordinance quiet hours in Weston?",
                "keywords": ["10 pm", "10:00", "7 am", "7:00", "quiet", "noise"],
                "critical_info": ["10", "7"],
                "min_score": 0.5
            },
            {
                "id": "permit-1", 
                "question": "What permits are required for building construction?",
                "keywords": ["building", "permit", "construction", "application"],
                "critical_info": ["permit"],
                "min_score": 0.4
            },
            {
                "id": "permit-2",
                "question": "How do I apply for a permit?",
                "keywords": ["application", "apply", "submit", "form", "online"],
                "critical_info": ["apply", "application"],
                "min_score": 0.4
            },
            {
                "id": "general-1",
                "question": "What are the penalties for ordinance violations?",
                "keywords": ["penalty", "fine", "violation", "enforcement"],
                "critical_info": ["penalty", "fine"],
                "min_score": 0.3
            },
        ]
        
    def check_api_health(self) -> bool:
        """Check if API is running"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def query_rag(self, question: str) -> Dict:
        """Send query to RAG system"""
        try:
            resp = requests.post(
                f"{self.base_url}/query",
                json={"query": question, "use_reranking": True},
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"error": f"Status {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def score_response(self, response: Dict, test_case: Dict) -> Tuple[float, Dict]:
        """Score response based on expected content"""
        if "error" in response:
            return 0.0, {"error": response["error"]}
        
        response_text = response.get("response", "").lower()
        details = {
            "found_keywords": [],
            "found_critical": [],
            "response_length": len(response_text),
            "has_sources": len(response.get("sources", [])) > 0
        }
        
        # Check keywords
        keyword_score = 0.0
        for keyword in test_case["keywords"]:
            if keyword.lower() in response_text:
                keyword_score += 1.0
                details["found_keywords"].append(keyword)
        
        # Check critical information (higher weight)
        critical_score = 0.0
        for critical in test_case["critical_info"]:
            if critical.lower() in response_text:
                critical_score += 2.0
                details["found_critical"].append(critical)
        
        # Calculate total score
        max_score = len(test_case["keywords"]) + (2 * len(test_case["critical_info"]))
        total_score = (keyword_score + critical_score) / max_score if max_score > 0 else 0.0
        
        return total_score, details
    
    def run_all_tests(self) -> Dict:
        """Run all test questions and generate report"""
        print("=== MuniRAG Automated Accuracy Test ===\n")
        
        # Check API health
        if not self.check_api_health():
            print("❌ API is not responding! Is the FastAPI server running on port 8000?")
            print("   The Streamlit app runs on 8501, but we need the API on 8000.")
            print("\nTo start the API server:")
            print("docker exec -d munirag-munirag-1 python main.py")
            return {"error": "API not available"}
        
        print("✓ API is healthy\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_questions),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "average_score": 0.0,
            "test_results": []
        }
        
        total_score = 0.0
        
        for i, test_case in enumerate(self.test_questions, 1):
            print(f"Test {i}/{len(self.test_questions)}: {test_case['question']}")
            
            # Query the system
            start_time = time.time()
            response = self.query_rag(test_case["question"])
            query_time = time.time() - start_time
            
            # Score the response
            score, details = self.score_response(response, test_case)
            total_score += score
            
            # Determine pass/fail
            if "error" in response:
                status = "ERROR"
                results["errors"] += 1
            elif score >= test_case["min_score"]:
                status = "PASS"
                results["passed"] += 1
            else:
                status = "FAIL"
                results["failed"] += 1
            
            # Store result
            test_result = {
                "test_id": test_case["id"],
                "question": test_case["question"],
                "status": status,
                "score": round(score, 3),
                "query_time": round(query_time, 2),
                "details": details
            }
            
            if status == "ERROR":
                test_result["error"] = response.get("error", "Unknown error")
            else:
                test_result["response_preview"] = response.get("response", "")[:200] + "..."
            
            results["test_results"].append(test_result)
            
            # Print summary
            if status == "PASS":
                print(f"  ✓ {status} (score: {score:.2f}, time: {query_time:.1f}s)")
            else:
                print(f"  ✗ {status} (score: {score:.2f}, time: {query_time:.1f}s)")
            
            if details.get("found_keywords"):
                print(f"    Found keywords: {', '.join(details['found_keywords'])}")
            
            print()
        
        # Calculate average
        results["average_score"] = round(total_score / len(self.test_questions), 3)
        
        # Print summary
        print("=== Test Summary ===")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ✓")
        print(f"Failed: {results['failed']} ✗")
        print(f"Errors: {results['errors']} ⚠")
        print(f"Average Score: {results['average_score']:.2f}")
        
        # Save results
        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {filename}")
        
        return results

def main():
    """Run automated accuracy tests"""
    tester = AutomatedAccuracyTester()
    
    # Check if we should use the Streamlit port
    print("Checking API availability...")
    if not tester.check_api_health():
        print("\nThe FastAPI server (port 8000) is not running.")
        print("Starting it now...")
        import subprocess
        # Start API server in background
        subprocess.Popen(["python", "main.py"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("Waiting for API to start...")
        time.sleep(5)
    
    # Run tests
    results = tester.run_all_tests()
    
    # Return error code based on results
    if results.get("errors", 0) > 0:
        exit(2)  # Errors occurred
    elif results.get("failed", 0) > 0:
        exit(1)  # Some tests failed
    else:
        exit(0)  # All tests passed

if __name__ == "__main__":
    main()