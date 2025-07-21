# Automated Web-Based Accuracy Testing Plan

## Overview
MuniRAG has a FastAPI backend (port 8000) that we can use for automated testing without needing to interact with Streamlit's complex WebSocket protocol.

## Available Endpoints

1. **POST /ingest/pdf** - Upload PDFs
2. **POST /query** - Ask questions
3. **GET /health** - Check service status

## Implementation Strategy

### Phase 1: Create Test Framework

#### 1. Test Data Structure (`test_data/qa_pairs.json`)
```json
{
  "weston-code-of-ordinances.pdf": {
    "questions": [
      {
        "id": "weston-1",
        "question": "What are the noise ordinance quiet hours?",
        "expected_keywords": ["10 pm", "10:00 pm", "7 am", "7:00 am", "quiet hours"],
        "expected_phrases": ["between 10:00 p.m. and 7:00 a.m."],
        "min_score": 0.7
      },
      {
        "id": "weston-2", 
        "question": "What permits are required for construction?",
        "expected_keywords": ["building permit", "construction", "application"],
        "expected_phrases": ["building permit required"],
        "min_score": 0.6
      }
    ]
  },
  "How to Register and Create a Permit Application.pdf": {
    "questions": [
      {
        "id": "permit-1",
        "question": "How do I create an account for permits?",
        "expected_keywords": ["register", "account", "email", "password"],
        "expected_phrases": ["create an account", "registration"],
        "min_score": 0.7
      }
    ]
  }
}
```

#### 2. Automated Test Runner (`automated_accuracy_test.py`)
```python
#!/usr/bin/env python3
"""
Automated accuracy testing via FastAPI endpoints
Can run without user intervention
"""

import json
import requests
import time
from typing import Dict, List
from datetime import datetime

class AutomatedAccuracyTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    def check_health(self) -> bool:
        """Verify API is running"""
        try:
            resp = requests.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except:
            return False
    
    def query_rag(self, question: str) -> Dict:
        """Send query to RAG system"""
        resp = requests.post(
            f"{self.base_url}/query",
            json={"query": question, "use_reranking": True}
        )
        return resp.json()
    
    def score_response(self, response: str, expected: Dict) -> float:
        """Score response based on expected content"""
        response_lower = response.lower()
        score = 0.0
        max_score = 0.0
        
        # Check keywords (partial credit)
        for keyword in expected["expected_keywords"]:
            max_score += 1.0
            if keyword.lower() in response_lower:
                score += 1.0
        
        # Check phrases (higher weight)
        for phrase in expected.get("expected_phrases", []):
            max_score += 2.0
            if phrase.lower() in response_lower:
                score += 2.0
        
        return score / max_score if max_score > 0 else 0.0
    
    def run_test_suite(self, qa_file="test_data/qa_pairs.json") -> Dict:
        """Run all tests and generate report"""
        with open(qa_file, 'r') as f:
            test_data = json.load(f)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "passed": 0,
            "failed": 0,
            "average_score": 0.0,
            "details": []
        }
        
        for pdf, data in test_data.items():
            for question_data in data["questions"]:
                results["total_questions"] += 1
                
                # Query the system
                response = self.query_rag(question_data["question"])
                
                # Score the response
                score = self.score_response(
                    response["response"], 
                    question_data
                )
                
                passed = score >= question_data["min_score"]
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["details"].append({
                    "pdf": pdf,
                    "question_id": question_data["id"],
                    "question": question_data["question"],
                    "score": score,
                    "passed": passed,
                    "response_preview": response["response"][:200]
                })
        
        results["average_score"] = sum(d["score"] for d in results["details"]) / len(results["details"])
        return results
```

### Phase 2: Continuous Testing

#### 1. Scheduled Test Runner (`run_scheduled_tests.py`)
```python
#!/usr/bin/env python3
"""Run tests periodically and alert on regression"""

import schedule
import time
from automated_accuracy_test import AutomatedAccuracyTester

def run_tests():
    tester = AutomatedAccuracyTester()
    
    if not tester.check_health():
        print("API not available!")
        return
        
    results = tester.run_test_suite()
    
    # Save results
    with open(f"test_results_{results['timestamp']}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Check for regression
    if results["average_score"] < 0.7:
        print(f"⚠️ ACCURACY REGRESSION: {results['average_score']:.2f}")
    else:
        print(f"✓ Tests passed: {results['average_score']:.2f}")

# Run every hour
schedule.every(1).hours.do(run_tests)

# Run once immediately
run_tests()

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Phase 3: Integration with Development

#### 1. Pre-commit Testing
Add to git hooks to prevent accuracy regression

#### 2. Docker Compose Integration
```yaml
# Add test runner service
test-runner:
  build: .
  command: python run_scheduled_tests.py
  volumes:
    - ./test_results:/app/test_results
  depends_on:
    - munirag
```

## How Claude Can Run This Independently

### Option 1: Direct Python Execution
```bash
# Copy test files to container
docker cp automated_accuracy_test.py munirag-munirag-1:/app/
docker cp test_data munirag-munirag-1:/app/

# Run tests
docker exec munirag-munirag-1 python automated_accuracy_test.py
```

### Option 2: External Test Container
```bash
# Create a separate test container that hits the API
docker run --network munirag_default -v $(pwd)/tests:/tests python:3.10 python /tests/run_tests.py
```

### Option 3: GitHub Actions
Automate testing on every push

## Benefits

1. **No Manual Intervention**: Tests run automatically
2. **Objective Metrics**: Scoring based on expected content
3. **Regression Detection**: Alerts when accuracy drops
4. **Historical Tracking**: Results saved with timestamps
5. **Easy Extension**: Just add more Q&A pairs

## Next Steps

1. Create initial Q&A test data
2. Implement basic test runner
3. Run baseline tests
4. Set up continuous testing
5. Add more sophisticated scoring