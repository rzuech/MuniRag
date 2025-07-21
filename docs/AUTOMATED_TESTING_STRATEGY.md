# Automated Testing Strategy for MuniRAG

## Problem Statement
After recent changes, retrieval accuracy has degraded. We need automated testing to:
1. Measure retrieval accuracy objectively
2. Catch regressions before they affect users
3. Test without manual intervention
4. Ensure consistent quality across updates

## Root Cause Analysis

### Potential Accuracy Issues:
1. **SEMANTIC_CHUNKING** might be disabled (should be true)
2. **all-MiniLM-L6-v2** 512 token limit affecting chunk quality
3. **Chunk size/overlap** settings might not be optimal
4. **TOP_K** settings (retrieval vs rerank) might need tuning

## Proposed Testing Framework

### 1. Test Data Structure
Create test questions for each PDF with expected content:

```python
TEST_CASES = {
    "weston-code-of-ordinances.pdf": [
        {
            "question": "What are the noise ordinance hours in Weston?",
            "expected_keywords": ["quiet hours", "10 PM", "7 AM", "decibel"],
            "expected_concepts": ["nighttime restrictions", "residential areas"]
        },
        {
            "question": "What are the requirements for a building permit?",
            "expected_keywords": ["application", "fee", "inspection", "approval"],
            "expected_concepts": ["construction", "safety", "compliance"]
        }
    ],
    "How to Register and Create a Permit Application.pdf": [
        {
            "question": "How do I create an account for permit applications?",
            "expected_keywords": ["register", "email", "password", "account"],
            "expected_concepts": ["online portal", "user registration"]
        }
    ]
}
```

### 2. Automated Testing Components

#### A. Accuracy Testing Script (`test_accuracy.py`)
```python
"""
Automated accuracy testing for MuniRAG
Tests retrieval quality against known questions/answers
"""

import json
import requests
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

class AccuracyTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = []
        
    def test_query(self, question: str, expected: Dict) -> Dict:
        """Test a single query and score the response"""
        # Make API call (or use direct Python import)
        response = self._query_rag(question)
        
        # Score the response
        keyword_score = self._score_keywords(response, expected["expected_keywords"])
        concept_score = self._score_concepts(response, expected["expected_concepts"])
        
        return {
            "question": question,
            "response": response,
            "keyword_score": keyword_score,
            "concept_score": concept_score,
            "overall_score": (keyword_score + concept_score) / 2
        }
    
    def run_full_test_suite(self) -> Dict:
        """Run all tests and generate report"""
        # Test each PDF's questions
        # Calculate aggregate scores
        # Generate detailed report
        pass
```

#### B. Configuration Testing (`test_configs.py`)
Test different configurations to find optimal settings:
- Chunk sizes: 400, 500, 600, 800
- Overlap: 50, 100, 150
- TOP_K: 3, 4, 5, 6
- With/without semantic chunking

#### C. Regression Testing (`test_regression.py`)
Compare current performance against baseline:
- Store baseline scores
- Alert on significant degradation
- Track performance over time

### 3. Implementation Plan

#### Phase 1: Create Test Infrastructure
1. Build test question database
2. Create scoring functions
3. Implement API testing client

#### Phase 2: Baseline Establishment
1. Test current configuration
2. Document baseline scores
3. Create performance benchmarks

#### Phase 3: Optimization
1. Test configuration variations
2. Find optimal settings
3. Document best practices

#### Phase 4: CI/CD Integration
1. Run tests on every change
2. Block deployments on score regression
3. Generate performance reports

## Quick Wins for Immediate Improvement

### 1. Check Current Configuration
```bash
# Create this script: check_config.py
docker exec munirag-munirag-1 python -c "
import os
print(f'SEMANTIC_CHUNKING: {os.getenv(\"SEMANTIC_CHUNKING\", \"not set\")}')
print(f'CHUNK_SIZE: {os.getenv(\"CHUNK_SIZE\", \"not set\")}')
print(f'TOP_K: {os.getenv(\"TOP_K\", \"not set\")}')
print(f'RETRIEVAL_TOP_K: {os.getenv(\"RETRIEVAL_TOP_K\", \"not set\")}')
"
```

### 2. Enable Semantic Chunking
If SEMANTIC_CHUNKING is false, this alone could fix accuracy:
```bash
echo "SEMANTIC_CHUNKING=true" >> .env
docker-compose restart munirag
```

### 3. Adjust Retrieval Settings
Try these optimized settings:
```bash
RETRIEVAL_TOP_K=10  # Cast wider net
RERANK_TOP_K=4      # Then narrow down
CHUNK_SIZE=500      # Smaller chunks for precision
CHUNK_OVERLAP=100   # Good overlap for context
```

## Testing Without User Intervention

### Option 1: Direct Python Testing (Recommended)
- Import MuniRAG modules directly
- No network overhead
- Full access to internals
- Can test at component level

### Option 2: API Testing
- More realistic (tests full stack)
- Can use from separate container
- Easier to parallelize
- Better for load testing

### Option 3: Headless Browser Testing
- Most realistic (tests UI)
- Can catch UI-specific issues
- Slower and more complex
- Good for final validation

## Next Steps

1. **Immediate**: Check and fix configuration
2. **Today**: Create test question database
3. **Tomorrow**: Build accuracy testing framework
4. **This Week**: Establish baselines and optimize
5. **Ongoing**: Automated regression testing