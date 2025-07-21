# MuniRAG Automated Accuracy Testing

## Overview
This comprehensive testing system measures retrieval accuracy objectively without manual intervention. It ingests test PDFs, executes predefined questions, scores responses algorithmically, and tracks performance trends.

## System Architecture

### Components
1. **Test Questions** (`test_questions.json`) - 20+ predefined questions
2. **Accuracy Scorer** (`src/accuracy_scorer.py`) - Multi-dimensional scoring
3. **Test Runner** (`src/automated_test_runner.py`) - Orchestrates testing
4. **Result Storage** - Qdrant collections + JSON files

### Data Flow
```
PDFs → Ingestion → Embeddings → Qdrant Storage
                                    ↓
Test Questions → Query → Retrieval → LLM Response → Scoring → Results
```

## Test Data Storage in Qdrant

### Collections Created
The testing system creates special collections that are **exempt from purging**:

1. **munirag_test_results**
   - Stores test run summaries
   - Fields: test_name, timestamp, overall_score, configuration
   - Used for tracking performance over time

2. **munirag_accuracy_metrics**
   - Stores detailed metrics per question
   - Fields: question_id, score, response_time, error_details
   - Used for deep analysis

3. **munirag_baseline_scores**
   - Stores baseline performance benchmarks
   - Used for regression detection

### Querying Past Test Results

#### Using Python
```python
from qdrant_client import QdrantClient

client = QdrantClient(host='qdrant', port=6333)

# Get all test runs
results = client.scroll(
    collection_name="munirag_test_results",
    limit=100,
    with_payload=True
)[0]

# Show test history
for result in results:
    payload = result.payload
    print(f"Test: {payload['test_name']}")
    print(f"Date: {payload['timestamp']}")
    print(f"Score: {payload['overall_score']:.1%}")
    print(f"Grade: {payload['overall_grade']}")
    print("-" * 40)
```

#### Via Docker
```bash
# List all test runs
docker exec munirag-munirag-1 python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='qdrant', port=6333)
results = client.scroll('munirag_test_results', limit=10, with_payload=True)[0]
for r in results:
    p = r.payload
    print(f\"{p['test_name']}: {p['overall_score']:.1%} ({p['timestamp'][:10]})\")
"

# Get latest test result
docker exec munirag-munirag-1 python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='qdrant', port=6333)
results = client.scroll('munirag_test_results', limit=1, with_payload=True, order_by='timestamp')[0]
if results:
    p = results[0].payload
    print(f\"Latest: {p['overall_grade']} - {p['overall_score']:.1%}\")
    print(f\"Config: {p['configuration']}\")
"
```

## Running Tests

### Prerequisites
1. **Test PDFs in container**: `/app/Test-PDFs/` directory
2. **Environment variables set**: Especially `SEMANTIC_CHUNKING=true`
3. **Services running**: Qdrant and Ollama must be available

### Quick Run (Recommended)
```bash
# Ensure PDFs are mounted in docker-compose.yml:
# volumes:
#   - ./Test-PDFs:/app/Test-PDFs

# Run tests
./quick_test_fixed.sh

# Or manually:
docker exec munirag-munirag-1 python run_accuracy_tests.py
```

### What Happens During Testing

1. **Phase 1: Ingestion**
   - Purges document collections (NOT test collections)
   - Ingests all PDFs from Test-PDFs directory
   - Reports chunk counts per PDF

2. **Phase 2: Testing**
   - Runs each test question
   - Retrieves relevant chunks
   - Generates LLM response
   - Scores using multi-dimensional criteria

3. **Phase 3: Reporting**
   - Calculates aggregate scores
   - Saves to Qdrant and JSON
   - Displays summary

### Example Output
```
=== MuniRAG Automated Accuracy Testing ===
Started at: 2025-07-20 15:30:00

=== PHASE 1: Ingesting Test PDFs ===
Found 4 test PDFs: weston-code-of-ordinances.pdf, How to Register...
Ingestion Summary:
  weston-code-of-ordinances.pdf: 14736 chunks
  Total: 15842 chunks

=== PHASE 2: Running Test Suite ===
Testing weston-code-of-ordinances.pdf (8 questions)
  weston-noise-1: good (0.75)
  weston-permit-1: excellent (0.92)

=== PHASE 3: Test Results ===
OVERALL PERFORMANCE: GOOD
Average Score: 78.5%
Total Questions: 20

Performance by Question Category:
  factual      82.3% [PASS]
  process      75.6% [PASS]
  contextual   68.9% [PASS]

✅ Tests PASSED - Accuracy meets threshold
```

## Multi-Dimensional Scoring

### Scoring Components
1. **Factual Accuracy (40%)**
   - Required elements present
   - Bonus elements found
   - No incorrect information

2. **Completeness (30%)**
   - Fully addresses question
   - Sufficient detail
   - Includes context

3. **Relevance (20%)**
   - On-topic response
   - Cites sources
   - Minimal irrelevant info

4. **Coherence (10%)**
   - Logical flow
   - Clear language
   - No contradictions

### Grade Scale
- **Excellent**: 90-100% - Production ready
- **Good**: 75-89% - Minor improvements needed
- **Acceptable**: 60-74% - Significant gaps
- **Poor**: 40-59% - Major issues
- **Failing**: <40% - System not working

## Configuration Impact

### Key Settings for Accuracy
```bash
# Most important
SEMANTIC_CHUNKING=true      # CRITICAL for accuracy
CHUNK_SIZE=500             # Smaller = more precise
CHUNK_OVERLAP=100          # Ensures continuity

# Retrieval tuning
RETRIEVAL_TOP_K=10         # Cast wider net
RERANK_TOP_K=4            # Focus on best matches
TOP_K=4                   # Final context size

# Model selection
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_MODEL=llama3.1:8b
```

## Troubleshooting

### Common Issues

#### "Test PDF directory not found"
```bash
# Check if PDFs are mounted
docker exec munirag-munirag-1 ls -la /app/Test-PDFs/

# If empty, update docker-compose.yml:
volumes:
  - ./Test-PDFs:/app/Test-PDFs
```

#### "SEMANTIC_CHUNKING not set"
```bash
# Add to .env file
echo "SEMANTIC_CHUNKING=true" >> .env
docker-compose restart munirag
```

#### Low Scores (<60%)
1. Check semantic chunking is enabled
2. Verify PDFs ingested correctly
3. Review failed questions for patterns
4. Adjust chunk size/overlap

## Continuous Improvement

### Adding Test Questions
Edit `test_questions.json`:
```json
{
  "id": "new-test-1",
  "category": "factual",
  "question": "What is the penalty for noise violations?",
  "required_elements": ["fine", "penalty", "dollar"],
  "bonus_elements": ["$500", "first offense"],
  "weight": 0.8,
  "difficulty": "medium"
}
```

### Tracking Performance
```bash
# Compare last 5 test runs
docker exec munirag-munirag-1 python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='qdrant', port=6333)
results = client.scroll('munirag_test_results', limit=5, with_payload=True)[0]
for r in sorted(results, key=lambda x: x.payload['timestamp']):
    p = r.payload
    print(f\"{p['timestamp'][:16]}: {p['overall_score']:.1%} - {p['test_name']}\")
"
```

### Best Practices
1. **Baseline First**: Run tests before any changes
2. **Test After Changes**: Verify improvements/regressions
3. **Document Context**: Note what changed between runs
4. **Monitor Trends**: Track scores over time
5. **Investigate Failures**: Understand why questions fail

## Integration with Development

### Pre-commit Testing
```bash
# Add to git hooks
#!/bin/bash
docker exec munirag-munirag-1 python run_accuracy_tests.py
if [ $? -ne 0 ]; then
    echo "Accuracy tests failed! Fix before committing."
    exit 1
fi
```

### CI/CD Integration
- Run tests on every PR
- Block merge if accuracy drops >5%
- Generate performance reports
- Track metrics in dashboard

## Future Enhancements
1. **A/B Testing**: Compare configurations automatically
2. **Question Generation**: Auto-create questions from PDFs
3. **Cross-Document**: Test information synthesis
4. **Performance Metrics**: Add speed tracking
5. **Visual Reports**: Generate charts and graphs