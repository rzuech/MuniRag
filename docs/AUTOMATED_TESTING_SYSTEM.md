# MuniRAG Automated Testing System

## Overview
The automated testing system evaluates retrieval accuracy without manual intervention. It ingests PDFs, runs predefined questions, scores responses using multi-dimensional criteria, and tracks performance over time.

## Components

### 1. Test Question Bank (`test_questions.json`)
- 20+ questions across multiple PDFs
- Categories: factual, process, definitional, contextual
- Each question includes:
  - Required elements (must be present)
  - Bonus elements (improve score)
  - Wrong answer penalties
  - Difficulty level and weight

### 2. Accuracy Scorer (`src/accuracy_scorer.py`)
Multi-dimensional scoring system:
- **Factual Accuracy (40%)**: Required/bonus elements present
- **Completeness (30%)**: Addresses question fully
- **Relevance (20%)**: On-topic with source citations
- **Coherence (10%)**: Logical flow and clarity

Grades:
- Excellent: 90%+ 
- Good: 75-89%
- Acceptable: 60-74%
- Poor: 40-59%
- Failing: <40%

### 3. Test Runner (`src/automated_test_runner.py`)
- Ingests test PDFs automatically
- Runs all test questions
- Scores responses
- Stores results in Qdrant (exempt from purges)
- Generates detailed reports

### 4. Result Storage
- **Qdrant Collections** (preserved during purges):
  - `munirag_test_results`: Test run summaries
  - `munirag_accuracy_metrics`: Performance tracking
- **JSON Files**: Detailed results in `test_results/`

## Running Tests

### Quick Test
```bash
# Copy script to container and run
docker cp run_accuracy_tests.py munirag-munirag-1:/app/
docker exec munirag-munirag-1 python run_accuracy_tests.py
```

### From Inside Container
```bash
docker exec -it munirag-munirag-1 bash
cd /app
python run_accuracy_tests.py
```

### Automated (No User Intervention)
The system can run completely automated:
1. PDFs are included in Docker image (but gitignored)
2. Test questions are predefined
3. Scoring is algorithmic
4. Results are stored automatically

## Understanding Results

### Console Output
```
OVERALL PERFORMANCE: GOOD
Average Score: 78.5%
Total Questions: 20

Performance by Question Category:
  factual      82.3% [PASS]
  process      75.6% [PASS]
  definitional 71.2% [PASS]
  contextual   68.9% [PASS]

Performance by Document:
  weston-code-of-ordinances.pdf           79.8%
  How to Register and Create Permit...    76.3%
```

### Detailed Reports
JSON reports in `test_results/` include:
- Individual question scores
- Sub-score breakdowns
- Required elements found/missing
- Response times
- Configuration snapshot

## Interpreting Scores

### What Good Scores Mean
- **80%+**: System reliably finds and presents correct information
- **70-79%**: Generally good but may miss some details
- **60-69%**: Acceptable but needs improvement
- **<60%**: Significant issues with retrieval or response quality

### Common Issues and Solutions

1. **Low Factual Accuracy**
   - Enable semantic chunking: `SEMANTIC_CHUNKING=true`
   - Reduce chunk size: `CHUNK_SIZE=500`
   - Increase retrieval count: `RETRIEVAL_TOP_K=10`

2. **Low Relevance**
   - Check if reranking is enabled
   - Verify embedding model is appropriate
   - Ensure PDFs were ingested correctly

3. **Low Completeness**
   - Increase context provided to LLM
   - Check token limits in generation
   - Verify chunks contain complete information

## Continuous Monitoring

### Manual Checks
Run tests after any significant change:
- Configuration updates
- Model changes
- Code modifications

### Automated Monitoring
```python
# Run tests every hour
from src.automated_test_runner import AutomatedTestRunner
runner = AutomatedTestRunner()
runner.run_continuous_monitoring(interval_minutes=60)
```

## Adding New Tests

### 1. Add Questions to test_questions.json
```json
{
  "id": "new-test-1",
  "category": "factual",
  "question": "What is the fee for a building permit?",
  "required_elements": ["fee", "dollar", "amount"],
  "bonus_elements": ["$100", "payment", "schedule"],
  "weight": 0.8,
  "difficulty": "easy"
}
```

### 2. Add PDFs to Test-PDFs/
Place new PDFs in the Test-PDFs directory (they're gitignored)

### 3. Update Expected Scores
Adjust grade thresholds if needed based on document complexity

## Best Practices

1. **Baseline First**: Establish baseline scores before making changes
2. **Test After Changes**: Run tests after any configuration update
3. **Track Trends**: Compare scores over time to detect regression
4. **Document Context**: Note what changed between test runs
5. **Review Failures**: Analyze failed questions to understand issues

## Troubleshooting

### "No PDFs found"
- Ensure PDFs are in `/app/Test-PDFs/` in container
- Rebuild container if PDFs were added: `docker-compose build munirag`

### Low Scores
1. Check configuration with `check_accuracy_issue.py`
2. Verify PDFs ingested correctly
3. Review failed questions for patterns
4. Adjust retrieval settings

### Test Failures
- Check logs in container: `docker logs munirag-munirag-1`
- Verify services are running: `docker-compose ps`
- Ensure Qdrant is healthy

## Future Enhancements

1. **A/B Testing**: Compare configurations automatically
2. **Regression Alerts**: Notify on score drops
3. **Question Generation**: Auto-generate test questions from PDFs
4. **Cross-PDF Questions**: Test information synthesis
5. **Performance Metrics**: Track speed alongside accuracy