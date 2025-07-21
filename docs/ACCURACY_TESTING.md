# MuniRAG Accuracy Testing Documentation

## Overview

The MuniRAG Accuracy Testing System provides automated evaluation of the RAG pipeline's performance using real municipal documents and carefully crafted test questions. The system measures multiple dimensions of accuracy to ensure reliable answers for citizens.

## Architecture

### Components

1. **Test Questions Bank** (`test_questions.json`)
   - 20+ questions across 6 municipal PDFs
   - Categorized by type: factual, procedural, definitional, regulatory
   - Each question includes expected elements and scoring rubric

2. **Multi-Dimensional Scorer** (`src/accuracy_scorer.py`)
   - Factual Accuracy (40%): Presence of required facts
   - Completeness (30%): Coverage of all relevant information
   - Relevance (20%): Focus on the question asked
   - Coherence (10%): Logical flow and clarity

3. **Test Runners**
   - `run_baseline_test.py`: Quick baseline testing (10 questions)
   - `run_accuracy_tests_simple.py`: Full test suite
   - `automated_test_runner.py`: Production monitoring

4. **Result Storage**
   - JSON files for historical tracking
   - Qdrant collections for trend analysis (future)

## Running Tests

### Prerequisites

1. Ensure PDFs are ingested:
```bash
docker exec munirag-munirag-1 python ingest_test_pdfs.py
```

2. Verify collection has documents:
```bash
docker exec munirag-munirag-1 python check_collection_data.py
```

### Running Baseline Test

Quick 8-question test for rapid feedback:

```bash
docker exec munirag-munirag-1 python run_baseline_test.py
```

Expected output:
```
=== MuniRAG Baseline Accuracy Test ===
Started at: 2025-07-21 00:48:51

[1] weston-noise-1: What are the noise ordinance quiet hours?
  ✓ Retrieved 4 chunks
  ✓ Generated response (315 chars)
  ✓ Score: 67.0% (acceptable)
  ✓ Found elements: ['PM']

BASELINE TEST SUMMARY
Average Score: 77.9%
Overall Assessment: ✅ GOOD - System is performing well
```

### Running Full Test Suite

Complete test with all questions:

```bash
docker exec munirag-munirag-1 python run_accuracy_tests_simple.py
```

### Running Automated Tests

For production monitoring:

```bash
docker exec munirag-munirag-1 python automated_test_runner.py
```

## Understanding Results

### Score Interpretation

| Score Range | Grade | Meaning |
|------------|-------|---------|
| 90-100% | Excellent | Outstanding accuracy, ready for production |
| 75-89% | Good | Strong performance, minor improvements possible |
| 60-74% | Acceptable | Functional but needs optimization |
| 40-59% | Poor | Significant issues requiring attention |
| 0-39% | Failing | Critical problems, not suitable for use |

### Key Metrics

1. **Overall Average Score**: System-wide accuracy percentage
2. **Pass Rate**: Percentage of questions scoring ≥60%
3. **Grade Distribution**: Breakdown by performance tier
4. **Required Elements Found**: Specific facts detected in responses

### Common Issues and Solutions

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Low scores (<60%) | Poor retrieval | Adjust chunk size, enable semantic chunking |
| Missing specific facts | Chunks too large | Reduce MAX_CHUNK_TOKENS |
| Irrelevant responses | Wrong embeddings | Verify embedding model consistency |
| No chunks retrieved | Empty collection | Re-ingest PDFs |

## Test Question Structure

Each test question includes:

```json
{
  "id": "weston-noise-1",
  "category": "factual",
  "question": "What are the noise ordinance quiet hours?",
  "required_elements": ["10", "7", "PM", "AM"],
  "bonus_elements": ["10:00", "7:00", "quiet hours"],
  "wrong_answer_penalties": ["24 hours", "no restrictions"],
  "weight": 1.0,
  "difficulty": "easy"
}
```

### Categories

- **Factual**: Specific facts, numbers, dates
- **Procedural**: Step-by-step processes
- **Definitional**: What something is
- **Regulatory**: Rules and requirements

## Configuration Impact

Key settings affecting accuracy:

```bash
# In .env or docker-compose.yml
SEMANTIC_CHUNKING=true      # Better context preservation
MAX_CHUNK_TOKENS=500        # Balance detail vs relevance
TOP_K=4                     # Number of chunks to retrieve
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5  # Model consistency
```

## Continuous Monitoring

### Setting Up Automated Tests

1. Configure cron job:
```bash
# Run tests every 6 hours
0 */6 * * * docker exec munirag-munirag-1 python automated_test_runner.py
```

2. Set accuracy thresholds:
```python
# In automated_test_runner.py
ACCURACY_THRESHOLD = 0.7  # Alert if below 70%
```

3. Monitor trends:
```bash
# View historical results
docker exec munirag-munirag-1 ls -la test_results/
```

## Troubleshooting

### Debug Individual Questions

```python
# simple_test.py - Test specific questions
questions = ["Your specific question here"]
```

### Check Retrieval Quality

```bash
docker exec munirag-munirag-1 python debug_test.py
```

### Verify Embeddings

```bash
docker exec munirag-munirag-1 python check_collection_data.py
```

## Best Practices

1. **Regular Testing**: Run baseline tests after any configuration change
2. **Version Tracking**: Save test results with configuration snapshots
3. **Incremental Improvements**: Focus on lowest-scoring questions first
4. **Document Updates**: Re-test after ingesting new versions of PDFs
5. **Model Consistency**: Always use same embedding model for ingest and query

## Future Enhancements

1. **Trend Analysis**: Track accuracy over time
2. **A/B Testing**: Compare different configurations
3. **Custom Rubrics**: Department-specific scoring rules
4. **User Feedback Loop**: Incorporate real user ratings
5. **Automated Optimization**: Self-tuning based on results

## Results Storage

Test results are saved in multiple formats:

1. **JSON Files**: `baseline_test_YYYYMMDD_HHMMSS.json`
2. **Summary Reports**: `BASELINE_TEST_RESULTS.md`
3. **Qdrant Collections** (planned):
   - `munirag_test_results`: Individual test runs
   - `munirag_accuracy_metrics`: Aggregated metrics

## Integration with CI/CD

Add to your pipeline:

```yaml
# .github/workflows/accuracy-test.yml
- name: Run Accuracy Tests
  run: |
    docker exec munirag-munirag-1 python run_baseline_test.py
    # Fail if accuracy below threshold
    python check_accuracy_threshold.py --min-score 0.7
```