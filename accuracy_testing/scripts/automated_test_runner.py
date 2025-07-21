"""
Automated test runner for MuniRAG accuracy testing
Handles PDF ingestion, query execution, and result storage
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.ingest import ingest_pdfs_parallel
from src.embedder import EmbeddingModel
from src.vector_store import MultiModelVectorStore
from src.retriever import retrieve
from src.llm import stream_answer
from src.utils import build_prompt
from src.accuracy_scorer import AccuracyScorer
from src.qdrant_manager import get_qdrant_manager
from src.logger import get_logger
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = get_logger("automated_test_runner")


class AutomatedTestRunner:
    """Runs automated accuracy tests and stores results"""
    
    def __init__(self, test_pdf_dir: str = "/app/Test-PDFs"):
        self.test_pdf_dir = Path(test_pdf_dir)
        self.embedder = EmbeddingModel()
        self.vector_store = MultiModelVectorStore()
        self.scorer = AccuracyScorer()
        self.qdrant_manager = get_qdrant_manager()
        
    def _init_test_collections(self):
        """Initialize Qdrant collections for test data"""
        client = self.qdrant_manager.client
        
        # Collection for test results
        collections = [c.name for c in client.get_collections().collections]
        
        if "munirag_test_results" not in collections:
            client.create_collection(
                collection_name="munirag_test_results",
                vectors_config=VectorParams(
                    size=384,  # Small embedding for metadata search
                    distance=Distance.COSINE
                )
            )
            logger.info("Created munirag_test_results collection")
        
        if "munirag_accuracy_metrics" not in collections:
            client.create_collection(
                collection_name="munirag_accuracy_metrics",
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            logger.info("Created munirag_accuracy_metrics collection")
    
    def ingest_test_pdfs(self) -> Dict[str, int]:
        """
        Ingest all test PDFs
        
        Returns:
            Dict mapping PDF names to chunk counts
        """
        logger.info(f"Ingesting test PDFs from {self.test_pdf_dir}")
        
        # Clear existing document collections (but not test collections)
        self.qdrant_manager.purge_munirag_collections()
        
        # Initialize test collections after purge
        self._init_test_collections()
        
        # Find all PDFs
        pdf_files = list(self.test_pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDFs found in {self.test_pdf_dir}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDFs to ingest")
        
        # Create mock file objects for ingestion
        class MockFile:
            def __init__(self, path):
                self.name = path.name
                self.path = str(path)
                self.size = path.stat().st_size
                
            def read(self):
                with open(self.path, 'rb') as f:
                    return f.read()
        
        mock_files = [MockFile(pdf) for pdf in pdf_files]
        
        # Ingest PDFs
        chunk_counts = {}
        success_count, error_count = ingest_pdfs_parallel(mock_files)
        
        # Get total chunk count
        info = self.vector_store.get_collection_info()
        total_chunks = info.get("vectors_count", 0)
        
        # Estimate chunks per PDF (rough)
        for pdf in pdf_files:
            chunk_counts[pdf.name] = total_chunks // len(pdf_files) if pdf_files else 0
        
        logger.info(f"Ingestion complete: {success_count} successful, {error_count} errors")
        return chunk_counts
    
    def run_test_suite(self, test_name: Optional[str] = None) -> Dict:
        """
        Run complete test suite
        
        Args:
            test_name: Optional name for this test run
            
        Returns:
            Complete test results with scores
        """
        test_name = test_name or f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting test suite: {test_name}")
        
        # Load test questions
        with open("test_questions.json", 'r') as f:
            test_data = json.load(f)
        
        results = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "configuration": self._get_current_config(),
            "test_results": [],
            "summary": {}
        }
        
        # Run tests for each PDF
        for pdf_name, pdf_data in test_data["test_suites"].items():
            logger.info(f"\nTesting {pdf_name} ({len(pdf_data['questions'])} questions)")
            
            for question_data in pdf_data["questions"]:
                result = self._test_single_question(question_data, pdf_name, pdf_data["metadata"])
                results["test_results"].append(result)
                
                # Log progress
                score = result["score_data"]["overall_score"]
                grade = result["score_data"]["grade"]
                logger.info(f"  {question_data['id']}: {grade} ({score:.2f})")
        
        # Calculate summary statistics
        results["summary"] = self.scorer.score_test_suite(results["test_results"])
        
        # Store results
        self._store_test_results(results)
        
        logger.info(f"\nTest suite complete: {results['summary']['overall_grade']} "
                   f"({results['summary']['overall_average']:.2f})")
        
        return results
    
    def _test_single_question(self, question_data: Dict, pdf_name: str, pdf_metadata: Dict) -> Dict:
        """Test a single question and score the response"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(question_data["question"])
            
            # Retrieve relevant chunks
            retrieved_chunks = retrieve(query_embedding, top_k=settings.TOP_K)
            
            # Generate response
            prompt = build_prompt(retrieved_chunks, question_data["question"])
            response_text = ""
            for chunk in stream_answer(prompt):
                response_text += chunk
            
            # Extract sources from chunks
            sources = [chunk[1].get("source", "unknown") for chunk in retrieved_chunks[:3]]
            
            # Score the response
            score_data = self.scorer.score_response(
                response_text,
                question_data,
                sources
            )
            
            # Compile result
            result = {
                "question_id": question_data["id"],
                "question": question_data["question"],
                "response": response_text,
                "score_data": score_data,
                "question_metadata": {
                    "category": question_data["category"],
                    "difficulty": question_data["difficulty"],
                    "pdf": pdf_name,
                    "pdf_type": pdf_metadata["document_type"]
                },
                "performance": {
                    "query_time": round(time.time() - start_time, 2),
                    "chunks_retrieved": len(retrieved_chunks),
                    "response_length": len(response_text)
                },
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error testing question {question_data['id']}: {e}")
            result = {
                "question_id": question_data["id"],
                "question": question_data["question"],
                "response": "",
                "score_data": {
                    "overall_score": 0,
                    "grade": "failing",
                    "sub_scores": {
                        "factual_accuracy": 0,
                        "completeness": 0,
                        "relevance": 0,
                        "coherence": 0
                    }
                },
                "question_metadata": {
                    "category": question_data["category"],
                    "difficulty": question_data["difficulty"],
                    "pdf": pdf_name
                },
                "performance": {
                    "query_time": round(time.time() - start_time, 2),
                    "chunks_retrieved": 0,
                    "response_length": 0
                },
                "status": "error",
                "error": str(e)
            }
        
        return result
    
    def _get_current_config(self) -> Dict:
        """Get current system configuration"""
        return {
            "embedding_model": os.getenv("EMBEDDING_MODEL", "unknown"),
            "semantic_chunking": os.getenv("SEMANTIC_CHUNKING", "unknown"),
            "chunk_size": os.getenv("CHUNK_SIZE", "unknown"),
            "chunk_overlap": os.getenv("CHUNK_OVERLAP", "unknown"),
            "top_k": os.getenv("TOP_K", "unknown"),
            "retrieval_top_k": os.getenv("RETRIEVAL_TOP_K", "unknown"),
            "rerank_top_k": os.getenv("RERANK_TOP_K", "unknown")
        }
    
    def _store_test_results(self, results: Dict):
        """Store test results in Qdrant for tracking"""
        client = self.qdrant_manager.client
        
        # Create a simple embedding from the test name
        test_embedding = [0.1] * 384  # Placeholder embedding
        
        # Store in test results collection
        point = PointStruct(
            id=str(hash(results["test_name"])),
            vector=test_embedding,
            payload={
                "test_name": results["test_name"],
                "timestamp": results["timestamp"],
                "overall_score": results["summary"]["overall_average"],
                "overall_grade": results["summary"]["overall_grade"],
                "configuration": results["configuration"],
                "summary": results["summary"]
            }
        )
        
        client.upsert(
            collection_name="munirag_test_results",
            points=[point]
        )
        
        # Also save to JSON file
        output_file = f"test_results/{results['test_name']}.json"
        os.makedirs("test_results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results stored in Qdrant and saved to {output_file}")
    
    def compare_with_baseline(self, current_results: Dict, baseline_name: str = "baseline") -> Dict:
        """Compare current results with baseline"""
        # TODO: Implement baseline comparison
        pass
    
    def run_continuous_monitoring(self, interval_minutes: int = 60):
        """Run tests continuously at specified interval"""
        logger.info(f"Starting continuous monitoring (every {interval_minutes} minutes)")
        
        while True:
            try:
                # Run test suite
                results = self.run_test_suite()
                
                # Check for regression
                if results["summary"]["overall_average"] < 0.7:
                    logger.warning(f"⚠️ ACCURACY BELOW THRESHOLD: {results['summary']['overall_average']:.2f}")
                
                # Wait for next run
                logger.info(f"Next test in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous monitoring")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(60)  # Wait a minute before retrying


def main():
    """Run automated tests"""
    runner = AutomatedTestRunner()
    
    # Ingest test PDFs
    logger.info("=== Phase 1: Ingesting Test PDFs ===")
    chunk_counts = runner.ingest_test_pdfs()
    
    if not chunk_counts:
        logger.error("No PDFs were ingested. Exiting.")
        return
    
    # Run test suite
    logger.info("\n=== Phase 2: Running Test Suite ===")
    results = runner.run_test_suite("initial_baseline")
    
    # Print summary
    print("\n=== TEST RESULTS SUMMARY ===")
    print(f"Overall Grade: {results['summary']['overall_grade'].upper()}")
    print(f"Overall Score: {results['summary']['overall_average']:.1%}")
    print(f"Total Questions: {results['summary']['total_questions']}")
    
    print("\nScores by Category:")
    for category, score in results['summary']['category_averages'].items():
        print(f"  {category}: {score:.1%}")
    
    print("\nScores by Difficulty:")
    for difficulty, score in results['summary']['difficulty_averages'].items():
        print(f"  {difficulty}: {score:.1%}")
    
    print(f"\n{results['summary']['summary']}")


if __name__ == "__main__":
    main()