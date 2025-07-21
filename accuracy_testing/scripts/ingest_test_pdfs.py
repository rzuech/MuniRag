#!/usr/bin/env python3
"""
Ingest test PDFs directly from filesystem
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/app')

from src.pdf_parallel_processor import ParallelPDFProcessor
from src.embedder import EmbeddingModel
from src.vector_store import MultiModelVectorStore
from src.config import settings
from src.logger import get_logger
import torch

logger = get_logger("ingest_test_pdfs")


def ingest_test_pdfs():
    """Ingest all PDFs from Test-PDFs directory"""
    test_dir = Path("/app/Test-PDFs")
    
    if not test_dir.exists():
        logger.error(f"Test directory {test_dir} not found!")
        return 0, 0
    
    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found!")
        return 0, 0
    
    logger.info(f"Found {len(pdf_files)} PDFs to ingest")
    
    # Initialize components
    vector_store = MultiModelVectorStore()
    embedder = EmbeddingModel()
    processor = ParallelPDFProcessor()
    
    success_count = 0
    error_count = 0
    
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing {pdf_path.name}")
            
            # Process PDF
            chunks = processor.process_pdf(str(pdf_path))
            
            if not chunks:
                logger.warning(f"No chunks extracted from {pdf_path.name}")
                error_count += 1
                continue
            
            logger.info(f"Extracted {len(chunks)} chunks")
            
            # Generate embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = embedder.embed_documents(texts)
            
            # Create documents
            documents = []
            for chunk in chunks:
                documents.append({
                    "id": None,
                    "content": chunk['content'],
                    "metadata": chunk['metadata']
                })
            
            # Store in vector database
            vector_store.add_documents(documents, embeddings)
            
            success_count += 1
            logger.info(f"✓ Successfully ingested {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            error_count += 1
    
    logger.info(f"Ingestion complete: {success_count} successful, {error_count} errors")
    return success_count, error_count


if __name__ == "__main__":
    success, errors = ingest_test_pdfs()
    
    # Get collection info
    from src.qdrant_manager import get_qdrant_manager
    manager = get_qdrant_manager()
    manager.health_check()
    
    sys.exit(0 if errors == 0 else 1)