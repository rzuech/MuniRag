#!/usr/bin/env python3
"""
Re-ingest PDFs after Qdrant corruption
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_parallel_processor import PDFParallelProcessor
from src.embedder import EmbeddingModel
from src.vector_store import MultiModelVectorStore
from src.config import settings
from src.logger import get_logger

logger = get_logger("reingest")

def reingest_pdfs(pdf_dir="Test-PDFs"):
    """Re-ingest all PDFs from a directory"""
    
    # Find all PDFs
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {pdf_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDFs to process")
    
    # Initialize components
    logger.info("Initializing components...")
    embedder = EmbeddingModel()
    vector_store = MultiModelVectorStore()
    processor = PDFParallelProcessor()
    
    # Process each PDF
    for pdf_path in pdf_files:
        logger.info(f"\nProcessing: {pdf_path.name}")
        
        try:
            # Extract text
            chunks = processor.process_file(str(pdf_path))
            logger.info(f"  Extracted {len(chunks)} chunks")
            
            if not chunks:
                logger.warning(f"  No text extracted from {pdf_path.name}")
                continue
            
            # Prepare documents
            documents = []
            for chunk in chunks:
                documents.append({
                    "content": chunk["text"],
                    "metadata": {
                        "source": pdf_path.name,
                        "page": chunk.get("page", 1),
                        "chunk_index": chunk.get("chunk_index", 0)
                    }
                })
            
            # Embed in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                texts = [doc["content"] for doc in batch]
                
                logger.info(f"  Embedding batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                embeddings = embedder.embed_documents(texts)
                
                # Store in vector store
                vector_store.add_documents(batch, embeddings)
            
            logger.info(f"✅ Completed: {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Get final stats
    stats = vector_store.get_stats()
    logger.info(f"\n🎉 Ingestion complete!")
    logger.info(f"Collection: {stats['name']}")
    logger.info(f"Total documents: {stats['vectors_count']}")


if __name__ == "__main__":
    reingest_pdfs()