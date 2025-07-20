"""
Parallel PDF ingestion using the fast PDF processor
Replaces the slow PyPDF2-based ingestion with PyMuPDF + multiprocessing
"""

from pathlib import Path
from datetime import datetime
from src.pdf_parallel_processor import ParallelPDFProcessor
from src.embedder import EmbeddingModel
from src.vector_store_v2 import MultiModelVectorStore
from src.config import settings
from src.logger import get_logger
import torch

logger = get_logger("ingest_parallel")


def ingest_website(url, max_pages=None):
    """
    Website ingestion placeholder - to be implemented
    """
    logger.warning("Website ingestion not yet implemented in the refactored version")
    return 0


def ingest_pdfs_parallel(files, progress_callback=None, progress_bar=None):
    """
    High-performance PDF ingestion using parallel processor.
    
    Key improvements:
    1. Uses PyMuPDF (10-50x faster than PyPDF2)
    2. Parallel page processing across CPU cores
    3. Semantic chunking for better quality
    4. OCR support for scanned documents
    
    Returns:
        tuple: (success_count, error_count)
    """
    vector_store = MultiModelVectorStore()
    embedder = EmbeddingModel()
    processor = ParallelPDFProcessor()
    
    success_count = 0
    error_count = 0
    total_files = len(files)
    
    logger.info(f"Starting parallel ingestion of {total_files} PDF files")
    logger.info(f"Using {settings.PDF_WORKERS or 'auto'} workers, semantic chunking: {settings.SEMANTIC_CHUNKING}")
    
    for file_idx, file in enumerate(files):
        try:
            # Update file-level progress
            if progress_callback:
                progress_callback(file_idx / total_files)
            
            # Validate file
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                logger.warning(f"File {file.name} too large ({file_size_mb:.1f}MB)")
                error_count += 1
                continue
            
            if not file.name.lower().endswith('.pdf'):
                logger.warning(f"File {file.name} is not a PDF")
                error_count += 1
                continue
            
            logger.info(f"Processing PDF {file.name} ({file_size_mb:.1f}MB)")
            start_time = datetime.now()
            
            # Process PDF with parallel processor
            def update_progress(current, total, message=""):
                if progress_bar:
                    # Map processor progress (0-0.9) to overall progress
                    overall_progress = 0.9 * (current / total) if total > 0 else 0
                    progress_bar.progress(overall_progress, text=message)
            
            try:
                # Save uploaded file to temporary location
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Process PDF and get chunks
                chunks = processor.process_pdf(
                    pdf_path=tmp_path,
                    progress_callback=update_progress
                )
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                if not chunks:
                    logger.warning(f"No content extracted from {file.name}")
                    error_count += 1
                    continue
                
                logger.info(f"Extracted {len(chunks)} chunks from {file.name}")
                
                # Update progress for embedding phase
                if progress_bar:
                    progress_bar.progress(0.9, text=f"Generating embeddings for {len(chunks)} chunks...")
                
                # Extract text and metadata
                texts = [chunk['content'] for chunk in chunks]
                metadata_list = [chunk['metadata'] for chunk in chunks]
                
                # Clear GPU cache before embedding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate embeddings in batches
                embed_start = datetime.now()
                embeddings = embedder.embed_documents(texts)
                embed_time = (datetime.now() - embed_start).total_seconds()
                
                embeddings_per_sec = len(texts) / embed_time if embed_time > 0 else 0
                logger.info(f"Generated {len(embeddings)} embeddings in {embed_time:.1f}s ({embeddings_per_sec:.0f} embeddings/sec)")
                
                # Store in vector database
                if progress_bar:
                    progress_bar.progress(0.95, text="Storing in database...")
                
                # Create documents with proper structure
                documents = []
                for i, (chunk, metadata) in enumerate(zip(chunks, metadata_list)):
                    documents.append({
                        "id": None,  # Let VectorStore generate UUID
                        "content": chunk['content'],
                        "metadata": metadata
                    })
                
                # Store in batches
                batch_size = 1000
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size]
                    vector_store.add_documents(batch_docs, batch_embeddings)
                
                success_count += 1
                
                # Final stats
                total_time = (datetime.now() - start_time).total_seconds()
                pages_processed = max(m.get('total_pages', 0) for m in metadata_list) if metadata_list else 0
                
                logger.info(f"Successfully processed {file.name}:")
                logger.info(f"  - Pages: {pages_processed}")
                logger.info(f"  - Chunks: {len(chunks)}")
                logger.info(f"  - Time: {total_time:.1f}s ({pages_processed/total_time:.1f} pages/sec)")
                logger.info(f"  - Semantic chunking: {settings.SEMANTIC_CHUNKING}")
                
            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"Unexpected error with {file.name}: {str(e)}")
            error_count += 1
    
    # Complete progress
    if progress_callback:
        progress_callback(1.0)
    if progress_bar:
        progress_bar.progress(1.0, text="Complete!")
    
    logger.info(f"PDF ingestion complete: {success_count} successful, {error_count} errors")
    return success_count, error_count