"""
Optimized PDF ingestion with proper GPU batching
Fixes the critical page-by-page embedding bottleneck
"""

from pathlib import Path
from pypdf import PdfReader
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embedder import EmbeddingModel
from src.vector_store_v2 import MultiModelVectorStore
from src.config import settings
from src.logger import get_logger
import torch

logger = get_logger("ingest_optimized")

# Initialize text splitter
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.MAX_CHUNK_TOKENS,
    chunk_overlap=int(settings.MAX_CHUNK_TOKENS * 0.1)
)


def ingest_pdfs_optimized(files, progress_callback=None, progress_bar=None):
    """
    Optimized PDF ingestion with batched embedding generation.
    
    Key optimizations:
    1. Extract ALL text first, then embed in large batches
    2. Minimize progress callbacks to avoid GPU sync
    3. Store documents in batches to reduce database calls
    
    Returns:
        tuple: (success_count, error_count)
    """
    embedder = EmbeddingModel()
    vector_store = MultiModelVectorStore(embedder.model_name)
    success_count = 0
    error_count = 0
    total_files = len(files)
    
    logger.info(f"Starting optimized ingestion of {total_files} PDF files")
    
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
            
            # Read PDF
            try:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                
                if total_pages == 0:
                    logger.warning(f"PDF {file.name} has no pages")
                    error_count += 1
                    continue
                
                logger.info(f"Processing PDF {file.name} with {total_pages} pages")
                
            except Exception as e:
                logger.error(f"Error reading PDF {file.name}: {str(e)}")
                error_count += 1
                continue
            
            # OPTIMIZATION: Extract ALL text and chunks first
            start_time = datetime.now()
            all_chunks = []
            all_metadata = []
            pages_with_text = 0
            
            # Phase 1: Text extraction (10% of progress)
            for page_no, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # Split into chunks
                        chunks = _splitter.split_text(text)
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            all_chunks.append(chunk)
                            all_metadata.append({
                                "type": "pdf",
                                "source": file.name,
                                "page": page_no,
                                "total_pages": total_pages,
                                "chunk_id": len(all_chunks) - 1,
                                "chunk_ref": f"{file.name}::page{page_no}::chunk{chunk_idx}"
                            })
                        
                        pages_with_text += 1
                    
                    # Update progress sparingly (every 10 pages)
                    if progress_bar and page_no % 10 == 0:
                        extraction_progress = 0.1 * (page_no / total_pages)
                        progress_bar.progress(extraction_progress, text=f"Extracting page {page_no}/{total_pages}")
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_no} of {file.name}: {str(e)}")
                    continue
            
            # Check if we got any content
            if not all_chunks:
                logger.warning(f"No text extracted from {file.name}")
                error_count += 1
                continue
            
            logger.info(f"Extracted {len(all_chunks)} chunks from {pages_with_text} pages")
            
            # Phase 2: Batch embedding generation (90% of progress)
            if progress_bar:
                progress_bar.progress(0.1, text=f"Generating embeddings for {len(all_chunks)} chunks...")
            
            # CRITICAL OPTIMIZATION: Embed ALL chunks in one batch
            try:
                # Clear GPU cache before large operation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                embed_start = datetime.now()
                embeddings = embedder.embed_documents(all_chunks)
                embed_time = (datetime.now() - embed_start).total_seconds()
                
                embeddings_per_sec = len(all_chunks) / embed_time if embed_time > 0 else 0
                logger.info(f"Generated {len(all_chunks)} embeddings in {embed_time:.1f}s ({embeddings_per_sec:.0f} embeddings/sec)")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for {file.name}: {str(e)}")
                error_count += 1
                continue
            
            # Phase 3: Store in database (batched)
            if progress_bar:
                progress_bar.progress(0.95, text="Storing in database...")
            
            try:
                # Create documents
                documents = []
                for i, (chunk, metadata) in enumerate(zip(all_chunks, all_metadata)):
                    documents.append({
                        "id": None,  # Let VectorStore generate UUID
                        "content": chunk,
                        "metadata": metadata
                    })
                
                # Store in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size]
                    vector_store.add_documents(batch_docs, batch_embeddings)
                
                success_count += 1
                
                # Final stats
                total_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Successfully processed {file.name}:")
                logger.info(f"  - Pages: {pages_with_text}/{total_pages}")
                logger.info(f"  - Chunks: {len(all_chunks)}")
                logger.info(f"  - Time: {total_time:.1f}s")
                logger.info(f"  - Speed: {len(all_chunks)/total_time:.1f} chunks/sec")
                
            except Exception as e:
                logger.error(f"Error storing documents for {file.name}: {str(e)}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"Unexpected error processing {file.name}: {str(e)}")
            error_count += 1
    
    # Complete progress
    if progress_callback:
        progress_callback(1.0)
    if progress_bar:
        progress_bar.progress(1.0, text="Complete!")
    
    logger.info(f"PDF ingestion complete: {success_count} successful, {error_count} errors")
    return success_count, error_count


def estimate_processing_time(file_size_mb: float, pages: int) -> float:
    """
    Estimate processing time based on optimized performance.
    
    Assumptions:
    - Text extraction: ~50 pages/second
    - Embedding generation: ~3000 chunks/second on GPU
    - Average 10 chunks per page
    """
    extraction_time = pages / 50
    embedding_time = (pages * 10) / 3000
    overhead_time = 2  # Model loading, database ops
    
    return extraction_time + embedding_time + overhead_time