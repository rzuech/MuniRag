"""
Adapter to connect the parallel PDF processor to existing code
With reliable CPU detection for all systems
"""
import multiprocessing as mp
import os
from typing import List, Tuple, Optional
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

def get_optimal_workers() -> int:
    """
    Get optimal number of workers with reliable detection
    
    Strategy:
    - Default: CPU count - 1 (leave one free for system)
    - Minimum: 1 worker (always at least one)
    - Maximum: 8 workers (diminishing returns after)
    - Override: PDF_WORKERS environment variable
    """
    # Check for environment override first
    env_workers = os.getenv('PDF_WORKERS', '').strip()
    if env_workers.isdigit():
        workers = int(env_workers)
        logger.info(f"Using PDF_WORKERS from environment: {workers}")
        return max(1, min(workers, 16))  # Clamp between 1-16
    
    # Get CPU count
    try:
        cpu_count = mp.cpu_count()
    except:
        cpu_count = 2  # Safe fallback
        
    # Calculate optimal workers
    if cpu_count <= 2:
        workers = 1  # Single CPU or dual-core: use 1 worker
    elif cpu_count <= 4:
        workers = cpu_count - 1  # Leave 1 core free
    else:
        workers = min(cpu_count - 1, 8)  # Cap at 8 workers
    
    logger.info(f"System has {cpu_count} CPUs, using {workers} workers for PDF processing")
    return workers

def should_use_parallel(file) -> bool:
    """
    Determine if parallel processing should be used for a PDF
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        bool: True if parallel processing would help
    """
    try:
        # Check file size (parallel helps more with larger files)
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb < 0.5:  # Less than 500KB
            logger.debug(f"File {file.name} too small for parallel ({file_size_mb:.1f}MB)")
            return False
            
        # Try to get page count
        try:
            reader = PdfReader(file)
            page_count = len(reader.pages)
            
            # Use parallel for files with more than 10 pages
            if page_count <= 10:
                logger.debug(f"File {file.name} has only {page_count} pages, using sequential")
                return False
                
            logger.info(f"File {file.name} has {page_count} pages, using parallel processing")
            return True
            
        except:
            # If we can't read the PDF, default to parallel for larger files
            return file_size_mb > 1.0
            
    except Exception as e:
        logger.warning(f"Error checking file {file.name}: {e}")
        return False

def extract_pdf_parallel(file, progress_callback=None) -> List[Tuple[int, str]]:
    """
    Extract text from PDF using parallel processing
    
    Args:
        file: Streamlit uploaded file object
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of tuples (page_index, text)
    """
    from pypdf import PdfReader
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import tempfile
    
    # Save uploaded file to temp location (needed for multiprocessing)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Get total pages
        reader = PdfReader(tmp_path)
        total_pages = len(reader.pages)
        
        # Get optimal worker count
        num_workers = get_optimal_workers()
        
        # Create batches
        pages_per_worker = max(1, total_pages // num_workers)
        batches = []
        
        for i in range(0, total_pages, pages_per_worker):
            batch_end = min(i + pages_per_worker, total_pages)
            batches.append((i, batch_end))
        
        logger.info(f"Processing {total_pages} pages in {len(batches)} batches with {num_workers} workers")
        
        # Process batches in parallel
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            for start, end in batches:
                future = executor.submit(_process_pdf_batch, tmp_path, start, end)
                future_to_batch[future] = (start, end)
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_batch):
                batch_start, batch_end = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    # Update progress
                    completed += (batch_end - batch_start)
                    if progress_callback:
                        progress_callback(completed / total_pages)
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
        
        # Sort results by page index
        results.sort(key=lambda x: x[0])
        
        return results
        
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

def _process_pdf_batch(pdf_path: str, start_page: int, end_page: int) -> List[Tuple[int, str]]:
    """
    Process a batch of PDF pages (runs in separate process)
    
    Args:
        pdf_path: Path to PDF file
        start_page: Starting page index
        end_page: Ending page index (exclusive)
        
    Returns:
        List of tuples (page_index, text)
    """
    from pypdf import PdfReader
    
    results = []
    reader = PdfReader(pdf_path)
    
    for page_idx in range(start_page, end_page):
        try:
            page = reader.pages[page_idx]
            text = page.extract_text()
            
            # Only include pages with meaningful text
            if text and text.strip():
                results.append((page_idx, text))
        except Exception as e:
            # Log error but continue processing other pages
            print(f"Error extracting page {page_idx}: {e}")
    
    return results

# For backward compatibility
def extract_pdf_sequential(file) -> List[Tuple[int, str]]:
    """
    Fallback sequential extraction (original method)
    """
    from pypdf import PdfReader
    
    results = []
    reader = PdfReader(file)
    
    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text and text.strip():
                results.append((page_idx, text))
        except Exception as e:
            logger.error(f"Error extracting page {page_idx}: {e}")
    
    return results