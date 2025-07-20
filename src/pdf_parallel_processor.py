"""
High-performance parallel PDF processor with OCR support
"""
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from typing import List, Dict, Tuple, Iterator, Optional
import time
import os
import re
import hashlib
from pathlib import Path
import numpy as np
from queue import Empty
import warnings
warnings.filterwarnings("ignore", message=".*flash_attn.*")

try:
    import pymupdf4llm
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR not available. Install pytesseract and Pillow for OCR support.")

from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelPDFProcessor:
    """High-performance PDF processor with parallel processing and OCR support"""
    
    def __init__(self, 
                 num_workers: Optional[int] = None,
                 enable_ocr: bool = True,
                 ocr_threshold: int = 50,
                 batch_size: int = 10,
                 semantic_chunking: bool = True,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize parallel PDF processor
        
        Args:
            num_workers: Number of worker processes (defaults to CPU count)
            enable_ocr: Enable OCR for scanned PDFs
            ocr_threshold: Minimum text characters per page to skip OCR
            batch_size: Number of pages to process in each batch
            semantic_chunking: Use semantic chunking instead of fixed-size
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_threshold = ocr_threshold
        self.batch_size = batch_size
        self.semantic_chunking = semantic_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer for chunk sizing
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # For semantic chunking
        self._embedder = None
        self._semantic_threshold = 0.7
        
        logger.info(f"Initialized ParallelPDFProcessor with {self.num_workers} workers")
        
    def process_pdf(self, 
                   pdf_path: str, 
                   progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Process PDF with parallel extraction and optional OCR
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Callback function for progress updates
            
        Returns:
            List of document chunks with metadata
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Get PDF info
        total_pages = self._get_pdf_info(str(pdf_path))
        logger.info(f"PDF has {total_pages} pages")
        
        # Create page batches for parallel processing
        batches = self._create_batches(total_pages, self.batch_size)
        
        # Process pages in parallel
        with Manager() as manager:
            # Shared progress counter
            progress_dict = manager.dict()
            progress_dict['completed'] = 0
            progress_dict['total'] = total_pages
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for batch_start, batch_end in batches:
                    future = executor.submit(
                        self._process_page_batch,
                        str(pdf_path),
                        batch_start,
                        batch_end,
                        self.enable_ocr,
                        self.ocr_threshold
                    )
                    futures.append(future)
                
                # Collect results as they complete
                all_pages = []
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        all_pages.extend(batch_results)
                        
                        # Update progress
                        progress_dict['completed'] += len(batch_results)
                        if progress_callback:
                            progress_callback(
                                progress_dict['completed'], 
                                progress_dict['total']
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
        
        # Sort pages by page number
        all_pages.sort(key=lambda x: x['page_num'])
        
        # Combine all text
        full_text = "\n\n".join(page['text'] for page in all_pages if page['text'])
        
        # Extract metadata
        metadata = self._extract_metadata(str(pdf_path))
        metadata['pages'] = total_pages
        metadata['processing_time'] = time.time() - start_time
        
        # Chunk the text
        if self.semantic_chunking:
            chunks = self._semantic_chunk_text(full_text, all_pages, metadata)
        else:
            chunks = self._fixed_chunk_text(full_text, metadata)
            
        logger.info(f"Processed {total_pages} pages in {metadata['processing_time']:.2f}s")
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    @staticmethod
    def _process_page_batch(pdf_path: str, 
                           start_page: int, 
                           end_page: int,
                           enable_ocr: bool,
                           ocr_threshold: int) -> List[Dict]:
        """Process a batch of pages (runs in separate process)"""
        results = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(start_page, min(end_page, len(doc))):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Check if OCR is needed
            if enable_ocr and len(text.strip()) < ocr_threshold:
                # Page might be scanned, try OCR
                try:
                    # Convert page to image
                    pix = page.get_pixmap(dpi=300)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Run OCR
                    ocr_text = pytesseract.image_to_string(img)
                    
                    # Use OCR text if it's longer
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
            
            # Extract additional info
            page_result = {
                'page_num': page_num + 1,
                'text': text,
                'has_images': len(page.get_images()) > 0,
                'has_tables': len(list(page.find_tables())) > 0,  # Convert to list first
                'ocr_used': enable_ocr and len(text.strip()) < ocr_threshold
            }
            
            results.append(page_result)
        
        doc.close()
        return results
    
    def _semantic_chunk_text(self, 
                           text: str, 
                           pages: List[Dict], 
                           metadata: Dict) -> List[Dict]:
        """Create semantic chunks that keep related content together"""
        
        # Initialize embedder if needed
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Get embeddings for all sentences
        logger.info(f"Computing embeddings for {len(sentences)} sentences...")
        embeddings = self._embedder.encode(sentences, show_progress_bar=False)
        
        # Group sentences into semantic chunks
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        current_tokens = self._count_tokens(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i]
            sentence_tokens = self._count_tokens(sentence)
            
            # Calculate similarity
            similarity = cosine_similarity(
                [current_embedding], 
                [embedding]
            )[0][0]
            
            # Check if we should add to current chunk
            if (similarity > self._semantic_threshold and 
                current_tokens + sentence_tokens <= self.chunk_size):
                # Add to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
                # Update centroid embedding
                all_embeddings = [current_embedding] + [embedding]
                current_embedding = np.mean(all_embeddings, axis=0)
            else:
                # Start new chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = self._generate_chunk_id(chunk_text, len(chunks))
                
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': len(chunks),
                        'tokens': current_tokens,
                        'sentences': len(current_chunk),
                        'semantic_coherence': float(similarity)
                    }
                })
                
                # Reset for new chunk
                current_chunk = [sentence]
                current_embedding = embedding
                current_tokens = sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id(chunk_text, len(chunks))
            
            chunks.append({
                'id': chunk_id,
                'content': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'tokens': current_tokens,
                    'sentences': len(current_chunk)
                }
            })
        
        return chunks
    
    def _fixed_chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Create fixed-size chunks with overlap"""
        chunks = []
        
        # Split text into tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Create chunks with overlap
        start = 0
        while start < len(tokens):
            # Get chunk tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk document
            chunk_id = self._generate_chunk_id(chunk_text, len(chunks))
            
            chunks.append({
                'id': chunk_id,
                'content': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'tokens': len(chunk_tokens),
                    'chunk_total': (len(tokens) // self.chunk_size) + 1
                }
            })
            
            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter - can be improved
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate unique chunk ID"""
        return hashlib.md5(f"{text[:50]}_{index}".encode()).hexdigest()
    
    def _get_pdf_info(self, pdf_path: str) -> int:
        """Get PDF page count"""
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    
    def _extract_metadata(self, pdf_path: str) -> Dict:
        """Extract PDF metadata"""
        doc = fitz.open(pdf_path)
        metadata = {
            'file_path': pdf_path,
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
        }
        doc.close()
        return metadata
    
    def _create_batches(self, total_pages: int, batch_size: int) -> List[Tuple[int, int]]:
        """Create page batches for parallel processing"""
        batches = []
        for i in range(0, total_pages, batch_size):
            batches.append((i, min(i + batch_size, total_pages)))
        return batches


# Quick test function
def test_parallel_processor():
    """Test the parallel processor"""
    processor = ParallelPDFProcessor(
        num_workers=4,
        enable_ocr=True,
        semantic_chunking=True
    )
    
    # Test with a sample PDF if available
    test_pdf = "test.pdf"
    if os.path.exists(test_pdf):
        def progress_callback(completed, total):
            print(f"Progress: {completed}/{total} pages ({completed/total*100:.1f}%)")
        
        chunks = processor.process_pdf(test_pdf, progress_callback)
        print(f"Processed {len(chunks)} chunks")
        return chunks
    else:
        print("No test PDF found")
        return []


if __name__ == "__main__":
    import io
    test_parallel_processor()