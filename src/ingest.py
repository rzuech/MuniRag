"""
=============================================================================
INGEST.PY - Document Ingestion System for MuniRag
=============================================================================

This module handles the ingestion of documents from various sources (PDFs, websites)
into the ChromaDB vector database for later retrieval and question answering.

PURPOSE:
- Process PDF files and extract text from each page
- Crawl websites and extract meaningful content
- Split large documents into manageable chunks
- Convert text chunks into embeddings for similarity search
- Store everything in ChromaDB with metadata

DOCUMENT PROCESSING PIPELINE:
1. Input validation (file size, URL format, etc.)
2. Content extraction (PDF text, website content)
3. Text splitting into chunks (for better retrieval)
4. Embedding generation (convert text to numbers)
5. Storage in vector database with metadata

SECURITY FEATURES:
- File size limits prevent memory exhaustion
- URL validation blocks local/private addresses
- Error handling prevents crashes on bad documents
- Request timeouts prevent hanging on slow sites

USAGE:
    from ingest import ingest_pdfs, ingest_website
    
    # Process uploaded PDF files
    success_count, error_count = ingest_pdfs(uploaded_files)
    
    # Crawl a website
    pages_crawled = ingest_website("https://example.com")
"""

from pathlib import Path
import trafilatura
import PyPDF2
import requests
from urllib.parse import urlparse, urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedder import EmbeddingService
from retriever import get_collection
from config import MAX_CHUNK_TOKENS, MAX_FILE_SIZE_MB, MAX_PAGES_CRAWL, REQUEST_TIMEOUT
from logger import get_logger

# Get a logger specific to this module
logger = get_logger("ingest")

# Initialize the text splitter
# This splits long documents into smaller chunks for better retrieval
# chunk_overlap creates some overlap between chunks to preserve context
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_TOKENS,
    chunk_overlap=int(MAX_CHUNK_TOKENS * 0.1)  # 10% overlap
)


def _add_chunks(chunks, metadata, collection):
    """
    Add text chunks to the ChromaDB collection with embeddings.
    
    This function:
    1. Converts text chunks into embeddings using the embedding service
    2. Creates unique IDs for each chunk
    3. Stores chunks, embeddings, and metadata in ChromaDB
    
    Args:
        chunks (list): List of text chunks to add
        metadata (dict): Metadata for all chunks (source, type, etc.)
        collection: ChromaDB collection to add to
    
    Raises:
        Exception: If embedding generation or storage fails
    """
    try:
        if not chunks:
            logger.warning("No chunks to add - empty chunks list")
            return
            
        # Get the singleton embedding service
        embedder = EmbeddingService()
        
        # Convert text chunks to embeddings
        # This is the most computationally expensive part
        logger.debug(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = embedder.encode(chunks).tolist()
        
        # Create unique IDs for each chunk
        # Format: "source_filename::chunk0", "source_filename::chunk1", etc.
        ids = [f"{metadata['source']}::chunk{i}" for i, _ in enumerate(chunks)]
        
        # Store everything in ChromaDB
        collection.add(
            documents=chunks,           # The actual text content
            embeddings=embeddings,      # Numerical representations
            ids=ids,                   # Unique identifiers
            metadatas=[metadata] * len(chunks)  # Same metadata for all chunks
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks for {metadata['source']}")
        
    except Exception as e:
        logger.error(f"Error adding chunks for {metadata['source']}: {str(e)}")
        raise


def validate_url(url):
    """
    Validate URL format and check for security issues.
    
    This function prevents:
    - Invalid URL formats
    - Localhost/private IP crawling (security risk)
    - Non-HTTP(S) protocols
    
    Args:
        url (str): URL to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)
        
        # Check protocol
        if parsed.scheme not in ['http', 'https']:
            return False, "URL must use HTTP or HTTPS protocol"
        
        # Check if URL has a domain
        if not parsed.netloc:
            return False, "Invalid URL format - no domain found"
        
        # Block localhost and private IPs for security
        # This prevents crawling internal services
        blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '10.', '192.168.', '172.']
        netloc_lower = parsed.netloc.lower()
        
        if any(netloc_lower.startswith(blocked) for blocked in blocked_hosts):
            return False, "Cannot crawl localhost or private IP addresses"
        
        return True, "Valid URL"
        
    except Exception as e:
        return False, f"URL validation error: {str(e)}"


def ingest_pdfs(files, progress_callback=None):
    """
    Ingest multiple PDF files with comprehensive error handling.
    
    This function processes each PDF file:
    1. Validates file size and format
    2. Extracts text from each page
    3. Splits text into chunks
    4. Generates embeddings and stores in database
    5. Provides progress updates to the UI
    
    Args:
        files (list): List of uploaded PDF files from Streamlit
        progress_callback (callable): Optional function to report progress
    
    Returns:
        tuple: (success_count, error_count)
    """
    collection = get_collection()
    success_count = 0
    error_count = 0
    total_files = len(files)
    
    logger.info(f"Starting ingestion of {total_files} PDF files")
    
    for i, file in enumerate(files):
        try:
            # Update progress bar in UI
            if progress_callback:
                progress_callback(i / total_files)
            
            # === FILE VALIDATION ===
            
            # Check file size to prevent memory issues
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                logger.warning(f"File {file.name} too large ({file_size_mb:.1f}MB, max {MAX_FILE_SIZE_MB}MB)")
                error_count += 1
                continue
            
            # Verify file extension
            if not file.name.lower().endswith('.pdf'):
                logger.warning(f"File {file.name} is not a PDF")
                error_count += 1
                continue
            
            # === PDF PROCESSING ===
            
            try:
                # Read PDF structure
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # Check if PDF has content
                if total_pages == 0:
                    logger.warning(f"PDF {file.name} has no pages")
                    error_count += 1
                    continue
                
                logger.info(f"Processing PDF {file.name} with {total_pages} pages")
                
            except Exception as e:
                logger.error(f"Error reading PDF {file.name}: {str(e)}")
                error_count += 1
                continue
            
            # === TEXT EXTRACTION ===
            
            # Process each page individually
            pages_processed = 0
            for page_no, page in enumerate(reader.pages, start=1):
                try:
                    # Extract text from page
                    text = page.extract_text()
                    
                    # Skip pages with no meaningful text
                    if text and text.strip():
                        # Split page text into chunks
                        chunks = _splitter.split_text(text)
                        
                        if chunks:
                            # Create metadata for this page
                            metadata = {
                                "type": "pdf",
                                "source": file.name,
                                "page": page_no,
                                "total_pages": total_pages
                            }
                            
                            # Add chunks to database
                            _add_chunks(chunks, metadata, collection)
                            pages_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_no} of {file.name}: {str(e)}")
                    continue
            
            # === FINAL VALIDATION ===
            
            if pages_processed > 0:
                success_count += 1
                logger.info(f"Successfully processed {pages_processed}/{total_pages} pages from {file.name}")
            else:
                error_count += 1
                logger.warning(f"No text extracted from {file.name}")
                
        except Exception as e:
            logger.error(f"Unexpected error processing {file.name}: {str(e)}")
            error_count += 1
    
    # Complete progress bar
    if progress_callback:
        progress_callback(1.0)
    
    logger.info(f"PDF ingestion complete: {success_count} successful, {error_count} errors")
    return success_count, error_count


def ingest_website(url, max_pages=None):
    """
    Crawl a website and ingest content with comprehensive error handling.
    
    This function:
    1. Validates the URL for security
    2. Crawls the main page and discovers links
    3. Extracts meaningful content from each page
    4. Processes and stores content in the database
    5. Limits crawling to the same domain
    
    Args:
        url (str): Starting URL to crawl
        max_pages (int): Maximum pages to crawl (default from config)
    
    Returns:
        int: Number of pages successfully crawled
    
    Raises:
        ValueError: If URL is invalid or insecure
    """
    if max_pages is None:
        max_pages = MAX_PAGES_CRAWL
    
    # === URL VALIDATION ===
    
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        logger.error(f"Invalid URL {url}: {error_msg}")
        raise ValueError(f"Invalid URL: {error_msg}")
    
    # === CRAWLING SETUP ===
    
    visited = set()        # URLs we've already processed
    to_visit = {url}      # URLs we need to process
    collection = get_collection()
    pages_crawled = 0
    
    # Create HTTP session for connection reuse (more efficient)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'MuniRag/1.0 (Municipal Document Crawler)'
    })
    
    logger.info(f"Starting website crawl of {url} (max {max_pages} pages)")
    
    try:
        # === MAIN CRAWLING LOOP ===
        
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop()
            
            # Skip if already processed
            if current_url in visited:
                continue
                
            visited.add(current_url)
            
            try:
                logger.info(f"Crawling: {current_url}")
                
                # === PAGE FETCHING ===
                
                # Fetch page with timeout
                response = session.get(current_url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                # Check if content is HTML
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.warning(f"Skipping non-HTML content: {current_url}")
                    continue
                
                # === CONTENT EXTRACTION ===
                
                # Extract clean text using trafilatura
                text = trafilatura.extract(response.text)
                
                # Skip pages with no meaningful content
                if not text or len(text.strip()) < 100:
                    logger.warning(f"No significant text extracted from {current_url}")
                    continue
                
                # === TEXT PROCESSING ===
                
                # Split into chunks for better retrieval
                chunks = _splitter.split_text(text)
                
                if chunks:
                    # Get page title for metadata
                    page_title = "Unknown"
                    try:
                        metadata_extract = trafilatura.extract_metadata(response.text)
                        if metadata_extract and metadata_extract.get('title'):
                            page_title = metadata_extract['title']
                    except:
                        pass
                    
                    # Create metadata for this page
                    metadata = {
                        "type": "web",
                        "source": current_url,
                        "title": page_title
                    }
                    
                    # Add to database
                    _add_chunks(chunks, metadata, collection)
                    pages_crawled += 1
                
                # === LINK DISCOVERY ===
                
                # Find links on the same domain
                try:
                    base_domain = urlparse(url).netloc
                    links = trafilatura.extract_links(response.text)
                    
                    if links:
                        for link in links:
                            # Convert relative URLs to absolute
                            absolute_link = urljoin(current_url, link)
                            link_domain = urlparse(absolute_link).netloc
                            
                            # Only crawl same-domain links
                            if (link_domain == base_domain and 
                                absolute_link not in visited and
                                len(visited) + len(to_visit) < max_pages):
                                to_visit.add(absolute_link)
                
                except Exception as e:
                    logger.warning(f"Error extracting links from {current_url}: {str(e)}")
                
            except requests.exceptions.Timeout:
                logger.error(f"Timeout crawling {current_url}")
                continue
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error crawling {current_url}: {e.response.status_code}")
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error crawling {current_url}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error crawling {current_url}: {str(e)}")
                continue
    
    finally:
        # Clean up HTTP session
        session.close()
    
    logger.info(f"Website crawling complete: {pages_crawled} pages crawled from {len(visited)} URLs visited")
    return pages_crawled