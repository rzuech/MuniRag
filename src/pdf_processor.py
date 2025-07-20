try:
    import pymupdf4llm
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from typing import List, Dict, Tuple
import re
# Removed langchain dependency - using custom text splitter
from transformers import AutoTokenizer
from src.config import settings
import hashlib

class PDFProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"  # Used for token counting
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text from PDF using PyMuPDF with structure preservation"""
        if PYMUPDF_AVAILABLE:
            try:
                # Extract as markdown to preserve structure
                md_text = pymupdf4llm.to_markdown(
                    pdf_path,
                    page_chunks=True,
                    write_images=False,
                    image_format="png",
                    dpi=150
                )
                
                # Extract metadata using fitz
                doc = fitz.open(pdf_path)
                metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "pages": doc.page_count,
                    "file_path": pdf_path
                }
                doc.close()
                
                return md_text, metadata
            except Exception as e:
                print(f"Error extracting PDF with PyMuPDF: {e}")
        
        # Fallback extraction methods
        try:
            from pypdf import PdfReader
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text, {"file_path": pdf_path, "pages": len(reader.pages)}
        except Exception as e:
            print(f"Error extracting PDF with pypdf: {e}")
            raise Exception(f"Could not extract text from PDF: {pdf_path}")
    
    def chunk_text_semantically(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk text using semantic boundaries with token-based sizing"""
        
        # Use our custom recursive text splitter instead of langchain
        chunks = self._recursive_split_text(
            text=text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            # Generate unique ID for chunk
            chunk_id = hashlib.md5(
                f"{metadata.get('file_path', '')}_{i}_{chunk[:50]}".encode()
            ).hexdigest()
            
            chunk_doc = {
                "id": chunk_id,
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "tokens": self._token_length(chunk)
                }
            }
            
            # Add section context if available
            section_header = self._extract_section_header(text, chunk)
            if section_header:
                chunk_doc["metadata"]["section"] = section_header
            
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def _token_length(self, text: str) -> int:
        """Calculate token length of text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _extract_section_header(self, full_text: str, chunk: str) -> str:
        """Extract the section header for a chunk"""
        # Find chunk position in full text
        chunk_start = full_text.find(chunk)
        if chunk_start == -1:
            return ""
        
        # Look backwards for the nearest header
        text_before = full_text[:chunk_start]
        headers = re.findall(r'^#{1,4}\s+(.+)$', text_before, re.MULTILINE)
        
        return headers[-1] if headers else ""
    
    def _recursive_split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Custom recursive text splitter to replace langchain's RecursiveCharacterTextSplitter
        
        This implementation:
        1. Tries to split on natural boundaries (headers, paragraphs, sentences)
        2. Respects chunk size limits based on token count
        3. Maintains overlap between chunks for context continuity
        4. Works hierarchically through separators
        """
        # Define separators in order of preference (most to least desirable split points)
        separators = [
            "\n## ",   # H2 headers - best split point
            "\n### ",  # H3 headers - very good split point
            "\n#### ", # H4 headers - good split point
            "\n\n",    # Paragraphs - decent split point
            "\n",      # Lines - okay split point
            ". ",      # Sentences - acceptable split point
            ", ",      # Clauses - less ideal
            " ",       # Words - not ideal but necessary
            ""         # Characters - last resort
        ]
        
        # If text is already small enough, return it as a single chunk
        if self._token_length(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Try each separator in order
        for separator in separators:
            if separator and separator in text:
                # Split by this separator
                parts = text.split(separator)
                
                # Reconstruct chunks respecting size limits
                chunks = []
                current_chunk = ""
                
                for i, part in enumerate(parts):
                    # Add separator back (except for first part)
                    if i > 0 and separator:
                        part = separator + part
                    
                    # Check if adding this part would exceed chunk size
                    potential_chunk = current_chunk + part
                    if self._token_length(potential_chunk) <= chunk_size:
                        current_chunk = potential_chunk
                    else:
                        # Current chunk is full, save it
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        # Start new chunk with overlap
                        if chunks and chunk_overlap > 0:
                            # Get overlap from end of previous chunk
                            overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                            current_chunk = overlap_text + part
                        else:
                            current_chunk = part
                
                # Don't forget the last chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If all chunks are within size limits, we're done
                all_valid = all(self._token_length(chunk) <= chunk_size for chunk in chunks)
                if all_valid and chunks:
                    return chunks
        
        # If no separator worked, fall back to character-level splitting
        return self._split_by_char_count(text, chunk_size, chunk_overlap)
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Extract overlap text from the end of a chunk.
        Tries to break at word boundaries for cleaner overlap.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Get the last 'overlap_size' tokens
        if len(tokens) <= overlap_size:
            return text
        
        overlap_tokens = tokens[-overlap_size:]
        overlap_text = self.tokenizer.decode(overlap_tokens)
        
        # Try to start at a word boundary
        first_space = overlap_text.find(' ')
        if first_space > 0 and first_space < len(overlap_text) // 2:
            overlap_text = overlap_text[first_space + 1:]
        
        return overlap_text
    
    def _split_by_char_count(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Fallback method to split text by character count when no good separators exist.
        Still respects token-based chunk size.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Find end position for this chunk
            end = start
            while end < text_length:
                chunk_candidate = text[start:end]
                if self._token_length(chunk_candidate) > chunk_size:
                    # We've exceeded the limit, back up
                    end = max(start + 1, end - 1)
                    break
                end += 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            if chunk_overlap > 0 and end < text_length:
                # Calculate overlap in characters (approximate)
                overlap_text = self._get_overlap_text(chunk, chunk_overlap)
                start = end - len(overlap_text)
            else:
                start = end
            
            # Prevent infinite loop
            if start <= end - len(chunk):
                start = end
        
        return chunks