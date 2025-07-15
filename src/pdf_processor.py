try:
    import pymupdf4llm
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from typing import List, Dict, Tuple
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
                
                # Extract metadata
                doc = pymupdf4llm.Document(pdf_path)
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
        
        # Custom text splitter that respects markdown structure
        separators = [
            "\n## ",  # H2 headers
            "\n### ", # H3 headers
            "\n#### ", # H4 headers
            "\n\n",   # Paragraphs
            "\n",     # Lines
            ". ",     # Sentences
            " ",      # Words
            ""        # Characters
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=self._token_length,
            is_separator_regex=False
        )
        
        # Split into chunks
        chunks = splitter.split_text(text)
        
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