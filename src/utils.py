"""
=============================================================================
UTILS.PY - Utility Functions for MuniRag
=============================================================================

This module provides utility functions used across the MuniRag application,
particularly for prompt generation and text processing.

PURPOSE:
- Build sophisticated prompts for the AI model
- Format and organize retrieved documents
- Provide helper functions for text processing
- Handle file operations and data formatting

PROMPT ENGINEERING:
The build_prompt function is crucial for RAG quality. It:
1. Organizes sources by type (PDF vs web)
2. Provides clear structure for the AI
3. Includes metadata like page numbers
4. Sets clear instructions for citation

USAGE:
    from utils import build_prompt, sanitize_filename
    
    # Build a prompt for AI generation
    prompt = build_prompt(context_docs, user_question)
    
    # Clean up filenames
    clean_name = sanitize_filename("My Document (2024).pdf")
"""

from datetime import datetime
from src.logger import get_logger

# Get a logger specific to this module
logger = get_logger("utils")


def build_prompt(context_docs, user_q):
    """
    Build an enhanced prompt with better context organization for AI generation.
    
    This function creates a structured prompt that helps the AI understand:
    1. What documents are available as sources
    2. How to cite those sources properly
    3. What question needs to be answered
    4. How to format the response
    
    The prompt engineering here is critical for RAG quality:
    - Clear source organization improves citation accuracy
    - Detailed instructions reduce hallucination
    - Structured format makes responses more professional
    
    Args:
        context_docs (list): List of (document_text, metadata) tuples
        user_q (str): The user's question
    
    Returns:
        str: Formatted prompt ready for the AI model
        
    Example:
        docs = [("Budget text...", {"type": "pdf", "source": "budget.pdf"})]
        prompt = build_prompt(docs, "What is the road budget?")
    """
    # Handle case where no relevant documents were found
    if not context_docs:
        logger.warning("No context documents provided for prompt building")
        return (
            "You are an AI assistant for municipal documents. "
            "No relevant documents were found for this question. "
            "Please inform the user that you need relevant documents to answer their question."
        )
    
    # === ORGANIZE SOURCES BY TYPE ===
    # Separate PDF documents from web content for better organization
    
    pdf_sources = []
    web_sources = []
    
    for i, (doc, meta) in enumerate(context_docs):
        source_info = {
            'index': i + 1,  # 1-based indexing for user-friendly citations
            'content': doc,
            'metadata': meta
        }
        
        # Categorize by document type
        if meta.get('type') == 'pdf':
            pdf_sources.append(source_info)
        else:
            web_sources.append(source_info)
    
    # === BUILD CONTEXT SECTION ===
    # Create a well-structured context section with clear source identification
    
    context_parts = []
    
    # Add PDF sources first (usually more authoritative)
    if pdf_sources:
        context_parts.append("=== PDF DOCUMENTS ===")
        for source in pdf_sources:
            meta = source['metadata']
            
            # Build descriptive source label
            source_desc = f"{meta.get('source', 'Unknown PDF')}"
            
            # Add page information if available
            if 'page' in meta:
                source_desc += f" (Page {meta['page']}"
                if 'total_pages' in meta:
                    source_desc += f" of {meta['total_pages']}"
                source_desc += ")"
            
            # Add source header and content
            context_parts.append(f"[Source {source['index']}] {source_desc}")
            context_parts.append(f"{source['content']}")
            context_parts.append("")  # Empty line for readability
    
    # Add web sources
    if web_sources:
        context_parts.append("=== WEBSITE CONTENT ===")
        for source in web_sources:
            meta = source['metadata']
            
            # Build descriptive source label with title if available
            source_desc = meta.get('source', 'Unknown Website')
            if 'title' in meta and meta['title'] != 'Unknown':
                source_desc += f" - {meta['title']}"
            
            # Add source header and content
            context_parts.append(f"[Source {source['index']}] {source_desc}")
            context_parts.append(f"{source['content']}")
            context_parts.append("")  # Empty line for readability
    
    # Join all context parts
    context_text = "\n".join(context_parts)
    
    # === BUILD COMPLETE PROMPT ===
    # Create a comprehensive prompt with clear instructions
    
    prompt = f"""You are an AI assistant specializing in municipal government documents and information.

INSTRUCTIONS:
- Answer ONLY based on the provided sources below
- Always cite your sources using [Source X] format where X is the source number
- If the sources don't contain enough information to answer the question completely, say so clearly
- Be helpful and provide specific details when available
- Focus on accuracy and cite multiple sources when they support the same point
- If you find conflicting information in sources, mention this and cite both sources
- Provide page numbers when citing PDF sources

CONTEXT SOURCES:
{context_text}

QUESTION: {user_q}

ANSWER (remember to cite sources like [Source 1] and include page numbers for PDFs):"""

    logger.debug(f"Built prompt with {len(context_docs)} sources for question: {user_q[:100]}...")
    return prompt


def sanitize_filename(filename):
    """
    Sanitize filename for safe storage and display.
    
    This function removes or replaces characters that could cause issues
    in file systems or web displays.
    
    Args:
        filename (str): Original filename
    
    Returns:
        str: Sanitized filename safe for storage
        
    Example:
        clean = sanitize_filename("My Document (2024).pdf")
        # Returns: "My-Document-2024.pdf"
    """
    import re
    
    # Remove or replace invalid characters
    # Keep only word characters, spaces, and hyphens
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    
    # Replace multiple spaces or hyphens with single hyphen
    sanitized = re.sub(r'[-\s]+', '-', sanitized)
    
    # Remove leading/trailing hyphens
    return sanitized.strip('-')


def format_file_size(size_bytes):
    """
    Format file size in human readable format.
    
    Converts bytes to appropriate units (B, KB, MB, GB) for display.
    
    Args:
        size_bytes (int): Size in bytes
    
    Returns:
        str: Formatted size string
        
    Example:
        format_file_size(1024000)  # Returns: "1000.0 KB"
        format_file_size(1500000)  # Returns: "1.4 MB"
    """
    if size_bytes == 0:
        return "0 B"
    
    # Convert to progressively larger units
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} TB"


def get_timestamp():
    """
    Get current timestamp in ISO format.
    
    Returns:
        str: Current timestamp in ISO format
        
    Example:
        timestamp = get_timestamp()
        # Returns: "2024-01-01T12:00:00.123456"
    """
    return datetime.now().isoformat()


def truncate_text(text, max_length=500):
    """
    Truncate text to specified length with ellipsis.
    
    Useful for displaying document previews without overwhelming the interface.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length including ellipsis
    
    Returns:
        str: Truncated text with ellipsis if needed
        
    Example:
        short = truncate_text("Very long document text...", 50)
        # Returns: "Very long document text..." (if under 50 chars)
        # Or: "Very long document text and more stuf..." (if over 50)
    """
    if len(text) <= max_length:
        return text
    
    # Leave room for ellipsis
    return text[:max_length-3] + "..."


def extract_document_summary(doc_text, max_sentences=3):
    """
    Extract a summary of document content for preview.
    
    Args:
        doc_text (str): Full document text
        max_sentences (int): Maximum sentences to include
    
    Returns:
        str: Document summary
    """
    import re
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', doc_text)
    
    # Clean and filter sentences
    clean_sentences = []
    for sentence in sentences[:max_sentences*2]:  # Look at more to find good ones
        sentence = sentence.strip()
        # Skip very short or empty sentences
        if len(sentence) > 20:
            clean_sentences.append(sentence)
            if len(clean_sentences) >= max_sentences:
                break
    
    return '. '.join(clean_sentences) + '.' if clean_sentences else doc_text[:200] + "..."


def validate_question(question):
    """
    Validate user question for basic quality checks.
    
    Args:
        question (str): User question
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not question or not question.strip():
        return False, "Question cannot be empty"
    
    if len(question.strip()) < 3:
        return False, "Question is too short"
    
    if len(question) > 1000:
        return False, "Question is too long (max 1000 characters)"
    
    return True, ""