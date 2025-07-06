"""
=============================================================================
CONFIG.PY - Configuration Management for MuniRag
=============================================================================

This module handles all configuration settings for the MuniRag application.
It loads environment variables from .env files and provides validation.

PURPOSE:
- Centralize all configuration settings in one place
- Load settings from environment variables for flexibility
- Provide sensible defaults for all settings
- Validate configuration to catch errors early

ENVIRONMENT VARIABLES:
- EMBEDDING_MODEL: Which sentence transformer model to use for embeddings
- LLM_MODEL: Which Ollama model to use for generating responses
- OLLAMA_HOST: URL where Ollama service is running
- CHROMA_DIR: Directory to store ChromaDB vector database
- MAX_CHUNK_TOKENS: Maximum size of text chunks for embeddings
- TOP_K: Number of similar documents to retrieve for context
- MAX_FILE_SIZE_MB: Maximum allowed file size for uploads
- MAX_PAGES_CRAWL: Maximum pages to crawl from a website
- REQUEST_TIMEOUT: Timeout for HTTP requests in seconds
- LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
- RESET_DATA_ON_STARTUP: Whether to clear all data on startup

USAGE:
    from config import EMBEDDING_MODEL, validate_config
    validate_config()  # Check if all settings are valid
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows easy configuration without changing code
load_dotenv()

# === CORE AI MODEL SETTINGS ===
# These control which AI models are used for different tasks

# Embedding model for converting text to vectors
# intfloat/e5-large-v2 is a good balance of quality and speed
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")

# Language model for generating responses
# phi3:mini is lightweight, llama3:8b is more capable but slower
LLM_MODEL = os.getenv("LLM_MODEL", "phi3:mini")

# URL where Ollama service is running
# In Docker, this points to the ollama container
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# === DATA STORAGE SETTINGS ===
# These control where and how data is stored

# Directory for ChromaDB vector database
# This is where document embeddings are stored
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_data")

# === PROCESSING SETTINGS ===
# These control how documents are processed

# Maximum size of text chunks for embeddings (in tokens)
# Smaller chunks = more precise matches, larger chunks = more context
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", 500))

# Number of similar documents to retrieve for answering questions
# More documents = more context but slower processing
TOP_K = int(os.getenv("TOP_K", 4))

# === SECURITY AND LIMITS ===
# These settings prevent abuse and resource exhaustion

# Maximum file size for PDF uploads (in MB)
# Prevents users from uploading extremely large files
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))

# Maximum number of pages to crawl from a website
# Prevents excessive crawling that could overload the system
MAX_PAGES_CRAWL = int(os.getenv("MAX_PAGES_CRAWL", 20))

# Timeout for HTTP requests (in seconds)
# Prevents hanging on slow/unresponsive websites
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))

# === LOGGING SETTINGS ===
# Control how much detail is logged

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === ADMINISTRATIVE SETTINGS ===
# Settings for maintenance and testing

# Whether to reset all data when the application starts
# Useful for development/testing but dangerous in production
RESET_DATA_ON_STARTUP = os.getenv("RESET_DATA_ON_STARTUP", "false").lower() == "true"


def validate_config():
    """
    Validate that all configuration settings are present and valid.
    
    This function checks that:
    - All required environment variables are set
    - Numeric values are positive and reasonable
    - URLs are properly formatted
    - Directories can be created if they don't exist
    
    Raises:
        ValueError: If any configuration is invalid
        
    Example:
        try:
            validate_config()
            print("Configuration is valid")
        except ValueError as e:
            print(f"Configuration error: {e}")
    """
    errors = []
    
    # === CHECK REQUIRED VARIABLES ===
    # These settings must be provided and non-empty
    required_vars = {
        'EMBEDDING_MODEL': EMBEDDING_MODEL,
        'LLM_MODEL': LLM_MODEL,
        'OLLAMA_HOST': OLLAMA_HOST,
        'CHROMA_DIR': CHROMA_DIR
    }
    
    for var_name, var_value in required_vars.items():
        if not var_value:
            errors.append(f"Missing required environment variable: {var_name}")
    
    # === VALIDATE NUMERIC VALUES ===
    # Check that numeric settings are positive and reasonable
    
    if MAX_CHUNK_TOKENS <= 0:
        errors.append("MAX_CHUNK_TOKENS must be positive")
    elif MAX_CHUNK_TOKENS > 2000:
        errors.append("MAX_CHUNK_TOKENS is too large (max 2000 recommended)")
    
    if TOP_K <= 0:
        errors.append("TOP_K must be positive")
    elif TOP_K > 20:
        errors.append("TOP_K is too large (max 20 recommended)")
    
    if MAX_FILE_SIZE_MB <= 0:
        errors.append("MAX_FILE_SIZE_MB must be positive")
    elif MAX_FILE_SIZE_MB > 500:
        errors.append("MAX_FILE_SIZE_MB is too large (max 500 MB recommended)")
    
    if MAX_PAGES_CRAWL <= 0:
        errors.append("MAX_PAGES_CRAWL must be positive")
    elif MAX_PAGES_CRAWL > 100:
        errors.append("MAX_PAGES_CRAWL is too large (max 100 recommended)")
    
    if REQUEST_TIMEOUT <= 0:
        errors.append("REQUEST_TIMEOUT must be positive")
    elif REQUEST_TIMEOUT > 300:
        errors.append("REQUEST_TIMEOUT is too large (max 300 seconds recommended)")
    
    # === VALIDATE URL FORMAT ===
    # Check that OLLAMA_HOST is a valid URL
    if not OLLAMA_HOST.startswith(('http://', 'https://')):
        errors.append("OLLAMA_HOST must start with http:// or https://")
    
    # === VALIDATE LOG LEVEL ===
    # Check that LOG_LEVEL is a valid Python logging level
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if LOG_LEVEL.upper() not in valid_log_levels:
        errors.append(f"LOG_LEVEL must be one of: {', '.join(valid_log_levels)}")
    
    # === VALIDATE DIRECTORY PATHS ===
    # Check that we can create the data directory if it doesn't exist
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
    except OSError as e:
        errors.append(f"Cannot create CHROMA_DIR '{CHROMA_DIR}': {e}")
    
    # If any validation failed, raise an error with all issues
    if errors:
        raise ValueError("; ".join(errors))
    
    return True


def get_config_summary():
    """
    Get a summary of current configuration settings.
    
    Returns:
        dict: Dictionary containing all configuration values
        
    Example:
        config = get_config_summary()
        print(f"Using model: {config['EMBEDDING_MODEL']}")
    """
    return {
        'EMBEDDING_MODEL': EMBEDDING_MODEL,
        'LLM_MODEL': LLM_MODEL,
        'OLLAMA_HOST': OLLAMA_HOST,
        'CHROMA_DIR': CHROMA_DIR,
        'MAX_CHUNK_TOKENS': MAX_CHUNK_TOKENS,
        'TOP_K': TOP_K,
        'MAX_FILE_SIZE_MB': MAX_FILE_SIZE_MB,
        'MAX_PAGES_CRAWL': MAX_PAGES_CRAWL,
        'REQUEST_TIMEOUT': REQUEST_TIMEOUT,
        'LOG_LEVEL': LOG_LEVEL,
        'RESET_DATA_ON_STARTUP': RESET_DATA_ON_STARTUP
    }
