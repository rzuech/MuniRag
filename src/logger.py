"""
=============================================================================
LOGGER.PY - Centralized Logging System for MuniRag
=============================================================================

This module provides a centralized logging system for the MuniRag application.
It sets up both console and file logging with automatic log rotation.

PURPOSE:
- Provides structured logging across all modules
- Rotates log files to prevent disk space issues
- Allows different log levels (DEBUG, INFO, WARNING, ERROR)
- Helps with debugging and monitoring in production

FEATURES:
- Console logging for immediate feedback
- File logging with automatic rotation (10MB max, 5 backup files)
- Configurable log levels via environment variables
- Thread-safe logging operations
- Graceful fallback if file logging fails

USAGE:
    from logger import setup_logging
    logger = setup_logging()
    logger.info("Application started")
    logger.error("Something went wrong")
"""

import logging
import sys
import os
from datetime import datetime


def setup_logging(level=logging.INFO):
    """
    Setup comprehensive logging configuration for MuniRag.
    
    This function creates a logger that outputs to both console and file,
    with automatic log rotation to prevent disk space issues.
    
    Args:
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    Default is INFO to balance detail vs. noise
    
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        logger = setup_logging()
        logger.info("PDF processing started")
        logger.error("Failed to connect to Ollama")
    """
    
    # Create logs directory if it doesn't exist
    # This ensures we have a place to store log files
    log_dir = "/app/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a consistent log format for all messages
    # Format: "2024-01-01 12:00:00 - munirag - INFO - Your message here"
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get or create the main logger for MuniRag
    # All modules will use this same logger for consistency
    logger = logging.getLogger('munirag')
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicate log messages
    # This is important when the logger is reinitialized
    logger.handlers.clear()
    
    # === CONSOLE LOGGING ===
    # This handler outputs log messages to the console/terminal
    # Useful for development and Docker container logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # === FILE LOGGING WITH ROTATION ===
    # This handler saves logs to files with automatic rotation
    # When a log file reaches 10MB, it's archived and a new one is created
    try:
        from logging.handlers import RotatingFileHandler
        
        # Create a rotating file handler
        # - maxBytes: Maximum size per log file (10MB)
        # - backupCount: Number of archived log files to keep (5)
        # This prevents logs from consuming too much disk space
        file_handler = RotatingFileHandler(
            f"{log_dir}/munirag.log",
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5           # Keep 5 archived files
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        # If file logging fails (permissions, disk space, etc.),
        # we'll continue with console logging only
        logger.warning(f"Could not setup file logging: {e}")
    
    # Prevent log messages from propagating to parent loggers
    # This avoids duplicate messages in complex logging hierarchies
    logger.propagate = False
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance for a specific module.
    
    This function returns a child logger that inherits the configuration
    from the main MuniRag logger but can be identified by module name.
    
    Args:
        name (str): Name of the module/component requesting the logger
                   If None, returns the main logger
    
    Returns:
        logging.Logger: Logger instance for the specified module
        
    Example:
        logger = get_logger("embedder")
        logger.info("Embedding model loaded")
    """
    if name:
        return logging.getLogger(f'munirag.{name}')
    else:
        return logging.getLogger('munirag')


# Create a default logger instance for immediate use
# Other modules can import this directly: from logger import logger
logger = setup_logging()