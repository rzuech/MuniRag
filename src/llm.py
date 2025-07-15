"""
=============================================================================
LLM.PY - Language Model Interface for MuniRag
=============================================================================

This module handles communication with Ollama for generating AI responses.
It provides streaming responses for better user experience and comprehensive
error handling for production stability.

PURPOSE:
- Interface with Ollama service to generate responses
- Stream responses in real-time for better UX
- Handle connection issues and timeouts gracefully
- Validate prompts and manage model interactions

WHAT IS OLLAMA?
Ollama is a local AI service that runs language models like Llama, Phi, and Gemma.
It provides an API similar to OpenAI but runs entirely on your own hardware,
ensuring privacy and no external API costs.

HOW STREAMING WORKS:
Instead of waiting for the complete response, we receive it piece by piece
and display it incrementally. This makes the interface feel more responsive
and shows progress to users.

USAGE:
    from llm import stream_answer, check_ollama_connection
    
    # Check if Ollama is available
    if check_ollama_connection():
        # Stream a response
        for chunk in stream_answer("What is the city budget?"):
            print(chunk, end='')
"""

import requests
import json
from src.config import settings
from src.logger import get_logger

# Get a logger specific to this module
logger = get_logger("llm")


def check_ollama_connection():
    """
    Check if Ollama service is available and responsive.
    
    This function performs a quick health check to verify that:
    1. Ollama service is running
    2. It's accessible on the configured host/port
    3. It can respond to API requests
    
    Returns:
        bool: True if Ollama is available, False otherwise
        
    Example:
        if check_ollama_connection():
            print("Ollama is ready!")
        else:
            print("Ollama is not available")
    """
    try:
        # Make a quick request to the tags endpoint
        # This is a lightweight way to test connectivity
        response = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        
        logger.debug("Ollama connection check successful")
        return True
        
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama - service may not be running")
        return False
    except requests.exceptions.Timeout:
        logger.error("Ollama connection timeout - service may be overloaded")
        return False
    except Exception as e:
        logger.error(f"Ollama connection check failed: {str(e)}")
        return False


def stream_answer(prompt):
    """
    Stream AI response from Ollama with comprehensive error handling.
    
    This function sends a prompt to Ollama and yields response chunks
    as they arrive, providing real-time feedback to users.
    
    The function handles various error scenarios:
    - Empty or invalid prompts
    - Ollama service unavailable
    - Network timeouts
    - Invalid JSON responses
    - Model errors
    
    Args:
        prompt (str): The formatted prompt to send to the AI model
    
    Yields:
        str: Individual response chunks as they arrive
        
    Example:
        prompt = "What is the municipal budget for 2024?"
        for chunk in stream_answer(prompt):
            print(chunk, end='', flush=True)
    """
    # === INPUT VALIDATION ===
    
    if not prompt or not prompt.strip():
        logger.error("Empty prompt provided to stream_answer")
        yield "Error: Empty prompt provided"
        return
    
    # === CONNECTIVITY CHECK ===
    
    # Verify Ollama is available before attempting to generate
    if not check_ollama_connection():
        error_msg = "Cannot connect to Ollama service. Please check if Ollama is running and accessible."
        logger.error(error_msg)
        yield error_msg
        return
    
    try:
        # === REQUEST SETUP ===
        
        # Try chat API first (supports system messages), fall back to generate API
        use_chat_api = True
        url = f"{settings.OLLAMA_HOST}/api/chat"
        
        # Configure the generation request
        if use_chat_api:
            # Build messages array with system message for better RAG focus
            messages = [
                {
                    "role": "system",
                    "content": settings.SYSTEM_MESSAGE
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            payload = {
                "model": settings.LLM_MODEL,
                "messages": messages,
                "stream": True,  # Enable streaming for real-time responses
                "options": {
                    # Use parameters from settings for consistency
                    "temperature": settings.LLM_TEMPERATURE,
                    "top_p": settings.LLM_TOP_P,
                    "num_predict": settings.LLM_MAX_TOKENS
                }
            }
        else:
            # Fallback to old generate API format
            url = f"{settings.OLLAMA_HOST}/api/generate"
            payload = {
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "top_p": settings.LLM_TOP_P,
                    "num_predict": settings.LLM_MAX_TOKENS
                }
            }
        
        logger.info(f"Sending request to Ollama model: {settings.LLM_MODEL}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # === STREAMING REQUEST ===
        
        # Make the streaming request to Ollama
        with requests.post(url, json=payload, stream=True, timeout=settings.REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            
            # Track response for logging
            response_text = ""
            chunk_count = 0
            
            # === RESPONSE PROCESSING ===
            
            # Process each line of the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse JSON response
                        data = json.loads(line.decode('utf-8'))
                        
                        # === ERROR HANDLING ===
                        
                        # Check for errors in the response
                        if 'error' in data:
                            error_msg = data['error']
                            logger.error(f"Ollama returned error: {error_msg}")
                            yield f"Error from Ollama: {error_msg}"
                            return
                        
                        # === CONTENT EXTRACTION ===
                        
                        # Extract the response chunk (different field for chat vs generate API)
                        if use_chat_api:
                            # Chat API returns message with content
                            message = data.get('message', {})
                            response_chunk = message.get('content', '')
                        else:
                            # Generate API returns response directly
                            response_chunk = data.get('response', '')
                        
                        if response_chunk:
                            response_text += response_chunk
                            chunk_count += 1
                            yield response_chunk
                        
                        # === COMPLETION CHECK ===
                        
                        # Check if generation is complete
                        if data.get('done', False):
                            logger.info(f"Response generation completed ({chunk_count} chunks, {len(response_text)} characters)")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in Ollama response: {str(e)}")
                        # Continue processing other lines
                        continue
                    except KeyError as e:
                        logger.error(f"Unexpected response format from Ollama: {str(e)}")
                        # Continue processing other lines
                        continue
            
            # === FINAL VALIDATION ===
            
            # Check if we received any meaningful response
            if not response_text:
                logger.warning("No response text generated by Ollama")
                yield "No response generated. Please try rephrasing your question."
    
    # === ERROR HANDLING ===
    
    except requests.exceptions.Timeout:
        error_msg = f"Request timeout after {settings.REQUEST_TIMEOUT} seconds. The question may be too complex or the service is overloaded."
        logger.error(error_msg)
        yield error_msg
    
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Ollama service. Please check if Ollama is running and accessible."
        logger.error(error_msg)
        yield error_msg
    
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error from Ollama: {e.response.status_code}"
        if e.response.status_code == 404:
            error_msg += f" - Model '{settings.LLM_MODEL}' may not be available. Please check model name or pull the model first."
        logger.error(f"{error_msg} - {e.response.text}")
        yield error_msg
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error communicating with Ollama: {str(e)}"
        logger.error(error_msg)
        yield error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error during response generation: {str(e)}"
        logger.error(error_msg)
        yield error_msg


def get_available_models():
    """
    Get a list of available models from Ollama.
    
    Returns:
        list: List of available model names, empty list if error
    """
    try:
        response = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        model_names = [model['name'] for model in data.get('models', [])]
        
        logger.info(f"Found {len(model_names)} available models in Ollama")
        return model_names
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        return []