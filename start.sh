#!/bin/bash
# CRITICAL: Use persistent model cache to prevent re-downloading
export HF_HOME=/app/models/huggingface
export TRANSFORMERS_CACHE=/app/models/huggingface
export SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers
export HF_HUB_CACHE=/app/models/huggingface
export PYTHONPATH=/app:$PYTHONPATH

# Suppress ONLY FlashAttention warnings, keep other logs
export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_DISABLE_FLASH_ATTN_WARNING=1
export TOKENIZERS_PARALLELISM=false
# Enable Python and app logging
export PYTHONUNBUFFERED=1
export LOG_LEVEL=INFO

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0

# Temporarily disable offline mode to allow model downloads
# export HF_HUB_OFFLINE=1
echo "Online mode enabled for model downloads"

# Initialize Qdrant (purge if configured)
echo "Initializing Qdrant..."
cd /app && python3 -c "
from src.qdrant_manager import get_qdrant_manager
manager = get_qdrant_manager()
manager.initialize_on_startup()
"

# Start FastAPI in background with proper module path
cd /app && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

# Give FastAPI time to start
sleep 5

# Start Streamlit in foreground
cd /app && streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0