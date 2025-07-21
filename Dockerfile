# Use PyTorch base image to skip 2GB download
# This saves ~2GB download time since PyTorch is pre-installed
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install requirements (PyTorch already included in base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/chroma_data /app/qdrant_storage /app/Test-PDFs /app/test_results

# Copy test PDFs if they exist (but they're gitignored)
# This allows local testing while preventing git commits
COPY Test-PDFs/*.pdf /app/Test-PDFs/ || true

# Set up Hugging Face cache directory with proper permissions
# This prevents permission issues when downloading models
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME

# Expose ports
EXPOSE 8000 8501

# Pre-download embedding model to avoid runtime issues
# Using BGE model now instead of Jina
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')" || true

# Ensure startup script is executable
RUN chmod +x /app/start.sh

# Use shell form to ensure proper execution
CMD ["/bin/bash", "/app/start.sh"]