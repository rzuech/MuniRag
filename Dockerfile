FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN pip3 install -r requirements.txt

# Download models during build (optional, for faster startup)
# RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/chroma_data /app/qdrant_storage

# Expose ports
EXPOSE 8000 8501

# Copy the startup script from the source
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
