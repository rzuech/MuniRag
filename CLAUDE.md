# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
MuniRag is a Retrieval-Augmented Generation (RAG) application built with Streamlit, ChromaDB, and Ollama. It allows municipalities to upload PDFs and crawl websites to create a searchable knowledge base that can answer questions using local AI models.

## Architecture
- **Frontend**: Streamlit web interface (app.py:8)
- **Document Ingestion**: PDF processing with PyPDF2 and website crawling with trafilatura (ingest.py)
- **Embeddings**: Uses sentence-transformers with configurable models (embedder.py)
- **Vector Database**: ChromaDB for storing document embeddings (retriever.py)
- **LLM**: Ollama for question answering with streaming responses (llm.py)
- **Configuration**: Environment-based config management (config.py)

## Development Commands

### Running the Application
```bash
# Local development (requires GPU setup)
cp .env.example .env
docker compose up --build

# Access at http://localhost:8501
```

### Container Management
```bash
# Stop services
docker compose down

# View logs
docker compose logs munirag
docker compose logs ollama

# Rebuild after changes
docker compose up --build
```

## Key Configuration
- Environment variables in `.env` control models and behavior
- `EMBEDDING_MODEL`: Sentence transformer model (default: intfloat/e5-large-v2)
- `LLM_MODEL`: Ollama model for responses (default: llama3:8b)
- `OLLAMA_HOST`: Ollama service endpoint (default: http://ollama:11434)
- `MAX_CHUNK_TOKENS`: Text chunk size for embeddings (default: 300)
- `TOP_K`: Number of context documents to retrieve (default: 4)

## Data Flow
1. Documents ingested via ingest.py (PDFs or websites)
2. Text split into chunks and embedded via embedder.py
3. Embeddings stored in ChromaDB via retriever.py
4. User queries embedded and matched against stored documents
5. Retrieved context passed to LLM via llm.py for answer generation
6. Streaming responses displayed in Streamlit interface

## Important Notes
- GPU support required for optimal performance (NVIDIA Container Toolkit)
- ChromaDB data persisted in ./chroma_data volume
- Ollama models cached in ./ollama_models volume
- Document uploads stored in ./data volume
- Main application entry point: src/app.py:15