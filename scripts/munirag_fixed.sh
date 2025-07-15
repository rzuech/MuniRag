#!/bin/bash
# MuniRAG helper script - Fixed for Unix line endings

case "$1" in
    "check")
        echo "Running system check..."
        docker-compose exec munirag python3 /app/scripts/check_system.py
        ;;
    
    "reset")
        echo "Resetting Qdrant..."
        docker-compose exec munirag python3 /app/scripts/reset_qdrant.py
        ;;
    
    "logs")
        echo "Starting clean log viewer..."
        python3 scripts/clean_logs.py
        ;;
    
    "gpu")
        echo "Testing GPU embeddings..."
        docker-compose exec munirag python3 /app/scripts/test_gpu_embeddings.py
        ;;
    
    "count")
        echo "Checking document count..."
        docker-compose exec munirag python3 -c "from src.vector_store import VectorStore; print(f'Documents: {VectorStore().client.count(\"munirag_docs\")}')"
        ;;
    
    "download-jina")
        echo "Downloading Jina model..."
        docker-compose exec munirag python3 /app/scripts/download_jina_model.py
        ;;
    
    "rebuild")
        echo "Rebuilding with fixes..."
        docker-compose down
        docker-compose up --build -d
        echo "Waiting for services to start..."
        sleep 10
        docker-compose logs --tail 50
        ;;
    
    "test-upload")
        echo "Copy a small test PDF to container..."
        docker cp "Test-PDFs/Procurement Ethics Policy.pdf" munirag-munirag-1:/app/test.pdf
        echo "Test PDF copied to container as /app/test.pdf"
        ;;
    
    *)
        echo "MuniRAG Helper Commands:"
        echo "  bash scripts/munirag_fixed.sh check         - System health check"
        echo "  bash scripts/munirag_fixed.sh reset         - Reset Qdrant database"
        echo "  bash scripts/munirag_fixed.sh logs          - View clean logs"
        echo "  bash scripts/munirag_fixed.sh gpu           - Test GPU embeddings"
        echo "  bash scripts/munirag_fixed.sh count         - Count documents in DB"
        echo "  bash scripts/munirag_fixed.sh download-jina - Download Jina model"
        echo "  bash scripts/munirag_fixed.sh rebuild       - Rebuild containers"
        echo "  bash scripts/munirag_fixed.sh test-upload   - Copy test PDF"
        ;;
esac