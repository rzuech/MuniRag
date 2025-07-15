from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from src.rag_pipeline import RAGPipeline
from src.config import settings
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="MuniRAG API", version="2.0.0")

# Initialize RAG pipeline
rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    use_reranking: bool = True

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: list
    retrieved_chunks: int

@app.get("/")
async def root():
    return {
        "message": "MuniRAG API is running",
        "version": "2.0.0",
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL
    }

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Process the PDF
        result = rag.ingest_pdf(tmp_path)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = await rag.query(
            query=request.query,
            use_reranking=request.use_reranking
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )