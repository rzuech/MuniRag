from typing import List, Dict, Optional
import ollama
from src.embedder import EmbeddingModel
from src.vector_store import VectorStore
from src.reranker import Reranker
from src.pdf_processor import PDFProcessor
from src.config import settings
import asyncio

class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        self.pdf_processor = PDFProcessor()
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client()
        
    def ingest_pdf(self, pdf_path: str) -> Dict:
        """Process and ingest a PDF file"""
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text, metadata = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Chunk text
        chunks = self.pdf_processor.chunk_text_semantically(text, metadata)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        
        # Store in vector database
        self.vector_store.add_documents(chunks, embeddings)
        
        return {
            "status": "success",
            "chunks_created": len(chunks),
            "metadata": metadata
        }
    
    async def query(
        self, 
        query: str, 
        use_reranking: bool = True
    ) -> Dict:
        """Query the RAG system"""
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve documents
        if settings.USE_HYBRID_SEARCH:
            retrieved_docs = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=settings.RETRIEVAL_TOP_K
            )
        else:
            retrieved_docs = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=settings.RETRIEVAL_TOP_K
            )
        
        # Rerank if enabled
        if use_reranking and retrieved_docs:
            retrieved_docs = self.reranker.rerank(
                query=query,
                documents=retrieved_docs,
                top_k=settings.RERANK_TOP_K
            )
        
        # Generate response
        response = await self._generate_response(query, retrieved_docs)
        
        return {
            "query": query,
            "response": response,
            "sources": self._format_sources(retrieved_docs),
            "retrieved_chunks": len(retrieved_docs)
        }
    
    async def _generate_response(
        self, 
        query: str, 
        context_docs: List[Dict]
    ) -> str:
        """Generate response using LLM"""
        
        # Prepare context
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc['content']}" 
            for i, doc in enumerate(context_docs)
        ])
        
        # Prepare prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {query}

Instructions:
- Answer based solely on the provided context
- If the answer cannot be found in the context, say so
- Be concise but complete in your response
- Cite which document(s) you used for your answer

Answer:"""
        
        # Generate with Ollama
        try:
            response = self.ollama_client.generate(
                model=settings.LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": settings.LLM_TEMPERATURE,
                    "num_predict": settings.LLM_MAX_TOKENS
                }
            )
            return response["response"]
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _format_sources(self, docs: List[Dict]) -> List[Dict]:
        """Format source documents for response"""
        sources = []
        for doc in docs[:3]:  # Top 3 sources
            sources.append({
                "content_preview": doc["content"][:200] + "...",
                "metadata": doc.get("metadata", {}),
                "relevance_score": doc.get("rerank_score", doc.get("score", 0))
            })
        return sources