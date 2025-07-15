from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    SearchRequest, FieldCondition, MatchValue
)
from typing import List, Dict, Optional
import uuid
from src.config import settings
from rank_bm25 import BM25Okapi
import numpy as np

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = settings.COLLECTION_NAME
        self._ensure_collection()
        
        # For hybrid search
        self.bm25_index = None
        self.document_texts = []
        
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # Get dimension from current embedding model
            dimension = settings.get_embedding_dimension()
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """Add documents with their embeddings to the vector store"""
        points = []
        
        for doc, embedding in zip(documents, embeddings):
            # Create point for Qdrant
            # Generate UUID if ID is None or not provided
            point_id = doc.get("id")
            if point_id is None:
                point_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            )
            points.append(point)
            
            # Store for BM25
            self.document_texts.append(doc["content"])
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Update BM25 index
        self._update_bm25_index()
    
    def _update_bm25_index(self):
        """Update BM25 index for hybrid search"""
        if self.document_texts:
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in self.document_texts]
            self.bm25_index = BM25Okapi(tokenized_docs)
    
    def hybrid_search(
        self, 
        query_embedding: List[float], 
        query_text: str, 
        top_k: int = 10,
        alpha: float = 0.5  # Weight for dense vs sparse
    ) -> List[Dict]:
        """Perform hybrid search combining dense and sparse retrieval"""
        
        # Dense search
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 2  # Get more for re-ranking
        )
        
        # Sparse search with BM25
        sparse_scores = []
        if self.bm25_index:
            query_tokens = query_text.lower().split()
            sparse_scores = self.bm25_index.get_scores(query_tokens)
        
        # Combine scores
        combined_results = []
        
        for i, result in enumerate(dense_results):
            dense_score = result.score
            
            # Find corresponding BM25 score
            doc_index = self._find_document_index(result.payload["content"])
            sparse_score = sparse_scores[doc_index] if doc_index >= 0 else 0
            
            # Normalize and combine scores
            combined_score = alpha * dense_score + (1 - alpha) * self._normalize_score(sparse_score)
            
            combined_results.append({
                "content": result.payload["content"],
                "metadata": result.payload["metadata"],
                "score": combined_score,
                "dense_score": dense_score,
                "sparse_score": sparse_score
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results[:top_k]
    
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """Perform regular dense vector search"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.payload["content"],
                "metadata": result.payload["metadata"],
                "score": result.score
            })
        
        return formatted_results
    
    def _find_document_index(self, content: str) -> int:
        """Find document index in BM25 corpus"""
        try:
            return self.document_texts.index(content)
        except ValueError:
            return -1
    
    def _normalize_score(self, score: float) -> float:
        """Normalize BM25 scores to 0-1 range"""
        if score <= 0:
            return 0
        return 1 / (1 + np.exp(-score))