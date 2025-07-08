from sentence_transformers import CrossEncoder
from typing import List, Dict
import torch

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 5
    ) -> List[Dict]:
        """Rerank documents based on relevance to query"""
        
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [(query, doc["content"]) for doc in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add rerank scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        # Sort by rerank score
        reranked = sorted(
            documents, 
            key=lambda x: x["rerank_score"], 
            reverse=True
        )
        
        return reranked[:top_k]