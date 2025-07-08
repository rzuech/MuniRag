import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from .config import settings

class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the model based on type
        if "jina" in self.model_name.lower():
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=self.device
            )
            # For Jina v3, use task-specific encoding
            self.encode_kwargs = {'task': 'retrieval.query'}
        else:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            self.encode_kwargs = {}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents for storage"""
        # For Jina v3, use document-specific task
        if "jina" in self.model_name.lower():
            embeddings = self.model.encode(
                texts,
                task='retrieval.passage',
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query for search"""
        # For Jina v3, use query-specific task
        if "jina" in self.model_name.lower():
            embedding = self.model.encode(
                query,
                task='retrieval.query',
                convert_to_numpy=True
            )
        else:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True
            )
        return embedding.tolist()

# Legacy compatibility functions
def embed(texts):
    """Legacy function for backward compatibility"""
    embedder = EmbeddingModel()
    if isinstance(texts, str):
        return [embedder.embed_query(texts)]
    elif isinstance(texts, list) and len(texts) == 1:
        return [embedder.embed_query(texts[0])]
    else:
        return embedder.embed_documents(texts)
