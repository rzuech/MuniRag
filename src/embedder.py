from sentence_transformers import SentenceTransformer
import torch
from config import EMBEDDING_MODEL

def get_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBEDDING_MODEL, device=device)

def embed(texts):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True)
