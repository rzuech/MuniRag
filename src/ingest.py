from pathlib import Path
import trafilatura, PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedder import embed
from retriever import get_collection
from config import MAX_CHUNK_TOKENS

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_TOKENS,
    chunk_overlap=int(MAX_CHUNK_TOKENS * 0.1)
)

def _add_chunks(chunks, metadata, collection):
    embeddings = embed(chunks).tolist()
    ids = [f"{metadata['source']}::chunk{i}" for i, _ in enumerate(chunks)]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[metadata]*len(chunks)
    )

def ingest_pdfs(files):
    collection = get_collection()
    for file in files:
        reader = PyPDF2.PdfReader(file)
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                chunks = _splitter.split_text(text)
                _add_chunks(
                    chunks,
                    metadata={"type": "pdf", "source": file.name, "page": page_no},
                    collection=collection
                )

def ingest_website(url, max_pages=20):
    """Fetch the root page + same‑domain links (breadth‑1)."""
    visited, to_visit = set(), {url}
    collection = get_collection()

    while to_visit and len(visited) < max_pages:
        link = to_visit.pop()
        visited.add(link)
        downloaded = trafilatura.fetch_url(link)
        text = trafilatura.extract(downloaded)
        if not text:
            continue
        chunks = _splitter.split_text(text)
        _add_chunks(
            chunks,
            metadata={"type": "web", "source": link},
            collection=collection
        )
        # discover more links
        for l in trafilatura.extract_links(downloaded) or []:
            if l.startswith(url) and l not in visited:
                to_visit.add(l)
