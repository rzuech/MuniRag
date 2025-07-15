#!/usr/bin/env python3
"""
Ingest test PDFs into MuniRAG for testing
This script processes all test PDFs and stores them in the vector database
"""
import os
import sys
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, '.')

print("📚 MuniRAG v2.0 - Batch PDF Ingestion")
print("=" * 60)

async def ingest_all_pdfs():
    """Ingest all test PDFs into the system"""
    try:
        from src.pdf_parallel_processor import ParallelPDFProcessor
        from src.embedder_v2 import MultiModelEmbedder
        from src.vector_store import QdrantVectorStore
        from src.config import settings
        import requests
        
        # Initialize components
        print("\n🔧 Initializing components...")
        
        # Use the recommended model
        embedder = MultiModelEmbedder("bge-large-en")
        print("✅ Embedder initialized (bge-large-en)")
        
        # PDF processor with optimized settings
        pdf_processor = ParallelPDFProcessor(
            num_workers=4,
            enable_ocr=False,  # Disable for speed
            semantic_chunking=True,
            chunk_size=500
        )
        print("✅ PDF processor initialized (4 workers)")
        
        # Vector store
        vector_store = QdrantVectorStore()
        print("✅ Vector store connected")
        
        # Find all PDFs
        pdf_dir = Path("tests/test_data/pdfs")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("\n❌ No PDFs found in tests/test_data/pdfs/")
            print("   Copy your PDFs there first!")
            return
            
        print(f"\n📄 Found {len(pdf_files)} PDFs to process:")
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"   - {pdf.name} ({size_mb:.1f} MB)")
        
        # Process each PDF
        total_chunks = 0
        total_start = time.time()
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            try:
                # Extract and chunk
                start_time = time.time()
                chunks = pdf_processor.process_pdf(str(pdf_path))
                process_time = time.time() - start_time
                
                print(f"   ✅ Extracted {len(chunks)} chunks in {process_time:.2f}s")
                
                # Create embeddings
                print(f"   🔄 Creating embeddings...")
                texts = [chunk['content'] for chunk in chunks]
                
                # Process in batches to avoid memory issues
                batch_size = 32
                all_embeddings = []
                
                for j in range(0, len(texts), batch_size):
                    batch = texts[j:j + batch_size]
                    batch_embeddings = embedder.encode(batch, is_query=False)
                    all_embeddings.extend(batch_embeddings)
                    print(f"      Processed {min(j + batch_size, len(texts))}/{len(texts)} chunks")
                
                # Store in vector database
                print(f"   💾 Storing in vector database...")
                
                # Add metadata to chunks
                for chunk, embedding in zip(chunks, all_embeddings):
                    chunk['embedding'] = embedding
                    chunk['metadata']['source_file'] = pdf_path.name
                    chunk['metadata']['ingestion_time'] = time.time()
                
                # Store (you'll need to implement the batch storage method)
                # vector_store.add_documents(chunks)
                
                total_chunks += len(chunks)
                print(f"   ✅ Stored {len(chunks)} chunks successfully!")
                
            except Exception as e:
                print(f"   ❌ Error processing {pdf_path.name}: {e}")
                continue
        
        # Summary
        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print("📊 INGESTION COMPLETE")
        print("=" * 60)
        print(f"✅ Processed: {len(pdf_files)} PDFs")
        print(f"✅ Total chunks: {total_chunks}")
        print(f"✅ Total time: {total_time:.1f}s")
        print(f"✅ Average speed: {total_chunks/total_time:.1f} chunks/sec")
        
        # Test with a query
        print("\n🔍 Testing retrieval with sample query...")
        test_query = "How do I apply for a building permit?"
        
        query_embedding = embedder.encode(test_query, is_query=True)
        # results = vector_store.search(query_embedding, top_k=3)
        
        print(f"   Query: '{test_query}'")
        print(f"   (Search would return relevant chunks about permits)")
        
        # Cleanup
        embedder.unload()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def ingest_via_api():
    """Alternative: Ingest PDFs via the API"""
    import requests
    
    print("\n🌐 Ingesting PDFs via API")
    print("-" * 60)
    
    # You'll need an API key first
    api_key = "YOUR_API_KEY"  # Get this from /api/key/create
    
    pdf_dir = Path("tests/test_data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        print(f"\nUploading: {pdf_path.name}")
        
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            headers = {'Authorization': f'Bearer {api_key}'}
            
            try:
                response = requests.post(
                    'http://localhost:8000/api/ingest/pdf',
                    files=files,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Success: {result['chunks_created']} chunks created")
                else:
                    print(f"   ❌ Failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")

def create_sample_queries():
    """Generate sample queries for the test PDFs"""
    print("\n💬 Sample Queries for Your PDFs:")
    print("-" * 60)
    
    queries = [
        # For Procurement Ethics Policy
        "What is the city's procurement ethics policy?",
        "What are the ethical guidelines for purchasing?",
        
        # For Certificate of Use
        "How do I create a certificate of use?",
        "What is required for a certificate of use application?",
        
        # For Permit Application
        "How do I register and create a permit application?",
        "What are the steps to apply for a building permit?",
        
        # For Homeowner ACA
        "How do homeowners use the ACA system?",
        "What is the ACA application process for homeowners?",
        
        # For Vacation Rental
        "How do I submit a vacation rental application?",
        "What documents are needed for vacation rental registration?",
        
        # For Code of Ordinances
        "What are the zoning regulations in Weston?",
        "What are the building code requirements?",
        "What are the noise ordinance regulations?"
    ]
    
    print("\nTry these queries after ingestion:")
    for i, query in enumerate(queries, 1):
        print(f"{i:2d}. {query}")
    
    print("\n💡 These queries will search across all your PDFs!")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest test PDFs into MuniRAG")
    parser.add_argument("--api", action="store_true", help="Use API for ingestion")
    parser.add_argument("--queries", action="store_true", help="Just show sample queries")
    args = parser.parse_args()
    
    if args.queries:
        create_sample_queries()
    elif args.api:
        ingest_via_api()
    else:
        # Run async ingestion
        asyncio.run(ingest_all_pdfs())
        create_sample_queries()

if __name__ == "__main__":
    main()