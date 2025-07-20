"""
Enhanced Streamlit app with multi-model support and dimension mismatch handling.

This is an example of how to integrate the new multi-model capabilities into
the existing MuniRAG application.
"""

import streamlit as st
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import enhanced components
from src.vector_store_v2 import MultiModelVectorStore, VectorStore
from src.config_v2 import enhanced_settings
from src.model_migration import ModelMigrationManager, render_model_selection_ui

# Use optimized ingest for better GPU performance
try:
    from src.ingest_optimized import ingest_pdfs_optimized
    logger.info("Using optimized PDF ingestion with GPU batching")
except ImportError:
    from src.ingest import ingest_pdfs as ingest_pdfs_optimized
    logger.warning("Using standard PDF ingestion (optimized not available)")

from src.ingest import ingest_website
from src.embedder import EmbeddingModel
from src.retriever import retrieve
from src.llm import stream_answer
from src.utils import build_prompt

# Page configuration
st.set_page_config(
    page_title="MuniRag - AI Assistant", 
    page_icon="🏛️", 
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/yourusername/munirag',
        'Report a bug': 'https://github.com/yourusername/munirag/issues',
        'About': 'MuniRAG - Multi-Model RAG for Municipalities'
    }
)

# Initialize session state
if "current_model" not in st.session_state:
    st.session_state.current_model = enhanced_settings.EMBEDDING_MODEL

if "history" not in st.session_state:
    st.session_state.history = []

if "show_model_warning" not in st.session_state:
    st.session_state.show_model_warning = False


# Sidebar with enhanced model selection
with st.sidebar:
    st.title("🔧 Settings")
    
    # Model selection with dimension info
    with st.expander("📊 Embedding Model", expanded=True):
        current_model = st.session_state.current_model
        available_models = enhanced_settings.get_available_models()
        
        # Format model names with dimensions
        model_options = {
            model: f"{model} ({enhanced_settings.get_embedding_dimension(model)}D)"
            for model in available_models
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=available_models.index(current_model) if current_model in available_models else 0,
            key="model_selector"
        )
        
        # Check for dimension mismatch
        if selected_model != current_model:
            current_dim = enhanced_settings.get_embedding_dimension(current_model)
            new_dim = enhanced_settings.get_embedding_dimension(selected_model)
            
            if current_dim != new_dim:
                st.warning(
                    f"⚠️ Dimension mismatch!\n"
                    f"Current: {current_dim}D → New: {new_dim}D\n"
                    f"Documents will be stored separately."
                )
                
                if st.button("View Migration Options", type="secondary"):
                    st.session_state.show_migration_center = True
            
            if st.button("Apply Model Change", type="primary"):
                st.session_state.current_model = selected_model
                enhanced_settings.EMBEDDING_MODEL = selected_model
                st.success(f"Switched to {selected_model}")
                st.rerun()
    
    st.divider()
    
    # Document ingestion
    st.title("📂 Add Content")
    
    # Show current collection stats
    with st.expander("📈 Collection Stats"):
        try:
            vector_store = MultiModelVectorStore(st.session_state.current_model)
            stats = vector_store.get_collection_stats()
            
            current_collection = vector_store.current_collection
            if current_collection in stats:
                info = stats[current_collection]
                st.metric("Documents", f"{info['points_count']:,}")
                st.caption(f"Collection: {current_collection}")
            else:
                st.info("No documents yet")
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    # PDF upload
    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if pdfs:
        if st.button("Ingest PDFs"):
            progress_bar = st.progress(0, text="Starting PDF ingestion...")
            start_time = datetime.now()
            
            try:
                # Use the enhanced vector store
                with st.spinner("Processing PDFs..."):
                    ingest_pdfs_optimized(pdfs, progress_bar=progress_bar)
                    
                elapsed = (datetime.now() - start_time).total_seconds()
                st.success(f"✅ Ingested {len(pdfs)} PDF(s) in {int(elapsed//60)}m {int(elapsed%60)}s")
                
                # Clear cache to reflect new documents
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                    
            except Exception as e:
                st.error(f"❌ Error during ingestion: {str(e)}")
            finally:
                progress_bar.empty()
    
    # Website crawl
    url = st.text_input("Crawl a website (depth 1)")
    if st.button("Ingest website") and url:
        with st.spinner("Crawling website..."):
            try:
                ingest_website(url)
                st.success("Website crawled and indexed")
            except Exception as e:
                st.error(f"Error crawling website: {str(e)}")


# Main content area
st.markdown("# 🏛️ MuniRag - Multi-Model Edition")
st.markdown("Ask questions about your municipal documents. Now with support for multiple embedding models!")

# Show migration center if requested
if st.session_state.get("show_migration_center", False):
    with st.container():
        st.divider()
        st.subheader("🔄 Migration Center")
        
        migration_manager = ModelMigrationManager()
        
        tab1, tab2 = st.tabs(["Migrate Data", "History"])
        
        with tab1:
            migration_manager.render_migration_ui()
        
        with tab2:
            migration_manager.render_migration_history()
        
        if st.button("Close Migration Center"):
            st.session_state.show_migration_center = False
            st.rerun()
        
        st.divider()

# Initialize embedding model with caching
@st.cache_resource
def get_embedder(model_name: str):
    """Get embedder for specific model with caching"""
    return EmbeddingModel(model_name)

# Chat interface
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# User input
user_q = st.chat_input("Type your question…")

if user_q:
    st.session_state.history.append(("user", user_q))
    st.chat_message("user").markdown(user_q)
    
    try:
        # Get embedder for current model
        embedder = get_embedder(st.session_state.current_model)
        
        # Embed query
        q_emb = embedder.embed_query(user_q)
        
        # Retrieve with model-aware vector store
        if enhanced_settings.ENABLE_MULTI_MODEL:
            # Use enhanced vector store
            vector_store = MultiModelVectorStore(st.session_state.current_model)
            
            # Perform search based on settings
            if enhanced_settings.ENABLE_CROSS_MODEL_SEARCH:
                # Search across all models with same dimension
                search_results = vector_store.search(
                    query_embedding=q_emb,
                    top_k=enhanced_settings.TOP_K,
                    search_all_models=True
                )
            else:
                # Search only in current model's collection
                search_results = vector_store.search(
                    query_embedding=q_emb,
                    top_k=enhanced_settings.TOP_K
                )
            
            # Convert to legacy format for compatibility
            context_docs = [
                (result["content"], result["metadata"]) 
                for result in search_results
            ]
        else:
            # Use legacy retriever
            context_docs = retrieve(q_emb)
        
        # Build prompt
        prompt = build_prompt(context_docs, user_q)
        
        # Display assistant answer incrementally
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            collected = ""
            
            for delta in stream_answer(prompt):
                collected += delta
                answer_placeholder.markdown(collected + "▌")
            
            answer_placeholder.markdown(collected)
            
            # Show sources with model information
            with st.expander("Sources"):
                for i, doc in enumerate(context_docs[:3], 1):
                    if isinstance(doc, tuple):
                        content, metadata = doc
                    else:
                        content = doc.get("content", "")
                        metadata = doc.get("metadata", {})
                    
                    label = metadata.get("source", "unknown")
                    model_used = metadata.get("embedding_model", st.session_state.current_model)
                    
                    st.markdown(f"**[{label}]** (Model: {model_used})")
                    st.markdown(f"{content[:500]}…")
        
        st.session_state.history.append(("assistant", collected))
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        logger.error(f"Query processing error: {str(e)}", exc_info=True)

# Footer with model info
with st.container():
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"Current Model: {st.session_state.current_model}")
    
    with col2:
        dim = enhanced_settings.get_embedding_dimension(st.session_state.current_model)
        st.caption(f"Dimensions: {dim}")
    
    with col3:
        if st.button("🔧 Model Management", key="footer_model_mgmt"):
            st.session_state.show_migration_center = True
            st.rerun()