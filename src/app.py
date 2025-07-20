import streamlit as st
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Use parallel ingest for 10-50x faster PDF processing
try:
    from src.ingest import ingest_pdfs_parallel
    logger.info("Using parallel PDF ingestion with PyMuPDF (10-50x faster)")
except ImportError:
    # Fallback to optimized ingest if parallel not available
    try:
        from src.deprecated.ingest_optimized import ingest_pdfs_optimized as ingest_pdfs_parallel
        logger.warning("Using optimized PDF ingestion (parallel not available)")
    except ImportError:
        from src.deprecated.ingest_backup import ingest_pdfs as ingest_pdfs_parallel
        logger.warning("Using standard PDF ingestion (no optimizations available)")

from src.deprecated.ingest_backup import ingest_website
from src.embedder import EmbeddingModel
from src.retriever import retrieve
from src.llm import stream_answer
from src.utils import build_prompt

st.set_page_config(page_title="MuniRag - AI Assistant", page_icon="🏛️", layout="wide")

with st.sidebar:
    st.title("📂 Add content")
    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
#    if pdfs:
#        if st.button("Ingest PDFs"):
#            ingest_pdfs(pdfs)
#            st.success(f"Ingested {len(pdfs)} PDF(s)")
    if pdfs:
        if st.button("Ingest PDFs"):
            progress_bar = st.progress(0, text="Starting PDF ingestion...")
            start_time = datetime.now()
            
            try:
                ingest_pdfs_parallel(pdfs, progress_bar=progress_bar)
                elapsed = (datetime.now() - start_time).total_seconds()
                st.success(f"✅ Ingested {len(pdfs)} PDF(s) in {int(elapsed//60)}m {int(elapsed%60)}s")
            except Exception as e:
                st.error(f"❌ Error during ingestion: {str(e)}")
            finally:
                progress_bar.empty()
#            st.success("Ingestion completed successfully!")

    url = st.text_input("Crawl a website (depth 1)")
    if st.button("Ingest website") and url:
        ingest_website(url)
        st.success("Website crawled and indexed")

#st.markdown("# 🏛️ Municipal AI Chat\nAsk anything about your docs or site")
st.markdown("# 🏛️ MuniRag\nAsk anything about your municipal documents or website")

if "history" not in st.session_state:
    st.session_state.history = []

# Initialize embedding model
@st.cache_resource
def get_embedder():
    return EmbeddingModel()

embedder = get_embedder()

user_q = st.chat_input("Type your question…")

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if user_q:
    st.session_state.history.append(("user", user_q))
    st.chat_message("user").markdown(user_q)

    # retrieval
    q_emb = embedder.embed_query(user_q)
    context_docs = retrieve(q_emb)

    # build prompt
    prompt = build_prompt(context_docs, user_q)

    # display assistant answer incrementally
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        collected = ""
        for delta in stream_answer(prompt):
            collected += delta
            answer_placeholder.markdown(collected + "▌")
        answer_placeholder.markdown(collected)

        # show sources
        with st.expander("Sources"):
            for i, (doc, meta) in enumerate(context_docs, 1):
                label = meta.get("source", "unknown")
                st.markdown(f"**[{label}]** {doc[:500]}…")
    st.session_state.history.append(("assistant", collected))
