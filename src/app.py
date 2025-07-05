import streamlit as st
from ingest import ingest_pdfs, ingest_website
from embedder import embed
from retriever import retrieve
from llm import stream_answer
from utils import build_prompt

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
            with st.spinner("Ingesting PDFs... Please wait!"):
                ingest_pdfs(pdfs)
            st.success(f"Ingested {len(pdfs)} PDF(s)")
#            st.success("Ingestion completed successfully!")

    url = st.text_input("Crawl a website (depth 1)")
    if st.button("Ingest website") and url:
        ingest_website(url)
        st.success("Website crawled and indexed")

#st.markdown("# 🏛️ Municipal AI Chat\nAsk anything about your docs or site")
st.markdown("# 🏛️ MuniRag\nAsk anything about your municipal documents or website")

if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.chat_input("Type your question…")

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if user_q:
    st.session_state.history.append(("user", user_q))
    st.chat_message("user").markdown(user_q)

    # retrieval
    q_emb = embed([user_q])[0]
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
