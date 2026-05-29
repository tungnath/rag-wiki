"""
RAG Wiki — Streamlit Chat Interface.

Thin UI layer that delegates to rag_engine/ for all RAG logic.
"""

import streamlit as st
import logging
import os
import shutil

from rag_engine.config import DOCUMENTS_FOLDER, CHROMA_PATH, BM25_CACHE_PATH, LLM_MODEL, SUPPORTED_EXTENSIONS
from rag_engine.ingest import load_and_chunk_documents, build_vectorstore, load_vectorstore, load_bm25_cache
from rag_engine.retriever import hybrid_retrieve
from rag_engine.llm import generate_answer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="RAG Wiki", page_icon="📚", layout="wide")
st.title("📚 RAG Wiki")
st.caption("100% open-source • Runs on the laptop • CPU only • Hybrid BM25 + Dense retrieval")

# ===================== ENGINE INIT =====================

@st.cache_resource
def init_engine():
    """Initialize the RAG engine: load or build vectorstore + BM25 index."""
    # Try loading existing vectorstore
    vectorstore = load_vectorstore()
    bm25_cache = load_bm25_cache()

    if vectorstore is not None and bm25_cache is not None:
        return vectorstore, bm25_cache

    # Need to build from scratch
    status = st.empty()
    def progress(msg):
        status.info(f"⏳ {msg}")

    try:
        chunks = load_and_chunk_documents(progress_callback=progress)
        vectorstore = build_vectorstore(chunks, progress_callback=progress)
        bm25_cache = load_bm25_cache()
        status.success(f"✅ Ingested {len(chunks)} chunks. Ready to chat!")
        return vectorstore, bm25_cache
    except Exception as e:
        status.error(f"❌ Ingestion failed: {e}")
        st.stop()

vectorstore, bm25_cache = init_engine()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.subheader("⚙️ Settings")

    # Model selector
    llm_model = st.selectbox(
        "LLM Model",
        ["llama3.2:3b", "qwen2.5:7b"],
        index=0,
        help="3b is faster, 7b gives better answers"
    )

    st.divider()
    st.subheader("📁 Documents")
    # List all supported documents
    doc_files = [
        f for f in os.listdir(DOCUMENTS_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    for doc in sorted(doc_files):
        ext = os.path.splitext(doc)[1].lower()
        icon = {"pdf": "📕", ".txt": "📄", ".md": "📝", ".csv": "📊", ".docx": "📘"}.get(ext, "📄")
        st.caption(f"{icon} {doc}")
    st.caption(f"Total: {len(doc_files)} documents")
    st.caption(f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    st.divider()
    if st.button("🔄 Re-ingest All Documents", use_container_width=True):
        # Clear everything
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        if os.path.exists(BM25_CACHE_PATH):
            os.remove(BM25_CACHE_PATH)
        st.cache_resource.clear()
        st.rerun()

# ===================== CHAT =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            # Retrieve
            docs = hybrid_retrieve(
                query=prompt,
                vectorstore=vectorstore,
                bm25_cache=bm25_cache,
            )

            # Generate
            answer, sources = generate_answer(
                query=prompt,
                docs=docs,
                model=llm_model,
            )

            # Display answer
            st.markdown(answer)

            # Display sources
            if sources:
                st.divider()
                st.caption("**📄 Sources Used:**")
                # Deduplicate sources
                seen = set()
                for src in sources:
                    key = f"{src['filename']}:p{src['page']}"
                    if key not in seen:
                        seen.add(key)
                        st.caption(f"• {src['filename']} — Page {src['page']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})