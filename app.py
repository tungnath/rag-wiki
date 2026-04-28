import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import os

# ===================== CONFIG =====================
st.set_page_config(page_title="RAG Wiki", page_icon="📚")
st.title("📚 RAG Wiki")
st.caption("100% open-source • Runs on your laptop • CPU only")

PDF_FOLDER = "./documents"
CHROMA_PATH = "./chroma_db"
LLM_MODEL = "llama3.2:3b"          # change if you pulled another model
EMBED_MODEL = "nomic-embed-text"

# ===================== INGESTION =====================
@st.cache_resource
def get_vectorstore():
    if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model=EMBED_MODEL))

    st.info("First-time setup: Ingesting all PDFs... (this takes a few minutes)")
    docs = []

    for file in os.listdir(PDF_FOLDER):
        if not file.lower().endswith(".pdf"):
            continue
        loader = PyMuPDFLoader(os.path.join(PDF_FOLDER, file))
        loaded_docs = loader.load()

        # === KEY FIX #1: Add filename to every chunk ===
        for doc in loaded_docs:
            doc.page_content = f"Document: {file}\n\n{doc.page_content}"

        docs.extend(loaded_docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_PATH
    )
    st.success(f"✅ Ingested {len(chunks)} chunks from {len(os.listdir(PDF_FOLDER))} PDFs")
    return vectorstore

# ===================== UI =====================
vectorstore = get_vectorstore()
#2: Use MMR for diversity + more candidates ===
retriever = vectorstore.as_retriever(
    search_type="mmr",                    # pulls diverse results
    search_kwargs={"k": 8, "fetch_k": 20} # more candidates → better chance of both PDFs
)

llm = ChatOllama(model=LLM_MODEL, temperature=0)

# Sidebar controls
with st.sidebar:
    st.subheader("📁 Documents Folder")
    st.write(f"`{PDF_FOLDER}`")
    if st.button("🔄 Re-ingest All PDFs"):
        if os.path.exists(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)
        st.cache_resource.clear()
        st.rerun()

# Main chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get relevant chunks
            docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Prompt with sources
            # system_prompt = """Answer ONLY using the provided context.
            # Be concise and accurate. At the end list sources as [Document: filename - Page X]."""
            #3: Stronger prompt for ambiguity ===
            system_prompt = """Answer ONLY using the provided context.
            If the term 'MCP' (or any acronym) has different meanings in different documents,
            clearly explain each meaning and cite the exact document it comes from.
            Always list every relevant source at the end."""
            
            messages = [
                ("system", system_prompt),
                ("human", f"Context:\n{context}\n\nQuestion: {prompt}")
            ]
            
            response = llm.invoke(messages)
            answer = response.content
            
            st.markdown(answer)
            
            # Show sources
            st.caption("Sources:")
            for doc in docs:
                source = doc.metadata.get("source", "unknown").split("/")[-1]
                page = doc.metadata.get("page", "?") + 1
                st.caption(f"• {source} (Page {page})")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})