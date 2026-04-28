import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
import os

# ===================== CONFIG =====================
st.set_page_config(page_title="RAG Wiki", page_icon="📚")
st.title("📚 RAG Wiki")
st.caption("100% open-source • Runs on the laptop • CPU only")

PDF_FOLDER = "./documents"
CHROMA_PATH = "./chroma_db"
LLM_MODEL = "llama3.2:3b"          # change if you pulled another model
EMBED_MODEL = "mxbai-embed-large"  # 1024 dims vs 384 (nomic), better for acronym disambiguation

# ===================== INGESTION =====================

def extract_context_from_pages(pages, page_index):
    """Extract surrounding context for a page to disambiguate acronyms."""
    context_window = 2  # Look 2 pages before and after
    context_parts = []

    start = max(0, page_index - context_window)
    end = min(len(pages), page_index + context_window + 1)

    for i in range(start, end):
        if i != page_index and i < len(pages):
            # Extract first 100 chars of surrounding pages for context
            text = pages[i].page_content[:100] if pages[i].page_content else ""
            if text:
                context_parts.append(text.replace('\n', ' ')[:100])

    return " | ".join(context_parts) if context_parts else ""

@st.cache_resource
def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://localhost:11434")

    if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
        try:
            return Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings,
                collection_name="rag_wiki"
            )
        except Exception as e:
            st.warning(f"Could not load existing database: {e}. Re-ingesting...")
            import shutil
            shutil.rmtree(CHROMA_PATH)

    st.info("First-time setup: Ingesting all PDFs... (this takes a few minutes)")
    docs = []

    for file in os.listdir(PDF_FOLDER):
        if not file.lower().endswith(".pdf"):
            continue
        try:
            pdf_path = os.path.join(PDF_FOLDER, file)
            # Use absolute path to ensure PyMuPDFLoader works correctly
            pdf_path = os.path.abspath(pdf_path)
            loader = PyMuPDFLoader(pdf_path)
            loaded_docs = loader.load()

            # Phase 2: Enhanced chunk enrichment with metadata and context
            for i, doc in enumerate(loaded_docs):
                # Skip empty pages
                if len(doc.page_content.strip()) == 0:
                    continue

                doc.metadata['filename'] = file
                doc.metadata['source_page_index'] = i
                # Page number already in metadata as 'page'

            docs.extend([d for d in loaded_docs if len(d.page_content.strip()) > 0])
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
            continue

    if not docs:
        st.error("No documents loaded!")
        return None

    # Split into chunks - REDUCED size for mxbai-embed-large compatibility
    # mxbai supports ~512 tokens, approximately 2000 chars but we use 300 to be safe
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # Reduced from 800 to ensure token compatibility
        chunk_overlap=50,    # Reduced overlap to match smaller chunks
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        st.error("No chunks created from documents!")
        return None

    # Phase 2 continued: Add global chunk sequence and context metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['global_chunk_id'] = i
        # Extract section context (first 50 chars of chunk as mini-context)
        chunk.metadata['chunk_context'] = chunk.page_content[:50].replace('\n', ' ')

    # Create vector store with mxbai-embed-large
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="rag_wiki"
        )
        st.success(f"✅ Ingested {len(chunks)} chunks from {len(os.listdir(PDF_FOLDER))} PDFs")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise

# ===================== UI =====================
vectorstore = get_vectorstore()

# Two-stage retrieval with enhanced reranking
def retrieve_and_rerank(query, k=8, fetch_k=25):
    """
    Phase 3: Two-stage retrieval with reranking
    Stage 1: Dense retrieval using mxbai-embed-large (better semantic understanding)
    Stage 2: Reranking by semantic similarity, keyword overlap, and document diversity
    """
    # Stage 1: Dense retrieval
    candidates = vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k)

    # Stage 2: Reranking with multiple scoring factors
    reranked = []
    doc_counts = {}
    query_terms = set(query.lower().split())

    for doc, semantic_score in candidates:
        filename = doc.metadata.get('filename', 'unknown')

        # Track document frequency
        doc_counts[filename] = doc_counts.get(filename, 0) + 1

        # Diversity penalty: penalize multiple chunks from same document
        # This ensures we get chunks from different documents for acronym disambiguation
        diversity_penalty = (doc_counts[filename] - 1) * 0.15

        # Keyword overlap: how many query terms appear in the chunk
        chunk_terms = set(doc.page_content.lower().split())
        keyword_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1) if query_terms else 0

        # Exact phrase matching bonus
        exact_match_bonus = 0.1 if any(phrase in doc.page_content.lower() for phrase in query.lower().split()) else 0

        # Calculate final score
        final_score = semantic_score + (keyword_overlap * 0.2) + exact_match_bonus - diversity_penalty

        reranked.append((doc, final_score))

    # Sort by final score descending
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:k]]

retriever_func = lambda query: retrieve_and_rerank(query)

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
            docs = retrieve_and_rerank(prompt)

            # Phase 4: Intelligent context assembly - Group by document
            context_by_doc = {}
            doc_relevance_order = []  # Track order for source display

            for doc in docs:
                filename = doc.metadata.get('filename', 'unknown')
                if filename not in context_by_doc:
                    context_by_doc[filename] = []
                    doc_relevance_order.append(filename)
                context_by_doc[filename].append(doc)

            # Build context with clear document boundaries and page numbers
            context_parts = []
            for filename in doc_relevance_order:
                doc_list = context_by_doc[filename]
                context_parts.append(f"\n## Source Document: {filename}")
                context_parts.append("-" * 60)

                for idx, doc in enumerate(doc_list, 1):
                    page = doc.metadata.get('page', 0) + 1
                    context_parts.append(f"\n[Page {page}]")
                    context_parts.append(doc.page_content.strip())

                context_parts.append("")  # Blank line between documents

            context = "\n".join(context_parts)

            # Phase 4: Enhanced system prompt - Generic, handles any ambiguity
            system_prompt = """You are a helpful assistant answering questions based ONLY on the provided context from multiple documents.

IMPORTANT GUIDELINES:
1. Answer ONLY using information from the provided context
2. If a term (especially acronyms or technical terms) has multiple meanings across different documents, clearly explain ALL meanings found
3. Always cite the specific document source for each meaning (e.g., "In [Document Name]...")
4. When different documents define the same term differently, present both/all definitions clearly
5. List all source documents used at the end with exact page numbers
6. Do NOT make up or assume information not in the context
7. If the context doesn't contain information to answer the question, say so explicitly

Format for multiple meanings:
- Term/Acronym: [TERM]
  - Meaning 1: [From Document A, Page X]: [Definition]
  - Meaning 2: [From Document B, Page Y]: [Definition]
  - etc.

Always be thorough when dealing with acronyms or potentially ambiguous terms."""

            messages = [
                ("system", system_prompt),
                ("human", f"Context from documents:\n{context}\n\nQuestion: {prompt}")
            ]
            
            response = llm.invoke(messages)
            answer = response.content
            
            st.markdown(answer)
            
            # Show sources in order of relevance
            st.divider()
            st.caption("**📄 Sources Used:**")
            for doc in docs:
                source = doc.metadata.get("filename", "unknown")
                page = doc.metadata.get("page", "?")
                if isinstance(page, int):
                    page = page + 1
                else:
                    page = int(page) + 1 if isinstance(page, str) and page.isdigit() else "?"
                st.caption(f"• {source} — Page {page}")

    st.session_state.messages.append({"role": "assistant", "content": answer})