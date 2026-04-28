#!/usr/bin/env python
"""
Diagnostic script to test embedding model and ingestion pipeline
Run this BEFORE starting the Streamlit app to verify everything works
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("RAG Wiki Diagnostic Script")
print("=" * 70)

# Test 1: Check if Ollama is running
print("\n[1/5] Checking Ollama connection...")
try:
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    test_embed = embeddings.embed_query("hello world")
    print(f"✓ Ollama connection successful")
    print(f"✓ mxbai-embed-large embedding dimension: {len(test_embed)}")
except Exception as e:
    print(f"✗ Ollama error: {e}")
    print("  → Make sure Ollama is running: ollama serve")
    print("  → Pull model: ollama pull mxbai-embed-large")
    sys.exit(1)

# Test 2: Check PDF folder
print("\n[2/5] Checking PDF files...")
pdf_dir = Path("../documents")
if not pdf_dir.exists():
    print(f"✗ Documents folder not found!")
    sys.exit(1)

pdf_files = list(pdf_dir.glob("*.pdf"))
if not pdf_files:
    print(f"✗ No PDF files found in {pdf_dir}")
    sys.exit(1)

print(f"✓ Found {len(pdf_files)} PDF files:")
for pdf in pdf_files:
    print(f"  - {pdf.name}")

# Test 3: Test PDF loading
print("\n[3/5] Testing PDF loading...")
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    test_pdf = pdf_files[0]
    # Use absolute path
    pdf_abs_path = str(test_pdf.resolve())
    print(f"  Loading from: {pdf_abs_path}")
    loader = PyMuPDFLoader(pdf_abs_path)
    docs = loader.load()
    print(f"✓ Successfully loaded {test_pdf.name}")
    print(f"  → Total pages: {len(docs)}")

    # Filter non-empty pages
    non_empty = [d for d in docs if len(d.page_content.strip()) > 0]
    print(f"  → Non-empty pages: {len(non_empty)}")

    if non_empty:
        print(f"  → First non-empty page: {len(non_empty[0].page_content)} chars")
        docs = non_empty
    else:
        print("  ✗ ERROR: No non-empty pages found!")
        sys.exit(1)

except Exception as e:
    print(f"✗ PDF loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test text splitting
print("\n[4/5] Testing text splitting...")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    # Filter out empty pages
    non_empty_docs = [doc for doc in docs if len(doc.page_content) > 0]
    if not non_empty_docs:
        print(f"⚠ Warning: All pages are empty!")
        print("  Available docs:", len(docs))
        for i, doc in enumerate(docs):
            print(f"  Page {i}: {len(doc.page_content)} chars, metadata: {doc.metadata}")
        sys.exit(1)

    chunks = splitter.split_documents(non_empty_docs[:2])  # Test with first 2 non-empty pages
    print(f"✓ Text splitting successful")
    print(f"  → Non-empty pages: {len(non_empty_docs)} / {len(docs)}")
    print(f"  → Chunks created: {len(chunks)}")
    if chunks:
        print(f"  → Chunk size range: {min(len(c.page_content) for c in chunks)} - {max(len(c.page_content) for c in chunks)} chars")
except Exception as e:
    print(f"✗ Text splitting error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test Chroma database creation
print("\n[5/5] Testing Chroma database...")
try:
    from langchain_chroma import Chroma
    import shutil

    # Clean up any existing test database
    test_db_path = "../test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)

    # Try creating with test chunks
    vectorstore = Chroma.from_documents(
        documents=chunks[:5],  # Use just 5 chunks for testing
        embedding=embeddings,
        persist_directory=test_db_path,
        collection_name="test"
    )
    print(f"✓ Chroma database creation successful")
    print(f"  → Created in: {test_db_path}")

    # Clean up
    shutil.rmtree(test_db_path)
    print(f"✓ Test database cleaned up")

except Exception as e:
    print(f"✗ Chroma error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All diagnostic tests passed!")
print("=" * 70)
print("\nNext steps:")
print("1. Delete chroma_db if it exists: rmdir /s /q chroma_db")
print("2. Run the Streamlit app: streamlit run app.py")
print("3. Wait for ingestion to complete (first run takes 5-10 minutes)")
print("=" * 70)

