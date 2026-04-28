#!/usr/bin/env python
"""Debug PDF content loading"""

from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

pdf_files = list(Path("../documents").glob("*.pdf"))

for pdf_file in pdf_files[:2]:  # Check first 2 PDFs
    print(f"\n{'='*60}")
    print(f"File: {pdf_file.name}")
    print(f"{'='*60}")

    try:
        loader = PyMuPDFLoader(str(pdf_file))
        docs = loader.load()

        print(f"Pages loaded: {len(docs)}")

        for i, doc in enumerate(docs[:3]):  # Show first 3 pages
            content_len = len(doc.page_content)
            print(f"\nPage {i+1}: {content_len} chars")
            if content_len > 0:
                print(f"Preview: {doc.page_content[:200]}...")
            else:
                print("WARNING: Empty page content!")
                print(f"Metadata: {doc.metadata}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

