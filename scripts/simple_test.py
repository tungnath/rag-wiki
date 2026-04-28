#!/usr/bin/env python
"""Simple test to debug PDF loading"""

import sys
import os

pdf_path = "../documents/Intro_to_MCP.pdf"
print(f"Testing PDF: {pdf_path}")
print(f"Absolute path: {os.path.abspath(pdf_path)}")
print(f"File exists: {os.path.exists(pdf_path)}")
print(f"File size: {os.path.getsize(pdf_path)} bytes")

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    print("\n✓ PyMuPDFLoader imported successfully")

    loader = PyMuPDFLoader(pdf_path)
    print(f"✓ Loader created: {loader}")

    docs = loader.load()
    print(f"✓ Loaded {len(docs)} pages")

    for i in range(min(5, len(docs))):
        content = docs[i].page_content
        content_len = len(content)
        stripped_len = len(content.strip())
        print(f"\nPage {i}:")
        print(f"  Raw length: {content_len}")
        print(f"  Stripped length: {stripped_len}")
        print(f"  First 100 chars: {repr(content[:100])}")
        print(f"  Metadata: {docs[i].metadata}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

