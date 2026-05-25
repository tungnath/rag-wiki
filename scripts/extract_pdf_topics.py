"""Check SQL PDF content in detail."""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import fitz

PDF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documents")

filepath = os.path.join(PDF_FOLDER, "SQL_Window_Functions_1_1.pdf")
doc = fitz.open(filepath)
print(f"Pages: {len(doc)}")
for i, page in enumerate(doc):
    text = page.get_text()
    print(f"Page {i}: text len={len(text)}, images={len(page.get_images())}")
    if text.strip():
        print(f"  Content: {text.strip()[:500]}")
    blocks = page.get_text("blocks")
    print(f"  Text blocks: {len(blocks)}")
    for j, block in enumerate(blocks[:5]):
        print(f"    Block {j}: {str(block)[:200]}")
doc.close()

# Also check with different extraction
print("\n--- Trying text extraction with different method ---")
doc2 = fitz.open(filepath)
for i, page in enumerate(doc2):
    text = page.get_text("text")
    print(f"Page {i} text method: {len(text)} chars")
    text2 = page.get_text("html")
    print(f"Page {i} html method: {len(text2)} chars")
    # Check if it's an image-heavy page
    images = page.get_images()
    print(f"Page {i} images: {len(images)}")
doc2.close()
