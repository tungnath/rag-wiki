"""
LLM interaction: context assembly and prompt construction.

Groups retrieved chunks by source document with clear boundaries,
then sends to Ollama with a system prompt that handles ambiguity
without hardcoding specific terms.
"""

import logging
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import OLLAMA_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided document context.

RULES:
1. Base your answer ONLY on the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information to answer, say so explicitly.
3. When the same term, acronym, or concept appears in MULTIPLE documents with different meanings, present ALL meanings found, clearly labeling each with its source document and page number.
4. Always cite your sources using the format: [Document Name, Page X].
5. If only one document is relevant, cite only that document.
6. Do NOT invent or hallucinate information or sources.
7. Be concise but thorough.

FORMAT FOR AMBIGUOUS TERMS:
When a term has multiple meanings across documents:
- **Meaning 1** (from [Document A, Page X]): [explanation]
- **Meaning 2** (from [Document B, Page Y]): [explanation]

At the end of your answer, list all source documents used."""


def assemble_context(docs: list[Document]) -> tuple[str, list[dict]]:
    """Group retrieved chunks by source document with clear boundaries.
    
    Returns:
        Tuple of (context_string, sources_list).
        sources_list contains dicts with 'filename' and 'page' keys.
    """
    # Group by document
    doc_groups: dict[str, list[Document]] = {}
    doc_order: list[str] = []

    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        if filename not in doc_groups:
            doc_groups[filename] = []
            doc_order.append(filename)
        doc_groups[filename].append(doc)

    # Build context with clear document boundaries
    context_parts = []
    sources = []

    for filename in doc_order:
        chunks = doc_groups[filename]
        doc_title = chunks[0].metadata.get("doc_title", filename)

        context_parts.append(f"\n═══ Document: {filename} (Topic: {doc_title}) ═══")

        for chunk in chunks:
            page = chunk.metadata.get("page", 0)
            page_display = page + 1 if isinstance(page, int) else page

            # Use original content (without [Source: ...] prefix) for LLM context
            content = chunk.metadata.get("original_content", chunk.page_content)

            context_parts.append(f"\n[Page {page_display}]")
            context_parts.append(content.strip())

            sources.append({"filename": filename, "page": page_display})

        context_parts.append("")  # Blank line between documents

    return "\n".join(context_parts), sources


def generate_answer(
    query: str,
    docs: list[Document],
    model: str = LLM_MODEL,
    temperature: float = 0,
) -> tuple[str, list[dict]]:
    """Generate an answer using the LLM with retrieved context.
    
    Args:
        query: User's question.
        docs: Retrieved document chunks.
        model: Ollama model name.
        temperature: LLM temperature (0 = deterministic).
    
    Returns:
        Tuple of (answer_string, sources_list).
    """
    if not docs:
        return "I couldn't find any relevant information in the documents to answer your question.", []

    context, sources = assemble_context(docs)

    llm = ChatOllama(model=model, temperature=temperature, base_url=OLLAMA_BASE_URL)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context from documents:\n{context}\n\nQuestion: {query}"),
    ]

    try:
        response = llm.invoke(messages)
        return response.content, sources
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"Error generating answer: {e}", sources
