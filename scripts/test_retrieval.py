#!/usr/bin/env python
"""
Comprehensive Retrieval Test Suite — 100 Test Cases.

Tests RETRIEVAL ONLY (no LLM needed) to validate that the correct
documents are returned for each query. This runs fast because it
skips LLM inference entirely.

Usage:
    python scripts/test_retrieval.py

The script:
1. Loads/builds the vectorstore and BM25 index
2. Runs 100 queries
3. Checks if the expected document(s) appear in the top results
4. Reports pass/fail with detailed diagnostics
"""

import sys
import os
import time
import logging

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.config import FINAL_K
from rag_engine.ingest import (
    load_and_chunk_pdfs, build_vectorstore,
    load_vectorstore, load_bm25_cache
)
from rag_engine.retriever import hybrid_retrieve

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ================================================================
# TEST CASE DEFINITIONS
# Format: (test_id, query, expected_files, must_be_top_n)
#   - expected_files: list of filenames that MUST appear in results
#   - must_be_top_n: the expected file must be in the top N results
#                    (None = anywhere in results is fine)
# ================================================================

MCP_INTRO = "Intro_to_MCP.pdf"
MCP_GUIDE = "mcp guide.pdf"
LINUX = "linux-commands.pdf"
SQL = "SQL_Window_Functions_1_1.pdf"

# Any MCP document is acceptable for general MCP queries
MCP_ANY = [MCP_INTRO, MCP_GUIDE]

TEST_CASES = [
    # ----------------------------------------------------------
    # Category 1: MCP -- Intro_to_MCP.pdf (25 tests)
    # ----------------------------------------------------------
    ("1.01", "What is MCP?", MCP_ANY, 3),
    ("1.02", "What is Model Context Protocol?", MCP_ANY, 3),
    ("1.03", "Explain Model Context Protocol", MCP_ANY, 3),
    ("1.04", "What does MCP stand for?", MCP_ANY, 3),
    ("1.05", "How is MCP like USB-C?", MCP_ANY, 3),
    ("1.06", "What is the USB-C analogy for MCP?", MCP_ANY, 3),
    ("1.07", "How do protocols work in web communication?", [MCP_INTRO], 4),
    ("1.08", "What is the request response cycle?", [MCP_INTRO], 4),
    ("1.09", "Explain REST API as a common language", [MCP_INTRO], 4),
    ("1.10", "What are generative AI models?", [MCP_INTRO], 4),
    ("1.11", "What are the downsides of generative AI?", [MCP_INTRO], 3),
    ("1.12", "What is hallucination in AI models?", [MCP_INTRO], 4),
    ("1.13", "What are multi model agents?", [MCP_INTRO], 3),
    ("1.14", "How do LLMs use external tools?", MCP_ANY, 4),
    ("1.15", "What is an LLM with tools?", MCP_ANY, 4),
    ("1.16", "Give example of LLM using weather API", [MCP_INTRO], 4),
    ("1.17", "What is standardized communication in protocols?", [MCP_INTRO], 4),
    ("1.18", "What is interoperability in protocols?", [MCP_INTRO], 4),
    ("1.19", "Explain the concept of endpoints in web protocols", [MCP_INTRO], 4),
    ("1.20", "What are the types of HTTP requests?", [MCP_INTRO], 4),
    ("1.21", "How does a client send a request to a server?", MCP_ANY, 4),
    ("1.22", "What is GET and POST request?", [MCP_INTRO], 4),
    ("1.23", "What problems does generative AI have?", [MCP_INTRO], 4),
    ("1.24", "Why do LLMs hallucinate?", [MCP_INTRO], 4),
    ("1.25", "What is the role of a protocol in enabling services?", [MCP_INTRO], 4),

    # ----------------------------------------------------------
    # Category 2: MCP Guide -- mcp guide.pdf (25 tests)
    # ----------------------------------------------------------
    ("2.01", "What is the MCP architecture?", [MCP_GUIDE], 3),
    ("2.02", "Explain MCP host client server architecture", [MCP_GUIDE], 3),
    ("2.03", "What is the MCP Host?", [MCP_GUIDE], 3),
    ("2.04", "What is the MCP Client?", [MCP_GUIDE], 3),
    ("2.05", "What is the MCP Server?", [MCP_GUIDE], 3),
    ("2.06", "What are MCP tools resources and prompts?", [MCP_GUIDE], 3),
    ("2.07", "Why was MCP created?", [MCP_GUIDE], 3),
    ("2.08", "What is the M times N integration problem?", [MCP_GUIDE], 3),
    ("2.09", "How does MCP solve the integration problem?", [MCP_GUIDE], 3),
    ("2.10", "What is a 100 percent local MCP client?", [MCP_GUIDE], 3),
    ("2.11", "What is MCP powered agentic RAG?", [MCP_GUIDE], 3),
    ("2.12", "Explain MCP powered financial analyst project", [MCP_GUIDE], 3),
    ("2.13", "What is MCP powered voice agent?", [MCP_GUIDE], 3),
    ("2.14", "What is a unified MCP server?", [MCP_GUIDE], 3),
    ("2.15", "How does MCP work with Claude Desktop?", [MCP_GUIDE], 3),
    ("2.16", "What is MCP powered RAG over complex documents?", [MCP_GUIDE], 3),
    ("2.17", "What is MCP synthetic data generator?", [MCP_GUIDE], 3),
    ("2.18", "What is MCP deep researcher?", [MCP_GUIDE], 3),
    ("2.19", "How does MCP RAG over videos work?", [MCP_GUIDE], 3),
    ("2.20", "What is MCP audio analysis toolkit?", [MCP_GUIDE], 3),
    ("2.21", "What is the difference between MCP client and server?", [MCP_GUIDE], 3),
    ("2.22", "Who are the authors of the MCP guidebook?", [MCP_GUIDE], 3),
    ("2.23", "What is Daily Dose of Data Science?", [MCP_GUIDE], 3),
    ("2.24", "What are the MCP projects listed in the guide?", [MCP_GUIDE], 3),
    ("2.25", "How does MCP work with Cursor IDE?", [MCP_GUIDE], 3),

    # ----------------------------------------------------------
    # Category 3: Linux Commands -- linux-commands.pdf (25 tests)
    # ----------------------------------------------------------
    ("3.01", "What are Linux commands?", [LINUX], 3),
    ("3.02", "How to list files in Linux?", [LINUX], 3),
    ("3.03", "What is the ls command?", [LINUX], 3),
    ("3.04", "How to create an alias in Linux?", [LINUX], 3),
    ("3.05", "What does the alias command do?", [LINUX], 3),
    ("3.06", "What is the unalias command?", [LINUX], 3),
    ("3.07", "What is the pwd command in Linux?", [LINUX], 3),
    ("3.08", "How to change directory in Linux?", [LINUX], 3),
    ("3.09", "What does the cd command do?", [LINUX], 3),
    ("3.10", "How to copy files in Linux?", [LINUX], 3),
    ("3.11", "What is the cp command?", [LINUX], 3),
    ("3.12", "How to delete files in Linux?", [LINUX], 3),
    ("3.13", "What is the rm command?", [LINUX], 3),
    ("3.14", "How to move or rename files in Linux?", [LINUX], 3),
    ("3.15", "What is the mv command?", [LINUX], 3),
    ("3.16", "How to create a directory in Linux?", [LINUX], 3),
    ("3.17", "What is the mkdir command?", [LINUX], 3),
    ("3.18", "How to view manual pages in Linux?", [LINUX], 3),
    ("3.19", "What is the man command?", [LINUX], 3),
    ("3.20", "What does the touch command do?", [LINUX], 3),
    ("3.21", "How to change file permissions in Linux?", [LINUX], 3),
    ("3.22", "What is the chmod command?", [LINUX], 3),
    ("3.23", "What does sudo do?", [LINUX], 3),
    ("3.24", "How to shutdown a Linux system?", [LINUX], 3),
    ("3.25", "What is the htop command?", [LINUX], 3),

    # ----------------------------------------------------------
    # Category 4: SQL Window Functions -- SQL_Window_Functions_1_1.pdf (10 tests)
    # NOTE: This is an image-based PDF. If OCR fails, these tests
    #       will be skipped gracefully rather than counted as failures.
    # ----------------------------------------------------------
    ("4.01", "What are SQL window functions?", [SQL], 4),
    ("4.02", "Explain SQL window functions", [SQL], 4),
    ("4.03", "What is ROW_NUMBER in SQL?", [SQL], 4),
    ("4.04", "What is RANK in SQL?", [SQL], 4),
    ("4.05", "What is DENSE_RANK in SQL?", [SQL], 4),
    ("4.06", "What is PARTITION BY in SQL?", [SQL], 4),
    ("4.07", "What is ORDER BY in window functions?", [SQL], 4),
    ("4.08", "What is LEAD and LAG in SQL?", [SQL], 4),
    ("4.09", "What is NTILE in SQL?", [SQL], 4),
    ("4.10", "What are aggregate window functions?", [SQL], 4),

    # ----------------------------------------------------------
    # Category 5: Cross-document / Ambiguity (10 tests)
    # ----------------------------------------------------------
    ("5.01", "What is MCP and how does it work?", MCP_ANY, 3),
    ("5.02", "Compare MCP host and MCP client", [MCP_GUIDE], 4),
    ("5.03", "What is the MCP protocol architecture overview?", [MCP_GUIDE], 3),
    ("5.04", "How do AI models connect to external tools?", MCP_ANY, 4),
    ("5.05", "What is the translator analogy for MCP?", [MCP_GUIDE], 4),
    ("5.06", "How does MCP standardize AI tool connections?", MCP_ANY, 3),
    ("5.07", "What projects can be built with MCP?", [MCP_GUIDE], 3),
    ("5.08", "What is the difference between REST API and MCP?", MCP_ANY, 4),
    ("5.09", "How to build a local MCP client?", [MCP_GUIDE], 3),
    ("5.10", "What are the limitations of LLMs without MCP?", MCP_ANY, 4),

    # ----------------------------------------------------------
    # Category 6: Edge cases / Robustness (5 tests)
    # ----------------------------------------------------------
    ("6.01", "What is the exit command?", [LINUX], 4),
    ("6.02", "How to run a Python script from terminal?", [LINUX], 4),
    ("6.03", "What is the ./ notation?", [LINUX], 4),
    ("6.04", "How to colorize ls output?", [LINUX], 4),
    ("6.05", "What is Chainlit?", [MCP_GUIDE], 4),
]

assert len(TEST_CASES) == 100, f"Expected 100 test cases, got {len(TEST_CASES)}"


def run_tests():
    """Run all 100 retrieval test cases."""
    print("=" * 80)
    print("RAG Wiki -- Retrieval Test Suite (100 tests)")
    print("=" * 80)

    # -- Initialize engine --
    print("\n[1/3] Loading vectorstore and BM25 index...")
    vectorstore = load_vectorstore()
    bm25_cache = load_bm25_cache()

    if vectorstore is None or bm25_cache is None:
        print("  -> Vectorstore or BM25 not found. Building from scratch...")
        chunks = load_and_chunk_pdfs(progress_callback=lambda m: print(f"  -> {m}"))
        vectorstore = build_vectorstore(chunks, progress_callback=lambda m: print(f"  -> {m}"))
        bm25_cache = load_bm25_cache()
        print(f"  -> Built vectorstore with {len(chunks)} chunks")

    # Check which documents were ingested
    collection = vectorstore._collection
    all_meta = collection.get(include=["metadatas"])
    ingested_files = set()
    for meta in all_meta["metadatas"]:
        ingested_files.add(meta.get("filename", "unknown"))
    print(f"  -> Ingested documents: {sorted(ingested_files)}")

    # -- Run tests --
    print(f"\n[2/3] Running {len(TEST_CASES)} test cases...\n")

    results = {"pass": 0, "fail": 0, "skip": 0}
    failures = []
    skipped = []

    total_retrieval_time = 0

    for test_id, query, expected_files, top_n in TEST_CASES:
        # Check if expected document was actually ingested
        expected_in_db = any(f in ingested_files for f in expected_files)
        if not expected_in_db:
            results["skip"] += 1
            skipped.append((test_id, query, expected_files, "Document not in database"))
            print(f"  SKIP  {test_id}: {query[:50]}... (doc not ingested)")
            continue

        # Run retrieval
        t_start = time.time()
        try:
            docs = hybrid_retrieve(
                query=query,
                vectorstore=vectorstore,
                bm25_cache=bm25_cache,
            )
        except Exception as e:
            results["fail"] += 1
            failures.append((test_id, query, expected_files, f"ERROR: {e}"))
            print(f"  FAIL  {test_id}: {query[:50]}... (ERROR: {e})")
            continue
        t_elapsed = time.time() - t_start
        total_retrieval_time += t_elapsed

        # Check results
        retrieved_files = [doc.metadata.get("filename", "unknown") for doc in docs]

        # Check if any expected file appears in top_n results
        top_n_files = retrieved_files[:top_n]
        found = any(f in top_n_files for f in expected_files)

        if found:
            results["pass"] += 1
            print(f"  PASS  {test_id}: {query[:50]:<52} [{t_elapsed:.2f}s] -> {top_n_files}")
        else:
            results["fail"] += 1
            failures.append((test_id, query, expected_files, f"Got: {retrieved_files}"))
            print(f"  FAIL  {test_id}: {query[:50]:<52} [{t_elapsed:.2f}s]")
            print(f"        Expected: {expected_files}")
            print(f"        Got top-{top_n}: {top_n_files}")
            print(f"        All results: {retrieved_files}")

    # -- Report --
    print(f"\n{'=' * 80}")
    print("[3/3] TEST RESULTS SUMMARY")
    print(f"{'=' * 80}")

    total = results["pass"] + results["fail"] + results["skip"]
    testable = results["pass"] + results["fail"]
    pass_rate = (results["pass"] / testable * 100) if testable > 0 else 0

    print(f"\n  Total tests:     {total}")
    print(f"  Passed:          {results['pass']}")
    print(f"  Failed:          {results['fail']}")
    print(f"  Skipped:         {results['skip']}")
    print(f"  Pass rate:       {pass_rate:.1f}% ({results['pass']}/{testable})")
    print(f"  Avg retrieval:   {total_retrieval_time / max(testable, 1):.2f}s")
    print(f"  Total time:      {total_retrieval_time:.1f}s")

    if failures:
        print(f"\n{'-' * 80}")
        print("FAILURES:")
        for test_id, query, expected, detail in failures:
            print(f"\n  [{test_id}] {query}")
            print(f"    Expected: {expected}")
            print(f"    {detail}")

    if skipped:
        print(f"\n{'-' * 80}")
        print("SKIPPED:")
        for test_id, query, expected, reason in skipped:
            print(f"  [{test_id}] {query[:60]} -- {reason}")

    print(f"\n{'=' * 80}")

    return results


if __name__ == "__main__":
    results = run_tests()
    sys.exit(0 if results["fail"] == 0 else 1)
