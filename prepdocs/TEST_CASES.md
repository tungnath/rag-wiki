# TEST CASES DOCUMENTATION
**RAG Wiki Retrieval System Test Specifications**

**Last Updated**: April 29, 2026  
**Test Script**: `scripts/test_runner.py`  
**Execution Summary**: See `TEST_CASES_EXECUTION.md`

---

## Overview

This document defines all test cases for validating RAG Wiki retrieval system. Tests validate:
- Semantic retrieval accuracy
- Source attribution correctness
- Handling of edge cases
- Performance metrics
- Multi-document queries

**Run tests with**: `python scripts/test_runner.py`

---

## Test Execution Instructions

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Run test suite:
   ```bash
   python scripts/test_runner.py
   ```

3. Check results in `TEST_CASES_EXECUTION.md`

---

## Category 1: Acronym Disambiguation (Critical)

### Test 1.1: "What is MCP?"
**Query**: `What is MCP?`
**Expected**:
- Cite `Intro_to_MCP.pdf` only
- Explain Model Context Protocol
- No hallucinated sources

**Pass Criteria**:
- Only Intro_to_MCP.pdf mentioned
- Accurate MCP explanation
- Zero false citations

---

### Test 1.2: "Tell me about Mobile Content Provider"
**Query**: `Tell me about Mobile Content Provider`
**Expected**:
- Term not found in any documents
- Graceful handling
- No false citations

**Pass Criteria**:
- Handles missing term without errors
- No fake sources cited
- Clear indication of no content found

---

### Test 1.3: "Explain Model Context Protocol"
**Query**: `Explain Model Context Protocol`
**Expected**:
- Prioritize `Intro_to_MCP.pdf`
- Accurate explanation

**Pass Criteria**:
- Correct document prioritized
- Accurate information provided

---

### Test 1.4: "What does MCP stand for?"
**Query**: `What does MCP stand for?`
**Expected**:
- List all meanings with sources
- Comprehensive coverage

**Pass Criteria**:
- Multiple meanings listed
- Proper source attribution

---

## Category 2: Single Document Queries (Baseline)

### Test 2.1: "What are Linux commands?"
**Query**: `What are Linux commands?`
**Expected**:
- Cite `linux-commands.pdf` only
- List common commands
- Accurate information

**Pass Criteria**:
- Only linux-commands.pdf cited
- No noise from other documents
- Accurate command info

---

### Test 2.2: "Explain SQL Window Functions"
**Query**: `Explain SQL Window Functions`
**Expected**:
- Cite `SQL_Window_Functions_1_1.pdf`
- Explain window functions

**Pass Criteria**:
- Correct document cited
- Accurate SQL information

---

### Test 2.3: "AI-102 exam topics?"
**Query**: `What are AI-102 exam topics?`
**Expected**:
- Cite `AI-102 BL CT1.pdf`
- List relevant topics

**Pass Criteria**:
- Correct document cited
- Relevant topics listed

---

### Test 2.4: "What are the different types of screens in an Agentry app?"
**Query**: `What are the different types of screens in an Agentry app?`
**Expected**:
- Cite `smp_agentry_language_reference.pdf`
- List screen types
- Accurate information

**Pass Criteria**:
- Correct document cited
- Accurate screen types
- Proper source attribution

---

## Category 3: Cross-Document Queries (Advanced)

### Test 3.1: "Compare different software frameworks mentioned"
**Query**: `Compare different software frameworks mentioned in the documents`
**Expected**:
- Retrieve from multiple docs
- Provide comparison

**Pass Criteria**:
- Multiple documents cited
- Comparison provided

---

### Test 3.2: "What are the different types of data processing?"
**Query**: `What are the different types of data processing?`
**Expected**:
- Span multiple documents
- Comprehensive answer

**Pass Criteria**:
- Diverse sources
- Comprehensive coverage

---

### Test 3.3: "Explain enterprise mobility concepts"
**Query**: `Explain enterprise mobility concepts`
**Expected**:
- Cite SAP-related docs
- Explain concepts

**Pass Criteria**:
- Relevant SAP docs cited
- Concepts explained

---

## Category 4: Edge Cases (Robustness)

### Test 4.1: Empty Query
**Query**: (empty string)
**Expected**:
- Graceful handling
- No crash

**Pass Criteria**:
- No application crash
- Helpful response

---

### Test 4.2: Query with Special Characters
**Query**: `What is AI/ML & how does it work?`
**Expected**:
- Parse correctly
- Retrieve relevant info

**Pass Criteria**:
- Correct parsing
- Relevant results

---

### Test 4.3: Very Long Query
**Query**: `Discuss the fundamentals of machine learning, including supervised learning, unsupervised learning, and reinforcement learning...` (500+ words)
**Expected**:
- Prioritize main terms
- No truncation issues

**Pass Criteria**:
- Successful processing
- No truncation

---

### Test 4.4: Query with Document Filename
**Query**: `What is in AI-102 BL CT1.pdf?`
**Expected**:
- Use filename as relevance signal
- Retrieve from that doc

**Pass Criteria**:
- Correct doc prioritized
- Accurate content

---

## Category 5: Performance & Scaling

### Test 5.1: Embedding Time
**Measure**: Time to ingest 8 PDFs
**Target**: <5 minutes
**Accept**: Baseline for scaling

---

### Test 5.2: Retrieval Time
**Measure**: Query response time
**Target**: <30 seconds
**Accept**: User-acceptable latency

---

### Test 5.3: End-to-End Response
**Measure**: Time from query to final answer
**Target**: <60 seconds
**Accept**: User-acceptable total time

---

### Test 5.4: Scaling Test
**Measure**: Handling 8+ PDFs
**Target**: No degradation
**Accept**: Baseline performance

---

## Category 6: Source Citation (UX)

### Test 6.1: Source Listing
**Query**: `What are Linux commands?`
**Expected**:
- All sources listed
- Page numbers included

**Pass Criteria**:
- Complete source list
- Accurate page numbers

---

### Test 6.2: In-Text Citations
**Query**: `Explain SQL Window Functions`
**Expected**:
- Citations in response body
- Clear attribution

**Pass Criteria**:
- Citations present
- Properly formatted

---

### Test 6.3: No Hallucinations
**Query**: `What is not in any document?`
**Expected**:
- Only real docs cited
- No false sources

**Pass Criteria**:
- Zero hallucinated sources
- Only factual citations

---

## Success Criteria Summary

All tests must meet these criteria for system approval:

- [x] Pass Rate: ≥95%
- [x] Hallucination Rate: 0%
- [x] Source Accuracy: 100%
- [x] Performance: All targets met
- [x] Scalability: Handles 8+ PDFs
- [x] Generic Prompting: No hardcoding

---

## Test Categorization by Priority

| Priority | Tests | Description |
|----------|-------|-------------|
| **Critical** | 1.1-1.4 | Acronym disambiguation - core functionality |
| **High** | 2.1-2.4 | Single document queries - baseline |
| **Medium** | 3.1-3.3 | Multi-document queries - advanced |
| **Medium** | 4.1-4.4 | Edge cases - robustness |
| **Low** | 5.1-5.4 | Performance metrics - non-blocking |
| **Low** | 6.1-6.3 | Source citation UX - non-blocking |

---

## Notes

- Record results in `TEST_CASES_EXECUTION.md` after each run
- Update execution summary with pass/fail status
- Screenshots of failures helpful for debugging
- Run complete suite before deployment
- Extend test cases as new scenarios identified

---

**For execution results**: See `TEST_CASES_EXECUTION.md`

