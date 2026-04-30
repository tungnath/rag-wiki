# RAG-Wiki: Retrieval-Augmented Generation System

A powerful, CPU-friendly Retrieval-Augmented Generation (RAG) system designed to enable semantic search and question-answering across hundreds of PDF documents using local LLMs and embedding models.

## 🎯 Project Overview

RAG-Wiki addresses the challenge of disambiguating homonyms and retrieving accurate information across multiple documents. It uses:

- **Advanced Embeddings**: `mxbai-embed-large` (1024-dimensional) for superior semantic understanding
- **Two-Stage Retrieval**: Dense retrieval + intelligent reranking for accurate results
- **Local-First Architecture**: No external APIs—everything runs on your CPU using Ollama
- **Interactive UI**: Streamlit-based interface for intuitive document querying

### Key Features

- ✅ Multi-document support (100+ PDFs)
- ✅ Acronym disambiguation (e.g., MCP = Model Context Protocol vs Mobile Content Provider)
- ✅ Accurate source citations with page numbers
- ✅ CPU-only inference (no GPU required)
- ✅ Semantic search across unstructured documents
- ✅ Easy document ingestion pipeline
- ✅ Comprehensive test suite for validation

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **Ollama**: [Install from ollama.ai](https://ollama.ai)
- **RAM**: 8GB+ recommended (4GB minimum)
- **Disk Space**: ~2GB for models + database

### Step 1: Download LLM and Embedding Models

```bash
# Pull the small, fast LLM (choose one based on your preference)
ollama pull llama3.2:3b      # Fast, 3B parameters
# OR
ollama pull phi3:3.8b        # Compact, good quality
# OR
ollama pull gemma2:2b        # Lightweight, 2B parameters

# Pull the embedding model (critical for semantic understanding)
ollama pull mxbai-embed-large  # 1024 dimensions, better than nomic-embed-text
```

**Note**: Each model is ~500MB-2GB. Ensure Ollama is running before proceeding.

### Step 2: Set Up Project Structure

```bash
# Create project directory
mkdir rag-wiki && cd rag-wiki

# Create documents folder (place all your PDFs here)
mkdir documents

# Optional: Create virtual environment location
mkdir venv
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
pip install streamlit langchain langchain-community langchain-ollama langchain-chroma langchain-text-splitters chromadb pypdf pymupdf
```

### Step 5: Add Application Code

Copy `app.py` to your project root directory (or clone the repository).

### Step 6: Add Documents

Place all your PDF files in the `documents/` folder:

```
rag-wiki/
├── documents/
│   ├── file1.pdf
│   ├── file2.pdf
│   ├── file3.pdf
│   └── ...
├── app.py
├── venv/
└── README.md
```

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` in your browser.

**First Run**: Expect 5-10 minutes for initial embedding generation. Subsequent queries will be faster as embeddings are cached.

## 📖 Architecture Overview

### Retrieval Pipeline

```
Query
  ↓
[Embedding with mxbai-embed-large]
  ↓
[MMR Search: Dense Retrieval]
  ↓
[Reranking: Document Diversity + Semantic Scoring]
  ↓
[Context Assembly: Grouped by Source Document]
  ↓
[LLM Generation with llama3.2:3b]
  ↓
Response + Citations
```

### Why mxbai-embed-large?

- **1024 dimensions** vs 384 (nomic-embed-text) = 2.7x more expressive capacity
- **Better semantic understanding** for acronym disambiguation
- **Trained on diverse tasks** including terminology understanding
- Significantly improves retrieval accuracy for ambiguous queries

**Migration Note**: If switching from `nomic-embed-text`, delete the `chroma_db/` folder before re-running (incompatible embedding dimensions).

## 📁 Project Structure

```
rag-wiki/
├── app.py                      # Main Streamlit application
├── README.md                   # This file
├── requirements.txt            # Python dependencies (optional)
├── .gitignore                  # Git ignore rules
├── documents/                  # Your PDF files (100+ supported)
│   ├── file1.pdf
│   ├── file2.pdf
│   └── ...
├── chroma_db/                  # Vector database (auto-generated)
│   ├── chroma.sqlite3
│   └── ...
├── scripts/                    # Utility scripts
│   ├── __init__.py
│   ├── test_runner.py
│   ├── debug_pdf.py
│   └── diagnose.py
└── prepdocs/                   # Documentation
    ├── plan.md                 # Implementation plan
    ├── TEST_CASES.md          # Test suite documentation
    └── TEST_CASES_EXECUTION.md # Test execution results
```

## 🔧 Configuration

Edit `app.py` to customize:

```python
# Model Configuration
LLM_MODEL = "llama3.2:3b"           # Change LLM model
EMBED_MODEL = "mxbai-embed-large"   # Change embedding model

# Retrieval Parameters
CHUNK_SIZE = 800                     # Characters per chunk
CHUNK_OVERLAP = 100                  # Overlap between chunks
K_RETRIEVE = 8                        # Top K chunks to retrieve
FETCH_K = 25                         # Initial candidates before reranking

# Paths
CHROMA_PATH = "./chroma_db"          # Vector database location
DOCUMENTS_PATH = "./documents"       # PDF documents location
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Execute all test cases
python scripts/test_runner.py

# Or use the alternative test script
python scripts/run_tests.py
```

View test results in `prepdocs/TEST_CASES_EXECUTION.md`.

### Example Test Cases

1. **Acronym Disambiguation**: "What is MCP?" correctly identifies Model Context Protocol vs Mobile Content Provider
2. **Multi-document Queries**: Queries spanning multiple PDFs return accurate cross-document answers
3. **Source Citations**: All retrieved information includes source document and page numbers
4. **Edge Cases**: Handles special characters, long queries, and ambiguous terminology

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| First-run embedding time | 5-10 min (100 PDFs) |
| Subsequent query time | 2-5 sec (retrieval) + 5-15 sec (LLM inference) |
| Memory footprint | ~2-3 GB during inference |
| Embedding model size | ~500 MB |
| LLM model size | ~1.5 GB (llama3.2:3b) |

**Note**: All times are CPU-only estimates. GPU acceleration will significantly improve inference speed.

## 🐛 Troubleshooting

### Issue: Model Download Fails

```bash
# Ensure Ollama is running
ollama serve

# In another terminal, pull models
ollama pull mxbai-embed-large
ollama pull llama3.2:3b
```

### Issue: Vector Database Incompatible

```bash
# Delete old chroma_db if switching embedding models
rm -r ./chroma_db
# Windows: rmdir /s /q chroma_db

# Restart the application
streamlit run app.py
```

### Issue: "Model not found" Error

```bash
# List available models
ollama list

# Pull missing model
ollama pull mxbai-embed-large
```

### Issue: Slow Performance / High Memory Usage

- **Reduce CHUNK_SIZE** from 800 to 500-600
- **Reduce K_RETRIEVE** from 8 to 5-6
- **Use smaller LLM**: Try `gemma2:2b` or `phi3:3.8b` instead of `llama3.2:3b`
- **Consider GPU acceleration** if available

### Issue: Streamlit App Won't Start

```bash
# Verify all dependencies installed
pip list | grep -E "streamlit|langchain|chromadb"

# Reinstall dependencies
pip install --upgrade streamlit langchain langchain-community langchain-ollama langchain-chroma langchain-text-splitters chromadb pypdf pymupdf
```

## 🤝 Contributing

We welcome contributions! Here's how to help:

### 1. Fork and Clone

```bash
git clone <your-fork-url>
cd rag-wiki
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Modify code in `app.py` or scripts
- Add new test cases to `prepdocs/TEST_CASES.md`
- Update documentation

### 4. Run Tests

```bash
python scripts/test_runner.py
```

Ensure all tests pass before submitting.

### 5. Submit Pull Request

- Provide clear description of changes
- Include test results
- Reference any related issues

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and single-purpose

### Adding New Features

1. **Update plan.md** with your proposal
2. **Implement in app.py** with inline comments
3. **Add test cases** to TEST_CASES.md
4. **Update README.md** with new configuration options (if applicable)
5. **Test thoroughly** before committing

### Document Structure

When adding PDF documents:

- Use clear, descriptive filenames (e.g., `Intro_to_MCP.pdf`)
- Avoid special characters in filenames
- Organize related documents logically
- Consider document size (very large PDFs may slow ingestion)

## 📚 Additional Resources

- **Langchain Documentation**: [https://python.langchain.com](https://python.langchain.com)
- **Ollama Models**: [https://ollama.ai/library](https://ollama.ai/library)
- **Streamlit Docs**: [https://docs.streamlit.io](https://docs.streamlit.io)
- **Chroma DB**: [https://www.trychroma.com](https://www.trychroma.com)

## 🎓 How RAG Works

### Retrieval-Augmented Generation Flow

1. **Document Processing**: PDFs are split into overlapping chunks
2. **Embedding**: Each chunk is converted to a vector using `mxbai-embed-large`
3. **Storage**: Vectors are stored in Chroma vector database
4. **Query Embedding**: User query is embedded with the same model
5. **Retrieval**: Top-K similar chunks are retrieved using vector similarity
6. **Reranking**: Chunks are re-scored for document diversity
7. **Context Assembly**: Retrieved chunks are organized by source document
8. **Generation**: LLM generates response using context + query
9. **Citation**: Sources and page numbers are included in response

### Why This Matters

- **Better Accuracy**: Retrieval grounds LLM in actual document content (reduces hallucinations)
- **Transparency**: Users see exactly which documents answered their question
- **Scalability**: Works with hundreds of documents without memory explosion
- **Privacy**: All processing happens locally—no data sent to external APIs

## 🚫 What NOT to Do

- ❌ Don't commit the `chroma_db/` folder (too large, auto-generated)
- ❌ Don't commit model files or `ollama/` directories
- ❌ Don't hardcode API keys or secrets in code
- ❌ Don't manually edit the `.gitignore` without reviewing changes
- ❌ Don't remove test cases without community discussion


---

## 🎉 Quick Checklist for First-Time Setup

- [ ] Ollama installed and running
- [ ] Models downloaded (`llama3.2:3b` + `mxbai-embed-large`)
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed via `pip install`
- [ ] PDF documents placed in `documents/` folder
- [ ] `app.py` in project root
- [ ] Run `streamlit run app.py`
- [ ] Access app at `http://localhost:8501`

## 🔄 Workflow for Contributors

```
1. Review plan.md for current issues/roadmap
2. Check TEST_CASES.md for expected behavior
3. Run existing tests: python scripts/test_runner.py
4. Make your changes
5. Add/update test cases if needed
6. Run full test suite
7. Update documentation
8. Submit PR with test results
```

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Stable with ongoing improvements

For the latest updates and discussions, check the project issues and pull requests.

