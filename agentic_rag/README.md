# Agentic RAG

A course Q&A assistant demonstrating two approaches to Retrieval-Augmented Generation (RAG):
1. **Traditional RAG** - Direct search + context + LLM generation
2. **Agentic RAG** - Agent-based approach with tool calling for intelligent search

This project is part of the LLM Zoomcamp 2026 course.

## Overview

### What is RAG?

Retrieval-Augmented Generation (RAG) combines:
- **Retrieval**: Finding relevant documents from a knowledge base
- **Context**: Providing those documents to the LLM
- **Generation**: Using an LLM to generate answers based on the context

### Traditional RAG vs Agentic RAG

**Traditional RAG (RAGBase)**:
- Performs a single search query
- Constructs context from search results
- Sends context + question to LLM
- Simple, fast, direct approach

**Agentic RAG (AgenticRAGBase)**:
- Uses an LLM agent with a search tool
- Agent can decide when to search and what to search for
- Can perform multiple searches with different strategies
- Uses alternative keywords and synonyms automatically
- More intelligent but potentially slower

## Features

- **Two RAG Approaches**: Traditional and Agentic implementations for comparison
- **Local LLM**: Ollama with Llama 3.2
- **Multi-strategy Search**: Keyword matching with boosting and course filtering
- **FAQ Database**: Course-specific Q&A knowledge base
- **LangChain Integration**: Extensible with LangChain ecosystem
- **Modern Packaging**: `pyproject.toml` and `uv` for reproducible builds

## Project Structure

```
agentic_rag/
├── main.py                          # Entry point, runs both RAG implementations
├── pyproject.toml                   # Project dependencies and metadata
├── README.md                        # This file
└── utils/
    ├── agentic_rag_helper.py       # AgenticRAGBase class (agent-based approach)
    ├── rag_helper.py               # RAGBase class (traditional approach)
    └── load_dataset.py             # FAQ data loading and indexing
```

## Installation

### Prerequisites
- Python 3.13+
- Ollama installed and running locally
- `uv` package manager

### Setup

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai) and pull the model:
   ```bash
   ollama pull llama3.2
   ```

2. **Install `uv`**:
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or via pip
   pip install uv
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```
   Or with pip: `pip install -e .`

## Usage

### Quick Start

```bash
# Run both RAG approaches
uv run main.py

# Or with Python directly
python main.py
```

### Common `uv` Commands

```bash
uv sync              # Install dependencies
uv run main.py       # Run the script
uv add package-name  # Add a package
uv remove package-name  # Remove a package
uv sync --upgrade    # Update dependencies
```

### Customize

Edit `main.py` to run individual approaches:
```python
main(run_rag=True, run_agentic_rag=True)    # Both (default)
main(run_rag=True, run_agentic_rag=False)   # Traditional RAG only
main(run_rag=False, run_agentic_rag=True)   # Agentic RAG only
```

### In Your Code

```python
# Traditional RAG
from utils.rag_helper import RAGBase
from utils.load_dataset import build_index, load_faq_data
from langchain_ollama import ChatOllama

documents = load_faq_data()
index = build_index(documents)
llm = ChatOllama(model="llama3.2", temperature=0.2)
assistant = RAGBase(index, llm_model=llm)
response = assistant.rag("Your question")

# Agentic RAG
from utils.agentic_rag_helper import AgenticRAGBase
agent = AgenticRAGBase(index, llm_model=llm)
response = agent.chat("Your question")
```

## Core Classes

### RAGBase
Simple retrieval-augmented generation pipeline.

**Methods**: `rag(query)`, `search(query)`, `build_context()`, `llm_response(prompt)`

**Parameters**: `index`, `llm_model`, `instructions`, `prompt_template`, `course`

### AgenticRAGBase
Agent-based RAG with intelligent tool calling.

**Methods**: `chat(query)` (with dynamic search tool)

**Features**: Multi-step reasoning, alternative keyword searching, automatic strategy refinement

**Parameters**: Same as RAGBase

## Configuration

- **Model**: Llama 3.2 via Ollama
- **Temperature**: 0.2 (consistent answers)
- **Search Boost**: Questions weighted 3x more than sections
- **Top-k Results**: 5 documents per search
- **Course Filter**: Filters by course name ("llm-zoomcamp")

Edit `INSTRUCTIONS` in helper classes to customize behavior.

## Dependencies

| Package | Purpose |
|---------|---------|
| langchain | LLM framework |
| langchain-ollama | Ollama integration |
| minsearch | Keyword search |
| ollama | Local LLM client |
| pandas | Data processing |
| python-dotenv | Environment config |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Run `ollama serve` |
| Model not found | Run `ollama pull llama3.2` |
| Slow responses | Reduce `num_results`, use smaller model |
| `uv` command not found | Install with `pip install uv` |
| Environment issues | Run `uv sync --fresh` or `uv cache clean` |

## Learning Objectives

- Build a knowledge base from FAQ data
- Implement Traditional RAG pipeline
- Implement Agent-based RAG with tool calling
- Work with LangChain framework and local LLMs
- Compare different RAG approaches

## Next Steps

- Expand FAQ dataset with more course materials
- Experiment with vector embeddings and similarity search
- Fine-tune prompts and system instructions
- Add more tools (web search, database queries)
- Deploy as a REST API with FastAPI
- Evaluate and compare accuracy/latency metrics

## References

- [LangChain Docs](https://python.langchain.com/) - Framework documentation
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [uv Package Manager](https://astral.sh/uv/) - Fast Python package manager

## License

Part of LLM Zoomcamp 2026 course materials.

## Support

For questions about this project, refer to the course materials or community forums.
