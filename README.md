# Advanced RAG System for Nutritional Disorders

An industry-grade Retrieval-Augmented Generation (RAG) application specialized in nutritional and metabolic disorders. Built with LangChain, LangGraph, and ChromaDB.

## Features

- **Advanced RAG Pipeline**: Multi-stage workflow with query expansion, retrieval, response generation, and iterative refinement
- **Self-Query Retriever**: Metadata-aware retrieval with filtering on Category, DisorderType, and Page
- **Hypothetical Question Indexing (HyDE)**: Generates questions from document chunks for improved retrieval accuracy
- **Multi-Retriever Architecture**: Combines text and table retrievers for comprehensive coverage
- **Conversation Memory**: Mem0 integration for personalized, context-aware responses
- **Safety Guardrails**: Llama Guard 3 (via Groq) for content safety filtering
- **Streamlit UI**: User-friendly chat interface with session management

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Llama Guard (Safety)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Expansion (LLM)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Retriever (Text + Tables)                     │
│         ┌─────────────────┬─────────────────────┐               │
│         │  Self-Query     │   Self-Query        │               │
│         │  (Text)         │   (Tables)          │               │
│         └─────────────────┴─────────────────────┘               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Response Generation                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Evaluation Loop                                 │
│    ┌──────────────────────────────────────────────────┐         │
│    │  Groundedness Check → Refine Response (if < 0.7) │         │
│    │  Precision Check → Refine Query (if < 0.7)       │         │
│    └──────────────────────────────────────────────────┘         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Final Response                               │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Advance_RAG_Project/
├── main.py                     # Application entry point
├── Dockerfile                  # Container configuration
├── requirements.txt            # Dependencies
├── pyproject.toml             # Project metadata
│
├── agent_workflow/            # LangGraph workflow
│   ├── workflow.py            # Main workflow definition
│   └── tools/
│       └── rag.py             # RAG tool for agent
│
├── agents/                    # Agent components
│   ├── agent_state.py         # State definition
│   ├── agent_steps.py         # Workflow step functions
│   └── guard.py               # Llama Guard integration
│
├── core/                      # Core modules
│   ├── config.py              # Configuration management
│   ├── logging_config.py      # Logging setup
│   ├── evaluation.py          # Groundedness/precision scoring
│   ├── refinement.py          # Response/query refinement
│   ├── retriever.py           # Multi-retriever setup
│   └── routing.py             # Conditional routing logic
│
├── parsers/                   # Document parsing
│   └── llama_parser.py        # LlamaParse integration
│
├── pipeline/                  # Data pipeline
│   └── ingest_documents.py    # Document ingestion
│
├── scripts/                   # Utility scripts
│   ├── semantic_chunks.py     # Semantic chunking
│   └── build_hypothetical_q_store.py  # HyDE indexing
│
├── services/                  # Services
│   └── bot.py                 # NutritionBot class
│
├── ui/                        # User interface
│   └── ui.py                  # Streamlit app
│
└── data/                      # Data storage
    └── unzipped_docs/         # PDF documents
```

## Installation

### Prerequisites

- Python 3.11+
- OpenAI API access (or compatible endpoint)
- LlamaParse API key (for document parsing)
- Mem0 API key (optional, for conversation memory)
- Groq API key (optional, for safety guardrails)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Advance_RAG_Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # Required
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_API_BASE=https://api.openai.com/v1
   
   # For document parsing
   LLAMA_API_KEY=your-llama-parse-api-key
   
   # Optional - for conversation memory
   MEM0_API_KEY=your-mem0-api-key
   
   # Optional - for safety guardrails
   GROQ_API_KEY=your-groq-api-key
   
   # Optional - model overrides
   CHAT_MODEL=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-ada-002
   ```

5. **Prepare documents**
   
   Place your PDF documents in `data/unzipped_docs/Nutritional Medical Reference/`

6. **Build vector stores**
   ```bash
   # First, ingest documents
   python -m pipeline.ingest_documents
   
   # Then build hypothetical question stores
   python -m scripts.build_hypothetical_q_store
   ```

## Usage

### Run the Streamlit App

```bash
streamlit run main.py
```

Or:
```bash
python main.py
```

The app will be available at `http://localhost:8501`

### Docker Deployment

```bash
# Build the image
docker build -t nutrition-rag .

# Run the container
docker run -p 7860:7860 --env-file .env nutrition-rag
```

## Configuration

The application can be configured via environment variables or the `AppConfig` class in `core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `OPENAI_API_BASE` | Required | OpenAI API base URL |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model to use |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model |
| `LLAMA_API_KEY` | Optional | LlamaParse API key |
| `MEM0_API_KEY` | Optional | Mem0 memory API key |
| `GROQ_API_KEY` | Optional | Groq API key for guardrails |

### RAG Parameters

Adjust in `core/config.py`:

```python
groundedness_threshold: float = 0.7  # Min groundedness score
precision_threshold: float = 0.7     # Min precision score
max_refinement_iterations: int = 3   # Max refinement loops
retrieval_top_k: int = 5             # Documents to retrieve
```

## API Reference

### NutritionBot

```python
from services.bot import NutritionBot

bot = NutritionBot()
response = bot.handle_customer_query(
    user_id="user123",
    query="What are the symptoms of vitamin D deficiency?"
)
```

### Direct RAG Tool

```python
from agent_workflow.tools.rag import agentic_rag

result = agentic_rag.invoke("Explain protein-energy malnutrition")
```

## Development

### Logging

The application uses Python's logging module. Configure in `core/logging_config.py`:

```python
from core.logging_config import setup_logging
import logging

setup_logging(level=logging.DEBUG, log_file="app.log")
```

### Running Tests

```bash
pytest tests/
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY is required"**
   - Ensure `.env` file exists with valid API keys

2. **"Vector DB not found"**
   - Run the document ingestion pipeline first

3. **"Llama Guard error"**
   - GROQ_API_KEY not set (guardrails disabled gracefully)

4. **Slow responses**
   - The refinement loop may iterate multiple times
   - Adjust thresholds or max iterations in config

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- LangChain & LangGraph for the orchestration framework
- ChromaDB for vector storage
- LlamaParse for document parsing
- Mem0 for conversation memory
- Groq for Llama Guard inference
# Advance_RAG_Pilot_Generic_Project
