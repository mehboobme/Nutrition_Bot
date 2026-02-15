"""
Configuration module for the Advanced RAG application.

Handles environment variables, model initialization, and application settings.
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from llama_index.core import Settings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class AppConfig:
    """Application configuration with validation."""
    
    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    llama_api_key: str = field(default_factory=lambda: os.getenv("LLAMA_API_KEY", ""))
    mem0_api_key: str = field(default_factory=lambda: os.getenv("MEM0_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    
    # Model settings
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
    
    # RAG settings
    groundedness_threshold: float = 0.7
    precision_threshold: float = 0.7
    max_refinement_iterations: int = 3
    retrieval_top_k: int = 5
    
    # Paths (relative to project root)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    vector_db_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "research_db")
    hyp_questions_db_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "hyp_question_db")
    
    def validate(self) -> None:
        """Validate required configuration values."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        if not self.openai_api_base:
            errors.append("OPENAI_API_BASE is required")
        if not self.llama_api_key:
            logger.warning("LLAMA_API_KEY not set - document parsing may fail")
        if not self.mem0_api_key:
            logger.warning("MEM0_API_KEY not set - memory features disabled")
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set - guardrails disabled")
            
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get cached application configuration."""
    config = AppConfig()
    config.validate()
    return config


# Initialize configuration
config = get_config()

# Expose commonly used values for backward compatibility
api_key = config.openai_api_key
endpoint = config.openai_api_base
llamaparse_api_key = config.llama_api_key
MEM0_api_key = config.mem0_api_key

# Initialize the OpenAI embedding function for Chroma
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_base=endpoint,
    api_key=api_key,
    model_name=config.embedding_model_name
)

# Initialize the OpenAI Embeddings for LangChain
embedding_model = OpenAIEmbeddings(
    openai_api_base=endpoint,
    openai_api_key=api_key,
    model=config.embedding_model_name
)

# Initialize the Chat OpenAI model
llm = ChatOpenAI(
    openai_api_base=endpoint,
    openai_api_key=api_key,
    model=config.chat_model,
    streaming=True,
    temperature=0,
    max_retries=3,
)

# Set the LLM and embedding model in the LlamaIndex settings
Settings.llm = llm
Settings.embed_model = embedding_model

logger.info(f"Configuration loaded: model={config.chat_model}, embedding={config.embedding_model_name}")