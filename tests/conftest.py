"""Pytest fixtures for testing."""
import os
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("LLAMA_API_KEY", "test-llama-key")
    monkeypatch.setenv("MEM0_API_KEY", "test-mem0-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")


@pytest.fixture
def sample_agent_state() -> Dict[str, Any]:
    """Create a sample agent state for testing."""
    return {
        "query": "What are the symptoms of vitamin D deficiency?",
        "expanded_query": "What are the symptoms, signs, and clinical manifestations of vitamin D deficiency, including bone health issues, fatigue, and immune system effects?",
        "context": [
            {
                "content": "Vitamin D deficiency can cause fatigue, bone pain, muscle weakness, and mood changes.",
                "metadata": {"source": "test.pdf", "page": 1}
            },
            {
                "content": "Severe vitamin D deficiency may lead to rickets in children and osteomalacia in adults.",
                "metadata": {"source": "test.pdf", "page": 2}
            }
        ],
        "response": "Vitamin D deficiency symptoms include fatigue, bone pain, and muscle weakness.",
        "precision_score": 0.0,
        "groundedness_score": 0.0,
        "groundedness_loop_count": 0,
        "precision_loop_count": 0,
        "feedback": "",
        "query_feedback": "",
        "groundedness_check": False,
        "loop_max_iter": 3,
        "timestamp": "2024-01-01T00:00:00"
    }


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="0.85")
    return mock


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    mock = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Test document content about nutrition."
    mock_doc.metadata = {"source": "test.pdf", "page": 1}
    mock.invoke.return_value = [mock_doc]
    return mock


@pytest.fixture
def mock_memory_client():
    """Create a mock Mem0 memory client."""
    mock = MagicMock()
    mock.search.return_value = []
    mock.add.return_value = None
    return mock
