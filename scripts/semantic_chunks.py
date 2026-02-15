"""Semantic chunks retrieval from the vector store."""
import logging
from langchain_community.vectorstores import Chroma

from core.config import embedding_model, get_config

logger = logging.getLogger(__name__)
config = get_config()

# Use config for paths
vector_db_dir = str(config.vector_db_dir)
collection_name = "semantic_chunks"


def get_vectorstore() -> Chroma:
    """Get the semantic chunks vector store."""
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=vector_db_dir,
        collection_name=collection_name
    )


def get_semantic_chunks(query: str = "", k: int = 1000):
    """
    Retrieve semantic chunks from the vector store.
    
    Args:
        query: Search query (empty string retrieves all).
        k: Maximum number of chunks to retrieve.
        
    Returns:
        List of documents.
    """
    vs = get_vectorstore()
    return vs.similarity_search(query=query, k=k)


# For backwards compatibility
vectorstore = get_vectorstore()
semantic_chunks = get_semantic_chunks()
