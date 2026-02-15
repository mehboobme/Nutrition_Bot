"""Document ingestion pipeline for processing and storing PDFs."""
import os
import sys
import zipfile
import logging
from pathlib import Path

# Add parent directory to path to resolve 'core' import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma

from core.config import embedding_model, get_config

logger = logging.getLogger(__name__)
config = get_config()


def unzip_data(zip_path: str, extract_to: str) -> None:
    """
    Unzip data files if not already extracted.
    
    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract files to.
    """
    extract_dir = Path(extract_to)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        
    if not any(extract_dir.iterdir()):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted files to: {extract_to}")
    else:
        logger.info(f"Skipping unzip: Files already extracted at {extract_to}")


def process_and_store_documents() -> None:
    """
    Process PDF documents and store them in a vector database.
    
    Uses semantic chunking for better retrieval performance.
    """
    # Paths from config
    zip_path = config.data_dir / "Nutritional_Medical_Reference.zip"
    unzip_to = config.data_dir / "unzipped_docs"
    pdf_folder = unzip_to / "Nutritional Medical Reference"
    vector_db_dir = config.vector_db_dir
    collection_name = "semantic_chunks"

    # Skip entire process if Chroma DB already exists
    chroma_collections_path = vector_db_dir / "chroma-collections"
    if chroma_collections_path.exists():
        logger.warning(f"Vector DB already exists at '{vector_db_dir}'. Skipping.")
        return

    unzip_data(str(zip_path), str(unzip_to))

    logger.info(f"Loading PDFs from {pdf_folder}...")
    pdf_loader = PyPDFDirectoryLoader(str(pdf_folder))
    documents = pdf_loader.load()
    logger.info(f"Loaded {len(documents)} document(s)")

    if not documents:
        raise RuntimeError("No documents loaded. Check PDF directory path.")

    # Semantic chunking
    semantic_text_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=85
    )
    chunks = semantic_text_splitter.split_documents(documents)
    logger.info(f"Semantic chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks created from documents. Check splitting logic.")

    # Store in Chroma vector store
    logger.info("Storing chunks in Chroma vector DB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(vector_db_dir),
        collection_name=collection_name
    )

    logger.info(f"Vectorstore created with {len(chunks)} semantic chunks at {vector_db_dir}")


if __name__ == "__main__":
    process_and_store_documents()
