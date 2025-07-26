import os
import sys
import zipfile

# Add parent directory to path to resolve 'core' import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma  # ✅ updated per warning
from core.config import embedding_model


# === Step 1: Unzip documents only if needed ===
def unzip_data(zip_path: str, extract_to: str):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    if not any(os.scandir(extract_to)):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted files to: {extract_to}")
    else:
        print(f"📁 Skipping unzip: Files already extracted at {extract_to}")


# === Step 2: Load and split only if DB not exists ===
def process_and_store_documents():
    # Paths
    zip_path = "C:/NLP/GL/Advance_RAG_Project/data/Nutritional_Medical_Reference.zip"
    unzip_to = "C:/NLP/GL/Advance_RAG_Project/data/unzipped_docs"
    pdf_folder = os.path.join(unzip_to, "Nutritional Medical Reference")
    vector_db_dir = "C:/NLP/GL/Advance_RAG_Project/research_db"
    collection_name = "semantic_chunks"

    # Skip entire process if Chroma DB already exists
    if os.path.exists(os.path.join(vector_db_dir, "chroma-collections")):
        print(f"⚠️  Vector DB already exists at '{vector_db_dir}'. Skipping chunking & embedding.")
        return

    unzip_data(zip_path, unzip_to)

    print(f"📄 Loading PDFs from {pdf_folder} ...")
    pdf_loader = PyPDFDirectoryLoader(pdf_folder)
    documents = pdf_loader.load()
    print(f"✅ Loaded {len(documents)} document(s)")

    if not documents:
        raise RuntimeError("❌ No documents loaded. Check PDF directory path.")

    # Semantic chunking
    semantic_text_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=85
    )
    chunks = semantic_text_splitter.split_documents(documents)
    print(f"✂️  Semantic chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("❌ No chunks created from documents. Check splitting logic.")

    # Store in Chroma vector store
    print("💾 Storing chunks in Chroma vector DB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=vector_db_dir,
        collection_name=collection_name
    )

    print(f"✅ Vectorstore created with {len(chunks)} semantic chunks at {vector_db_dir}")


if __name__ == "__main__":
    process_and_store_documents()
