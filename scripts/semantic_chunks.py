from langchain_community.vectorstores import Chroma
from core.config import embedding_model

# Define the vector DB path and collection name used earlier
vector_db_dir = "C:/NLP/GL/Advance_RAG_Project/research_db"
collection_name = "semantic_chunks"

# Load the vectorstore
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=vector_db_dir,
    collection_name=collection_name
)

# Retrieve stored documents (semantic chunks)
semantic_chunks = vectorstore.similarity_search(query="", k=1000)

# Iterate over them
for i, document in enumerate(semantic_chunks):
    print(document.page_content)      # ✅ OK
    print(document.metadata['source'])  # ✅ OK if metadata was stored
