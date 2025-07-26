import os
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv()

import json
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from llama_index.core import Settings

# Define a function to read a JSON config file and return its contents as a dictionary.
# def read_config(config_file):
#   """Reads a JSON config file and returns a dictionary."""
#   with open(config_file, 'r') as f:
#     return json.load(f)
  
# config = read_config("config_GANLP.json")  #Copy and paste the path of the config file uploaded in Colab
# config2 = read_config("config2_emb_tested.json")
api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("OPENAI_API_BASE")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
llamaparse_api_key = os.getenv("LLAMA_API_KEY")
MEM0_api_key = os.getenv("MEM0_API_KEY")  
# Initialize the OpenAI embedding function for Chroma
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_base=endpoint,  # Fill in the API base URL
    api_key=api_key,  # Fill in the API key
    model_name='text-embedding-ada-002'  # Fill in the model name
)
# This initializes the OpenAI embedding function for the Chroma vectorstore, using the provided endpoint and API key.

# Initialize the OpenAI Embeddings
embedding_model = OpenAIEmbeddings(
    openai_api_base=endpoint,  # Fill in the endpoint
    openai_api_key=api_key,  # Fill in the API key
    model='text-embedding-ada-002'                 # Fill in the model name
)
# This initializes the OpenAI embeddings model using the specified endpoint, API key, and model name.

# Initialize the Chat OpenAI model
llm = ChatOpenAI(
    openai_api_base=endpoint,  # Fill in the endpoint
    openai_api_key=api_key,  # Fill in the API key
    model="gpt-4o-mini",  # Fill in the deployment name (e.g., gpt-4o-mini)
    streaming=True  # Enable streaming for real-time responses
)
# This initializes the Chat OpenAI model using the provided endpoint, API key, deployment name.

# set the LLM and embedding model in the LlamaIndex settings.
Settings.llm = llm
Settings.embedding = embedding_model

if not api_key or not endpoint:
    raise ValueError("Missing OpenAI API key or endpoint in config_GANLP.json")

if not llamaparse_api_key:
    raise ValueError("Missing LLAMA_KEY in config2_emb_tested.json")