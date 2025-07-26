#Expanded query
from typing import Dict, List, Any

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from core.config import llm, endpoint, api_key, embedding_model, llamaparse_api_key, embedding_function
from core.retriever import text_vectorstore, table_vectorstore

from agents.agent_state import AgentState
from core.retriever import multi_retriever

from agents.agent_state import AgentState

def expand_query(state: AgentState) -> AgentState:
    """
    Expands the user query to improve retrieval of nutrition disorder-related information using few-shot prompting.

    Args:
        state (Dict): The current state of the workflow, containing the user query.

    Returns:
        Dict: The updated state with the expanded query.
    """
    system_message = '''You are an expert medical research assistant specialized in nutritional disorders. Given a brief user query, rewrite and expand it into a detailed, comprehensive search query
    that covers relevant aspects such as dietary deficiencies, metabolic disorders, vitamin and mineral imbalances, obesity, and related health conditions. Make sure the expanded query improves retrieval by including synonyms, related terms, and clarifications.'''

    expand_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{query}")
    ])

    chain = expand_prompt | llm | StrOutputParser()
    state['expanded_query'] = chain.invoke({"query": state['query']})
    return state

# Retrieve context
def retrieve_context(state: AgentState) -> AgentState:
    """
    Retrieves context from the vector store using the expanded or original query.

    Args:
        state (Dict): The current state of the workflow, containing the query and expanded query.

    Returns:
        Dict: The updated state with the retrieved context.
    """
    query = state['expanded_query']
    print("Query used for retrieval:", query)  # Debugging: Print the query

    # Retrieve documents from the vector store
    docs = multi_retriever.invoke(query)
    print("Retrieved documents:", docs)  # Debugging: Print the raw docs object

    # Extract both page_content and metadata from each document
    state['context'] = [
        {
            "content": doc.page_content,  # The actual content of the document
            "metadata": doc.metadata  # The metadata (e.g., source, page number, etc.)
        }
        for doc in docs
    ]

    print("Extracted context with metadata:", state['context'])  # Debugging: Print the extracted context
    return state

# craft response
def craft_response(state: Dict) -> Dict:
    """
    Generates a response using the retrieved context, focusing on nutrition disorders.

    Args:
        state (Dict): The current state of the workflow, containing the query and retrieved context.

    Returns:
        Dict: The updated state with the generated response.
    """
    print("---------craft_response---------")
    system_message = '''You are a knowledgeable medical assistant specialized in nutritional and metabolic disorders.
Using the provided context from authoritative documents, answer the user's query clearly and concisely.
Focus on evidence-based information and ensure your response is relevant to the user's question.
If the context does not provide a direct answer, acknowledge it and offer a best-effort response based on related information.'''

    response_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nContext: {context}\n\nfeedback: {feedback}")
    ])

    chain = response_prompt | llm
    response = chain.invoke({
        "query": state['query'],
        "context": "\n".join([doc["content"] for doc in state['context']]),
        "feedback": state.get('feedback', '') # add feedback to the prompt
    })
    state['response'] = response
    print("intermediate response: ", response)

    return state


