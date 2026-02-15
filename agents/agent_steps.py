"""Agent step functions for the RAG workflow."""
import logging
from typing import Dict, Any

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from agents.agent_state import AgentState
from core.config import llm
from core.retriever import multi_retriever

logger = logging.getLogger(__name__)

def expand_query(state: AgentState) -> AgentState:
    """
    Expand the user query to improve retrieval of nutrition disorder-related information.

    Args:
        state: Current workflow state containing the user query.

    Returns:
        Updated state with the expanded query.
    """
    logger.info(f"Expanding query: {state['query'][:100]}...")
    
    system_message = '''You are an expert medical research assistant specialized in nutritional disorders. Given a brief user query, rewrite and expand it into a detailed, comprehensive search query
    that covers relevant aspects such as dietary deficiencies, metabolic disorders, vitamin and mineral imbalances, obesity, and related health conditions. Make sure the expanded query improves retrieval by including synonyms, related terms, and clarifications.'''

    expand_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{query}")
    ])

    chain = expand_prompt | llm | StrOutputParser()
    state['expanded_query'] = chain.invoke({"query": state['query']})
    logger.debug(f"Expanded query: {state['expanded_query'][:200]}...")
    return state


def retrieve_context(state: AgentState) -> AgentState:
    """
    Retrieve context from the vector store using the expanded or original query.

    Args:
        state: Current workflow state containing the query and expanded query.

    Returns:
        Updated state with the retrieved context.
    """
    query = state['expanded_query']
    logger.info(f"Retrieving context for query: {query[:100]}...")

    # Retrieve documents from the vector store
    docs = multi_retriever.invoke(query)
    logger.info(f"Retrieved {len(docs)} documents")

    # Extract both page_content and metadata from each document
    state['context'] = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    logger.debug(f"Context preview: {str(state['context'])[:300]}...")
    return state


def craft_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a response using the retrieved context, focusing on nutrition disorders.

    Args:
        state: Current workflow state containing the query and retrieved context.

    Returns:
        Updated state with the generated response.
    """
    logger.info("Crafting response...")
    
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
        "feedback": state.get('feedback', '')
    })
    state['response'] = response
    logger.info(f"Response generated (length: {len(str(response))} chars)")

    return state


