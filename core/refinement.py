from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List, Any
from agents.agent_state import AgentState
from core.config import llm, endpoint, api_key, embedding_model, llamaparse_api_key, embedding_function
from core.retriever import multi_retriever

#refine response
def refine_response(state: Dict) -> Dict:
    """
    Suggests improvements for the generated response.

    Args:
        state (Dict): The current state of the workflow, containing the query and response.

    Returns:
        Dict: The updated state with response refinement suggestions.
    """
    print("---------refine_response---------")

    system_message = '''You are a medical expert assistant specialized in nutritional and metabolic disorders.
Given a user query and the AI-generated response, suggest clear, specific improvements
to enhance the accuracy, completeness, and relevance of the response.
Focus on correcting any factual inaccuracies, adding missing details, and improving clarity.
Provide your suggestions as concise bullet points or a short list without rewriting the entire response.'''

    refine_response_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\n"
                 "What improvements can be made to enhance accuracy and completeness?")
    ])

    chain = refine_response_prompt | llm | StrOutputParser()

    # Store response suggestions in a structured format
    feedback = f"Previous Response: {state['response']}\nSuggestions: {chain.invoke({'query': state['query'], 'response': state['response']})}"
    print("feedback: ", feedback)
    print(f"State: {state}")
    state['feedback'] = feedback
    return state

#refine query
def refine_query(state: Dict) -> Dict:
    """
    Suggests improvements for the expanded query.

    Args:
        state (Dict): The current state of the workflow, containing the query and expanded query.

    Returns:
        Dict: The updated state with query refinement suggestions.
    """
    print("---------refine_query---------")
    system_message = '''You are an expert medical research assistant specialized in nutritional disorders.
Given the original user query and its expanded version, suggest clear and specific improvements
to make the expanded query more comprehensive and effective for retrieving relevant documents.
Focus on adding synonyms, related concepts, clarifications, and removing ambiguities
to enhance search recall and precision. Provide concise suggestions without rewriting the entire query.'''

    refine_query_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Original Query: {query}\nExpanded Query: {expanded_query}\n\n"
                 "What improvements can be made for a better search?")
    ])

    chain = refine_query_prompt | llm | StrOutputParser()

    # Store refinement suggestions without modifying the original expanded query
    query_feedback = f"Previous Expanded Query: {state['expanded_query']}\nSuggestions: {chain.invoke({'query': state['query'], 'expanded_query': state['expanded_query']})}"
    print("query_feedback: ", query_feedback)
    print(f"Groundedness loop count: {state['groundedness_loop_count']}")
    state['query_feedback'] = query_feedback
    return state