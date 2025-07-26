from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.agent_state import AgentState

from typing import Dict, List, Any
from core.config import llm, endpoint, api_key, embedding_model, llamaparse_api_key, embedding_function
from core.retriever import multi_retriever
#check groundedness
def score_groundedness(state: Dict) -> Dict:
    """
    Checks whether the response is grounded in the retrieved context.

    Args:
        state (Dict): The current state of the workflow, containing the response and context.

    Returns:
        Dict: The updated state with the groundedness score.
    """
    print("---------check_groundedness---------")
    system_message = '''You are an expert evaluator tasked with scoring the groundedness of a response.
Given a response and the context from which it was generated, score how well the response is supported by the context on a scale from 0.0 to 1.0.
- A score of 1.0 means the response is fully grounded and directly supported by the context.
- A score of 0.0 means the response contains no support from the context or includes hallucinations.
Be as objective as possible and only rely on the provided context.
Return only the score as a float between 0.0 and 1.0 — no explanations.'''

    groundedness_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Context: {context}\nResponse: {response}\n\nGroundedness score:")
    ])

    chain = groundedness_prompt | llm | StrOutputParser()
    groundedness_score = float(chain.invoke({
        "context": "\n".join([doc["content"] for doc in state['context']]),
        "response": state['response'] #
    }))
    print("groundedness_score: ", groundedness_score)
    state['groundedness_loop_count'] += 1
    print("#########Groundedness Incremented###########")
    state['groundedness_score'] = groundedness_score

    return state

#check precision
def check_precision(state: Dict) -> Dict:
    """
    Checks whether the response precisely addresses the user’s query.

    Args:
        state (Dict): The current state of the workflow, containing the query and response.

    Returns:
        Dict: The updated state with the precision score.
    """
    print("---------check_precision---------")
    system_message = '''You are a precise evaluator. Given a user query and a response, score how accurately and directly the response addresses the query.
Ignore any irrelevant information and focus only on whether the response answers the query fully and clearly.
Score precision on a scale from 0.0 (not at all precise) to 1.0 (perfectly precise).
Return only a single float value with no explanation.'''

    precision_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\nPrecision score:")
    ])

    chain = precision_prompt | llm | StrOutputParser()
    precision_score = float(chain.invoke({
        "query": state['query'],
        "response": state['response']
    }))
    state['precision_score'] = precision_score
    print("precision_score:", precision_score)
    state['precision_loop_count'] += 1
    print("#########Precision Incremented###########")
    return state
