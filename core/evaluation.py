"""Evaluation module for scoring RAG responses on groundedness and precision."""
import re
import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.agent_state import AgentState
from core.config import llm

logger = logging.getLogger(__name__)


def _parse_score(response: str) -> float:
    """
    Safely parse a score from LLM response.
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        Float score between 0.0 and 1.0
    """
    try:
        # Try direct float conversion
        score = float(response.strip())
    except ValueError:
        # Extract first decimal number from response
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
        else:
            logger.warning(f"Could not parse score from: {response}, defaulting to 0.5")
            score = 0.5
    
    # Clamp to valid range
    return max(0.0, min(1.0, score))


def score_groundedness(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check whether the response is grounded in the retrieved context.

    Args:
        state: Current workflow state containing response and context.

    Returns:
        Updated state with groundedness score.
    """
    logger.info("Evaluating groundedness...")
    
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
    
    context_text = "\n".join([doc["content"] for doc in state['context']])
    response_text = str(state['response'])
    
    raw_score = chain.invoke({
        "context": context_text,
        "response": response_text
    })
    
    groundedness_score = _parse_score(raw_score)
    
    state['groundedness_loop_count'] += 1
    state['groundedness_score'] = groundedness_score
    
    logger.info(f"Groundedness score: {groundedness_score:.2f} (iteration {state['groundedness_loop_count']})")
    return state


def check_precision(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check whether the response precisely addresses the user's query.

    Args:
        state: Current workflow state containing query and response.

    Returns:
        Updated state with precision score.
    """
    logger.info("Evaluating precision...")
    
    system_message = '''You are a precise evaluator. Given a user query and a response, score how accurately and directly the response addresses the query.
Ignore any irrelevant information and focus only on whether the response answers the query fully and clearly.
Score precision on a scale from 0.0 (not at all precise) to 1.0 (perfectly precise).
Return only a single float value with no explanation.'''

    precision_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\nPrecision score:")
    ])

    chain = precision_prompt | llm | StrOutputParser()
    
    raw_score = chain.invoke({
        "query": state['query'],
        "response": str(state['response'])
    })
    
    precision_score = _parse_score(raw_score)
    
    state['precision_score'] = precision_score
    state['precision_loop_count'] += 1
    
    logger.info(f"Precision score: {precision_score:.2f} (iteration {state['precision_loop_count']})")
    return state
