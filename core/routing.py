"""Routing logic for the RAG workflow conditional edges."""
import logging
from typing import Dict, Any

from agents.agent_state import AgentState
from core.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


def should_continue_groundedness(state: Dict[str, Any]) -> str:
    """
    Decide if groundedness is sufficient or needs improvement.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name: 'check_precision', 'refine_response', or 'max_iterations_reached'
    """
    groundedness_score = state['groundedness_score']
    loop_count = state['groundedness_loop_count']
    threshold = config.groundedness_threshold
    max_iterations = config.max_refinement_iterations
    
    logger.debug(f"Groundedness check: score={groundedness_score:.2f}, threshold={threshold}, iteration={loop_count}")
    
    if groundedness_score >= threshold:
        logger.info(f"Groundedness passed ({groundedness_score:.2f} >= {threshold}), proceeding to precision check")
        return "check_precision"
    elif loop_count >= max_iterations:
        logger.warning(f"Max groundedness iterations reached ({loop_count})")
        return "max_iterations_reached"
    else:
        logger.info(f"Groundedness below threshold ({groundedness_score:.2f} < {threshold}), refining response")
        return "refine_response"


def should_continue_precision(state: Dict[str, Any]) -> str:
    """
    Decide if precision is sufficient or needs improvement.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name: 'pass', 'refine_query', or 'max_iterations_reached'
    """
    precision_score = state['precision_score']
    loop_count = state['precision_loop_count']
    threshold = config.precision_threshold
    max_iterations = config.max_refinement_iterations
    
    logger.debug(f"Precision check: score={precision_score:.2f}, threshold={threshold}, iteration={loop_count}")
    
    if precision_score >= threshold:
        logger.info(f"Precision passed ({precision_score:.2f} >= {threshold}), workflow complete")
        return "pass"
    elif loop_count > max_iterations:
        logger.warning(f"Max precision iterations reached ({loop_count})")
        return "max_iterations_reached"
    else:
        logger.info(f"Precision below threshold ({precision_score:.2f} < {threshold}), refining query")
        return "refine_query"


def max_iterations_reached(state: AgentState) -> AgentState:
    """Handle the case where max iterations are reached."""
    logger.warning("Max iterations reached - returning fallback response")
    state['response'] = (
        "I apologize, but I need more context to provide an accurate answer. "
        "Could you please provide more details or rephrase your question?"
    )
    return state
