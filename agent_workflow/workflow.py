from langgraph.graph import StateGraph, START, END
from agents.agent_state import AgentState
from agents.agent_steps import expand_query, retrieve_context, craft_response
from core.evaluation import score_groundedness, check_precision
from core.refinement import refine_response, refine_query
from core.routing import should_continue_groundedness, should_continue_precision, max_iterations_reached

def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    workflow.add_node("expand_query", expand_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("craft_response", craft_response)
    workflow.add_node("score_groundedness", score_groundedness)
    workflow.add_node("refine_response", refine_response)
    workflow.add_node("check_precision", check_precision)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("max_iterations_reached", max_iterations_reached)

    workflow.add_edge(START, "expand_query")
    workflow.add_edge("expand_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "craft_response")
    workflow.add_edge("craft_response", "score_groundedness")

    workflow.add_conditional_edges("score_groundedness", should_continue_groundedness, {
        "check_precision": "check_precision",
        "refine_response": "refine_response",
        "max_iterations_reached": "max_iterations_reached"
    })

    workflow.add_edge("refine_response", "craft_response")

    workflow.add_conditional_edges("check_precision", should_continue_precision, {
        "pass": END,
        "refine_query": "refine_query",
        "max_iterations_reached": "max_iterations_reached"
    })

    workflow.add_edge("refine_query", "expand_query")
    workflow.add_edge("max_iterations_reached", END)

    return workflow
