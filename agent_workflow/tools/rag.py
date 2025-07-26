# agent_workflow/tools/rag.py

from langchain.tools import tool
from agent_workflow.workflow import create_workflow

WORKFLOW_APP = create_workflow().compile()

@tool
def agentic_rag(query: str):
    """
    RAG tool to handle nutrition disorder queries using the compiled LangGraph workflow.
    """
    inputs = {
        "query": query,
        "expanded_query": "",
        "context": [],
        "response": "",
        "precision_score": 0.0,
        "groundedness_score": 0.0,
        "groundedness_loop_count": 0,
        "precision_loop_count": 0,
        "feedback": "",
        "query_feedback": "",
        "loop_max_iter": 3
    }
    output = WORKFLOW_APP.invoke(inputs)
    return output
