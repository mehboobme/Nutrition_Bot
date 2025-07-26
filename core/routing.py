from typing import Dict, List, Any
from agents.agent_state import AgentState

def should_continue_groundedness(state):
  """Decides if groundedness is sufficient or needs improvement."""
  print("---------should_continue_groundedness---------")
  print("groundedness loop count: ", state['groundedness_loop_count'])
  if state['groundedness_score'] >=0.7:  # Threshold for groundedness
      print("Moving to precision")
      return "check_precision"
  else:
      if state['groundedness_loop_count'] >=3:
        return "max_iterations_reached"
      else:
        print(f"---------Groundedness Score Threshold Not met. Refining Response-----------")
        return "refine_response"

def should_continue_precision(state: Dict) -> str:
    """Decides if precision is sufficient or needs improvement."""
    print("---------should_continue_precision---------")
    print("precision loop count: ", state['precision_loop_count'])
    if state['precision_score'] >=0.7:  # Threshold for precision
        return "pass"  # Complete the workflow
    else:
        if state['precision_loop_count'] >3:  # Maximum allowed loops
            return "max_iterations_reached"
        else:
            print(f"---------Precision Score Threshold Not met. Refining Query-----------")  # Debugging
            return "refine_query"  # Refine the query

def max_iterations_reached(state: AgentState) -> AgentState:
    """Handles the case where max iterations are reached."""
    state['response'] = "We need more context to provide an accurate answer."
    return state
