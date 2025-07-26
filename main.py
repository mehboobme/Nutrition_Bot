# main.py
import traceback
from ui.ui import nutrition_disorder_streamlit
from agent_workflow.workflow import create_workflow

if __name__ == "__main__":
    try:
        # Initialize the workflow once and pass it if needed
        app_workflow = create_workflow().compile()
        
        # Pass the workflow to Streamlit app if it expects it
        nutrition_disorder_streamlit()
        
    except Exception as e:
        print("Exception caught in main:")
        traceback.print_exc()
        raise
