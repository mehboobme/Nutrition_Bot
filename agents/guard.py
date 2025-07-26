#================================ Guardrails ===========================#
# Function to filter user input with Llama Guard
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq as Groq


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

llama_guard_client = Groq(api_key=groq_api_key, model="llama-guard-3-8b")

def filter_input_with_llama_guard(user_input):
    
    """
    Filters user input using Llama Guard to ensure it is safe.

    Parameters:
    - user_input: The input provided by the user.
    - model: The Llama Guard model to be used for filtering (default is "llama-guard-3-8b").

    Returns:
    - The filtered and safe input.
    """
    try:
        # Create a request to Llama Guard to filter the user input
        response = llama_guard_client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model=model,
        )
        # Return the filtered input
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Llama Guard: {e}")
        return None