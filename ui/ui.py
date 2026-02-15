"""Streamlit UI for the Nutrition Disorder Specialist chatbot."""
import logging
import os
from typing import Optional

import streamlit as st

from services.bot import NutritionBot
from agents.guard import filter_input_with_llama_guard
from core.logging_config import setup_logging

# Configure logging on import
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safety classifications that are considered acceptable
SAFE_CLASSIFICATIONS = {"SAFE", "S6", "S7"}


def _is_safe_input(user_input: str) -> tuple[bool, Optional[str]]:
    """
    Check if user input passes safety filters.
    
    Args:
        user_input: The user's message to check.
        
    Returns:
        Tuple of (is_safe, classification_result)
    """
    # Check if guardrails are enabled
    if not os.getenv("GROQ_API_KEY"):
        logger.debug("Guardrails disabled - GROQ_API_KEY not set")
        return True, "SAFE"
    
    try:
        result = filter_input_with_llama_guard(user_input)
        if result is None:
            logger.warning("Guard returned None, allowing input")
            return True, "SAFE"
        
        normalized = result.replace("\n", " ").strip().upper()
        is_safe = any(safe in normalized for safe in SAFE_CLASSIFICATIONS)
        logger.debug(f"Safety check result: {normalized}, is_safe={is_safe}")
        return is_safe, normalized
    except Exception as e:
        logger.error(f"Safety check failed: {e}")
        return True, "SAFE"  # Fail open if guard errors


def _initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None


def _render_login():
    """Render the login form."""
    with st.form("login_form", clear_on_submit=True):
        user_id = st.text_input(
            "Please enter your name to begin:",
            placeholder="Your name"
        )
        submit_button = st.form_submit_button("Start Chat")
        
        if submit_button and user_id:
            user_id = user_id.strip()
            if user_id:
                st.session_state.user_id = user_id
                welcome_msg = f"Welcome, {user_id}! How can I help you with nutrition disorders today?"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": welcome_msg
                })
                logger.info(f"User logged in: {user_id}")
                st.rerun()


def _render_chat():
    """Render the chat interface."""
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_query = st.chat_input("Type your question here (or 'exit' to end)...")
    
    if user_query:
        _handle_user_message(user_query)


def _handle_user_message(user_query: str):
    """Process and respond to a user message."""
    # Handle exit command
    if user_query.lower().strip() == "exit":
        _handle_exit()
        return

    # Add user message to history and display
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Safety check
    is_safe, classification = _is_safe_input(user_query)
    
    if not is_safe:
        logger.warning(f"Unsafe input detected: {classification}")
        inappropriate_msg = (
            "I apologize, but I cannot process that input as it may be inappropriate. "
            "Please try again with a different question."
        )
        with st.chat_message("assistant"):
            st.write(inappropriate_msg)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": inappropriate_msg
        })
        return

    # Generate response
    _generate_response(user_query)


def _generate_response(user_query: str):
    """Generate and display bot response."""
    try:
        # Initialize chatbot if needed
        if st.session_state.chatbot is None:
            with st.spinner("Initializing assistant..."):
                st.session_state.chatbot = NutritionBot()

        # Generate response with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.handle_customer_query(
                    st.session_state.user_id, 
                    user_query
                )
            st.write(response)
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response
        })
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_msg = (
            "I'm sorry, I encountered an error while processing your query. "
            "Please try again or rephrase your question."
        )
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_msg
        })


def _handle_exit():
    """Handle user exit request."""
    st.session_state.chat_history.append({"role": "user", "content": "exit"})
    with st.chat_message("user"):
        st.write("exit")
    
    goodbye_msg = "Goodbye! Feel free to return if you have more questions about nutrition disorders."
    st.session_state.chat_history.append({"role": "assistant", "content": goodbye_msg})
    with st.chat_message("assistant"):
        st.write(goodbye_msg)
    
    logger.info(f"User {st.session_state.user_id} logged out")
    
    # Reset session
    st.session_state.user_id = None
    st.session_state.chatbot = None
    st.rerun()


def nutrition_disorder_streamlit():
    """Main Streamlit application for the Nutrition Disorder Specialist."""
    # Page configuration
    st.set_page_config(
        page_title="Nutrition Disorder Specialist",
        page_icon="🥗",
        layout="centered"
    )
    
    # Header
    st.title("🥗 Nutrition Disorder Specialist")
    st.markdown(
        "Ask me anything about nutrition disorders, symptoms, causes, treatments, and more. "
        "Type **'exit'** to end the conversation."
    )
    st.divider()
    
    # Initialize session state
    _initialize_session_state()
    
    # Render appropriate view
    if st.session_state.user_id is None:
        _render_login()
    else:
        # Show user info in sidebar
        with st.sidebar:
            st.markdown(f"**Logged in as:** {st.session_state.user_id}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.chat_history = []
                st.session_state.chatbot = None
                st.rerun()
        
        _render_chat()


if __name__ == "__main__":
    nutrition_disorder_streamlit()
