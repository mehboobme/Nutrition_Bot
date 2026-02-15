"""Main entry point for the Nutrition Disorder Specialist application."""
import sys
import logging

from core.logging_config import setup_logging

# Setup logging first
logger = setup_logging(level=logging.INFO)


def main():
    """Run the Streamlit application."""
    try:
        logger.info("Starting Nutrition Disorder Specialist application...")
        
        # Import here to ensure logging is configured first
        from ui.ui import nutrition_disorder_streamlit
        
        # Run the Streamlit app
        nutrition_disorder_streamlit()
        
    except Exception as e:
        logger.exception(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
