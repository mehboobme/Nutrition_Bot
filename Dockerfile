# Base image with Python 3.11
FROM python:3.11-slim

# Create a non-root user named 'user'
RUN useradd -m -u 1000 user

# Use the 'user' for subsequent instructions
USER user

# Add user-specific bin directory to PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy and install dependencies as the 'user'
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the app code with correct ownership
COPY --chown=user . /app

# Expose the Streamlit port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Start Streamlit app on port 7860 (for Spaces compatibility)
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]