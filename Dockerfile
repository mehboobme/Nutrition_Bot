%%writefile Dockerfile

# Base image with the correct Python version
FROM python:3.10-slim

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

# Start Streamlit app on port 7860 (for Spaces compatibility)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

# Reference: https://huggingface.co/docs/hub/en/spaces-sdks-docker