FROM python:3.11-slim

# Install system dependencies (ffmpeg is required for audio extraction)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all project files and install dependencies from pyproject.toml
COPY . .
RUN pip install --no-cache-dir .

# Create data directories
RUN mkdir -p data/audio data/transcripts data/chroma

EXPOSE 8000 8501

# Default command: run the Streamlit UI app
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
