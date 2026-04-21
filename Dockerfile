FROM python:3.11-slim

# Install system dependencies (ffmpeg is required for audio extraction)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/audio data/transcripts data/chroma

EXPOSE 8000 8501
