FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/data

# Create data directories
RUN mkdir -p /data/education_resources \
    /data/scenarios \
    /data/evaluation \
    /data/collection

CMD ["python", "src/data/collection/main.py"] 