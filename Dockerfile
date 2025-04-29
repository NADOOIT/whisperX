# Dockerfile for reproducible WhisperX experiments
FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn[standard] jinja2 fastapi tk

# Copy entire repository
COPY . .

# Default command: run experiments
CMD ["bash", "agent_research_paper/run_experiments.sh"]
