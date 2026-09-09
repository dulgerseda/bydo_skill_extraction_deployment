# BYDO Skill Extraction — Dockerfile
# ----------------------------------------
# Build:  docker build -t bydo .
# Run:    docker compose up --build
#
# What goes into the image:
#   - Python 3.11 + all pip packages
#   - FastAPI app (html_upload.py)
#   - Pipeline scripts (extract, normalize, rerank)
#   - ESCO data files (~27 MB)
#   - Prompt files
#   - Offline regression tests
#
# What stays outside:
#   - OPENAI_API_KEY → environment variable, never baked in or mounted
#   - config.json  → model and sampling settings (volume mount)
#   - jobs/        → uploaded JD files
#   - results/     → JSON outputs
#   - cache/       → extraction cache, keyed by document hash

# Base image — pull Python 3.11 (slim = lightweight)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first, then install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files and folders into the image
COPY html_upload.py .
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY prompts/ ./prompts/
COPY tests/ ./tests/

# Create runtime folders (pipeline will write files here)
RUN mkdir -p jobs results answers normalized_answers tokens cache/extractions

# Expose port 8000 to the outside world
EXPOSE 8000

# Start FastAPI when the container runs
CMD ["uvicorn", "html_upload:app", "--host", "0.0.0.0", "--port", "8000"]