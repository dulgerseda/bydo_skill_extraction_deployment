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
#
# What stays outside (volume mount):
#   - config.json  → OpenAI API key (provided by client)
#   - jobs/        → uploaded JD files
#   - results/     → JSON outputs

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

# Create runtime folders (pipeline will write files here)
RUN mkdir -p jobs results answers normalized_answers tokens

# Expose port 8000 to the outside world
EXPOSE 8000

# Start FastAPI when the container runs
CMD ["uvicorn", "html_upload:app", "--host", "0.0.0.0", "--port", "8000"]