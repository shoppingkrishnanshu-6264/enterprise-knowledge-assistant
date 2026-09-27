FROM python:3.11-slim

# libmagic1 is required by `unstructured` for file-type detection.
# build-essential covers any dependency that needs to compile from source
# (e.g. some sentence-transformers / tokenizers wheels on certain platforms).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libmagic1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, src/, data/, and chroma_db/ are committed to the repo, so the
# image always ships with whatever was last ingested — no separate data-sync
# step is needed for the GKE warm-standby deploy. See .dockerignore for what
# is deliberately left out (venv, debug scripts, git metadata, etc.).
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
