# For Python preffer bookworm over alpine due to compatibility issues
FROM python:3.11-bookworm

# define env. vars - gradio vars are recommended by vendor; do not create pycache / cache for modules + send output immediately without buffering
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \

WORKDIR /app

# Force update
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && pip install -r /app/requirements.txt

# Copy application code
COPY . /app

# create data directory and make sure that exists
RUN mkdir -p /app/data/chroma

# expose port for Gradio UI
EXPOSE 7860

# run application
CMD ["python", "main.py"]