# Use an official Python base image
FROM python:3.11-slim

RUN groupmod -g 1000 www-data
RUN usermod -u 1000 -g 1000 www-data

# Variables d'environnement pour éviter les problèmes de permissions
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /api

# Install Poetry
RUN pip install --upgrade pip && pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock* /api/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi  --no-root

# Copy API code
COPY api/ /api/

# Copy detector code
COPY detector/ /detector/

# Copy model directory
COPY model/ /model/

# Pre-download YOLO model and create cache directories with proper permissions
RUN mkdir -p /tmp/ultralytics /tmp/huggingface /tmp/torch && \
    chmod 777 /tmp/ultralytics /tmp/huggingface /tmp/torch && \
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" && \
    chmod -R 777 /tmp/ultralytics

# Expose port
EXPOSE 8000

USER www-data

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]