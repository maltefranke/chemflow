# 1. Start from the PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PROJECT_ROOT=/app \
    HYDRA_JOBS=/app \
    WANDB_DIR=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ... (previous setup)

# Copy uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Tell uv it has permission to write to the system Python
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Copy your project files
COPY . /app/

# Use --system so uv knows to target the global environment
RUN uv pip install --system -e .

RUN uv pip install --system --no-build-isolation torch-cluster

CMD ["python"]