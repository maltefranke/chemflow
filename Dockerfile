# Base image with PyTorch + CUDA pre-installed
FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast pip installs at build and runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Copy full project so editable install can resolve deps + package structure.
# At runtime, live code is bind-mounted over /app via chemflow.toml,
# so you don't need to rebuild just because code changed.
# Only rebuild the image when pyproject.toml dependencies change.
COPY . /app/

RUN uv pip install --system -e .
RUN uv pip install --system --no-build-isolation torch-cluster

CMD ["python"]
