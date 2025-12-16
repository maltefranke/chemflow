#!/bin/bash

# Exit on error
set -e


# first create the .env file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cat > .env <<EOL
export PROJECT_ROOT="${SCRIPT_DIR}"
export HYDRA_JOBS="${SCRIPT_DIR}"
export WANDB_DIR="${SCRIPT_DIR}"
EOL

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi


echo "Creating virtual environment..."
uv venv


# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install project in editable mode (without torch-scatter)
echo "Installing project in editable mode..."
uv pip install -e .

# Get torch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))" || echo "cpu")

# Determine the correct find-links URL
SEARCH_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}.html"

# Install torch-scatter with the correct version
echo "Installing torch-cluster for torch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}..."
uv pip install torch-cluster -f "$SEARCH_URL" 

echo "Environment setup complete! You can now activate it with:"
echo "source .venv/bin/activate" 