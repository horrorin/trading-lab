# ==============================================================================
# Trading Lab — Research / Quant Development Container
# "Dev is Prod" GPU parity using official PyTorch CUDA runtime image
# ==============================================================================

FROM docker.io/pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------
# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Improve Python behavior inside containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Speed up JupyterLab startup in container environments
ENV JUPYTERLAB_DISABLE_DEV_BUILD=True

# ------------------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------------------
# - build-essential: required for compiling native extensions (e.g., some Python packages)
# - git: useful for installing packages from VCS or working inside the container
# - libta-lib*: required to build/install the Python TA-Lib wrapper reliably
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# TA-Lib from conda-forge (preferred: avoids autotools build issues)
# ------------------------------------------------------------------------------
RUN conda install -y -c conda-forge ta-lib && conda clean -a -y

# ------------------------------------------------------------------------------
# Non-root user setup (best practice for security and reproducibility)
# ------------------------------------------------------------------------------
# Create a dedicated user for research workloads
RUN useradd -m -u 1000 quantuser
USER quantuser

# Set working directory inside the container
WORKDIR /home/quantuser/trading-lab

# Ensure local user-installed Python packages are available on PATH
ENV PATH="/home/quantuser/.local/bin:${PATH}"

# ------------------------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------------------------
# Copy requirements first to leverage Docker layer caching
COPY --chown=quantuser:quantuser requirements.txt .

# Upgrade pip and install dependencies using the SAME Python shipped with PyTorch
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY --chown=quantuser:quantuser . .

# Default entrypoint: interactive Python REPL
# (JupyterLab is launched via Makefile instead)
CMD ["python"]
