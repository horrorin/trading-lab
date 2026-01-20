# ==============================================================================
# Trading Lab — Podman / DevOps Makefile
# ==============================================================================
# This Makefile standardizes container build, execution, and Jupyter workflows
# with GPU support and reproducible development practices.
# ==============================================================================

# ------------------------------------------------------------------------------
# Project Variables
# ------------------------------------------------------------------------------
IMAGE_NAME      := trading-lab
CONTAINER_NAME  := trading-lab
BUILD_STAMP     := .build_trading_lab.stamp
HF_CACHE_VOLUME := hf_cache_trading
PODMAN          := podman
CUDA_CHECK_IMG  := docker.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

# ------------------------------------------------------------------------------
# Host User Information (for permission consistency)
# ------------------------------------------------------------------------------
USER_ID  := $(shell id -u)
GROUP_ID := $(shell id -g)
USERNAME := $(shell whoami)

# ------------------------------------------------------------------------------
# Default Podman Run Flags (Shared Across Targets)
# ------------------------------------------------------------------------------
# - --rm: remove container after exit
# - --interactive / --tty: allow interactive shell
# - --userns=keep-id: keep host UID/GID mapping to avoid permission issues
# - --device nvidia.com/gpu=all: enable full GPU access
# - --security-opt=label=disable: avoid SELinux/labeling issues on some systems
# - Volume mounts:
#   * Persistent Hugging Face cache
#   * Bind mount of current project directory
# ------------------------------------------------------------------------------
PODMAN_RUN_FLAGS := \
    --rm \
    --interactive \
    --tty \
    --name $(CONTAINER_NAME) \
    --userns=keep-id \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --volume $(HF_CACHE_VOLUME):/home/quantuser/.cache/huggingface:Z \
    --volume $(CURDIR):/home/quantuser/trading-lab:rw,Z

# ==============================================================================
# Pre-requisite Checks
# ==============================================================================

check-podman:
	@command -v $(PODMAN) >/dev/null || (echo "Error: podman is not installed" && exit 1)
	@$(PODMAN) info --format '{{.Version.Version}}' >/dev/null 2>&1 || \
		(echo "Error: podman is not functioning correctly" && exit 1)

check-nvidia:
	@$(PODMAN) run --rm --device nvidia.com/gpu=all $(CUDA_CHECK_IMG) nvidia-smi >/dev/null 2>&1 || \
		(echo "Error: NVIDIA GPU not accessible in Podman." && exit 1)

.PHONY: check-podman check-nvidia

# ==============================================================================
# Build Logic (Stamp-Based Caching)
# ==============================================================================

# Rebuild only if Containerfile or requirements.txt change
$(BUILD_STAMP): Containerfile requirements.txt
	@echo "Detected changes in Containerfile or requirements.txt. Rebuilding image..."
	$(PODMAN) build \
		--tag $(IMAGE_NAME) \
		--label com.trading.lab.type=research \
		.

	@# Clean dangling images related to this research project label
	$(PODMAN) image prune -f --filter "label=com.trading.lab.type=research"

	touch $@

# Standard build target
build: check-podman $(BUILD_STAMP) ## Build image if dependencies changed

# Force full rebuild
rebuild: ## Force a complete rebuild from scratch
	rm -f $(BUILD_STAMP)
	$(MAKE) build

.PHONY: build rebuild

# ==============================================================================
# Operational Commands
# ==============================================================================

# Interactive shell inside the container
shell: build check-podman check-nvidia ## Open an interactive shell
	$(PODMAN) run $(PODMAN_RUN_FLAGS) $(IMAGE_NAME) /bin/bash

# Launch JupyterLab for research
notebook: build check-podman check-nvidia ## Start JupyterLab (localhost:8888)
	@echo "Starting JupyterLab (http://localhost:8888)"
	@mkdir -p notebooks
	@$(PODMAN) run $(PODMAN_RUN_FLAGS) -p 8888:8888 $(IMAGE_NAME) \
		jupyter lab \
		--ip=0.0.0.0 \
		--no-browser \
		--NotebookApp.token='' \
		--NotebookApp.password='' \
		--notebook-dir=notebooks

# Display environment and project status
info: ## Show project and environment info
	@echo "--- Project Info ---"
	@echo "User: $(USERNAME) (UID=$(USER_ID), GID=$(GROUP_ID))"
	@echo "Image: $(IMAGE_NAME)"
	@echo "Stamp: $(BUILD_STAMP)"
	@$(PODMAN) volume inspect $(HF_CACHE_VOLUME) >/dev/null 2>&1 && \
		echo "HF Cache: Mounted" || echo "HF Cache: Missing (will be created)"

# Stop running container if exists
stop: ## Stop and remove active container
	$(PODMAN) stop $(CONTAINER_NAME) || true

# Clean workspace artifacts
clean: ## Remove build stamp and Python caches
	rm -f $(BUILD_STAMP)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Workspace cleaned."

# Auto-generate help output
help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

.PHONY: shell notebook info stop clean help
