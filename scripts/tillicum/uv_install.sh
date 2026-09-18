#!/usr/bin/env bash
# Install uv with the official installer when it is missing, then `uv sync --frozen` the synced
# checkout.  Usage: uv_install.sh <scratch> <repo>   (idempotent; COMMANDS["uv_sync"])
#
# Caches and managed Pythons live under <scratch> because Tillicum home directories are 10 GB.
# NOTE: uv.lock pins the CPU torch wheel on Linux (pyproject [tool.uv.sources]); GPU training on
# the cluster needs a CUDA torch installed on top (see docs/CLUSTER.md, "torch on the cluster").
set -euo pipefail
scratch="${1:?usage: uv_install.sh <scratch> <repo>}"
repo="${2:?usage: uv_install.sh <scratch> <repo>}"
scratch="${scratch%/}"
export UV_CACHE_DIR="$scratch/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$scratch/.uv-python"
export PATH="$HOME/.local/bin:$scratch/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv: not found; installing with https://astral.sh/uv/install.sh"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
cd "$repo"
uv sync --frozen
echo "uv sync ok: $repo/.venv"
