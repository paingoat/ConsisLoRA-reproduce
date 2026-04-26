#!/usr/bin/env bash
# One-time development / RunPod setup for ConsisLoRA (aligns with README + batch notebook).
#
# If `conda` is not on PATH, this script **automatically installs Miniconda** (batch `-b`, no
# interactive installer wizard) on **Linux x86_64 / aarch64** into ${HOME}/miniconda3 by default.
# Override install location:  MINICONDA_PREFIX=/opt/miniconda3 bash script/setup_runpod_env.sh
# On Windows or macOS, install Miniconda/Anaconda yourself, then re-run this script.
#
# - `conda create` is intentionally WITHOUT `-y` so you can confirm when conda prompts.
#
# Usage (from repository root is fine):
#   bash script/setup_runpod_env.sh
#
# Optional — persistent model cache (e.g. RunPod /workspace):
#   export HF_HOME=/workspace/huggingface
#   mkdir -p "$HF_HOME"

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MINICONDA_PREFIX="${MINICONDA_PREFIX:-${HOME}/miniconda3}"

install_miniconda_linux() {
  local prefix="$1"
  local url
  case "$(uname -m)" in
    x86_64)  url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
    aarch64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
    *) echo "Unsupported architecture: $(uname -m). Install Miniconda manually." >&2; return 1 ;;
  esac
  echo "==> conda not found — installing Miniconda -> ${prefix} (batch, -b)…"
  tmp="$(mktemp)"
  if command -v wget &>/dev/null; then
    wget -q "$url" -O "$tmp"
  elif command -v curl &>/dev/null; then
    curl -fsSL "$url" -o "$tmp"
  else
    echo "Need wget or curl to download Miniconda." >&2
    rm -f "$tmp"
    return 1
  fi
  bash "$tmp" -b -p "$prefix"
  rm -f "$tmp"
  echo "==> Miniconda installed at ${prefix}"
}

if command -v conda &>/dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ "$(uname -s)" = "Linux" ]; then
  if [ -x "${MINICONDA_PREFIX}/bin/conda" ]; then
    echo "==> Using existing Miniconda at ${MINICONDA_PREFIX}"
    export PATH="${MINICONDA_PREFIX}/bin:${PATH}"
    # shellcheck source=/dev/null
    source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
  else
    install_miniconda_linux "$MINICONDA_PREFIX"
    export PATH="${MINICONDA_PREFIX}/bin:${PATH}"
    # shellcheck source=/dev/null
    source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
  fi
else
  echo "conda not found, and auto-install of Miniconda is only supported on Linux." >&2
  echo "Install from https://docs.conda.io/en/latest/miniconda.html then re-run this script." >&2
  exit 1
fi

echo "==> Conda: create env (no -y — confirm when asked)"
echo "    conda create -n consislora python=3.11"
conda create -n consislora python=3.11

echo "==> Install Python deps into 'consislora'"
conda activate consislora
pip install -r requirements.txt
pip install jupytext ipykernel

echo "==> ipykernel (Jupyter / VS Code)"
python -m ipykernel install --user --name consislora --display-name "Python (consislora)"

echo "==> Accelerate: non-interactive default (single GPU, fp16; matches training scripts)"
mkdir -p "${HOME}/.cache/huggingface/accelerate"
cp "${ROOT}/script/accelerate_default_config.yaml" "${HOME}/.cache/huggingface/accelerate/default_config.yaml"

echo "==> Hugging Face upload (notebook section E): copy .env.example to .env and set HF_TOKEN"
echo "==> Optional HF cache on RunPod (uncomment in your profile if you use /workspace):
#    export HF_HOME=/workspace/huggingface
#    mkdir -p \"\$HF_HOME\"
"
echo "==> Build notebook from Jupytext source"
bash "${ROOT}/script/jupytext_to_ipynb.sh"

echo "Done. Activate later with: conda activate consislora"
