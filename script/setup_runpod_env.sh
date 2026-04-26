#!/usr/bin/env bash
# One-time development / RunPod setup for ConsisLoRA (aligns with README + batch notebook).
#
# Miniconda is NOT installed by default — the script expects `conda` on PATH (many RunPod templates
# already ship conda, or you install Miniconda/Anaconda yourself first).
#
# Optional: auto-install Miniconda (batch, no interactive installer prompts) into $HOME/miniconda3:
#   INSTALL_MINICONDA=1 bash script/setup_runpod_env.sh
#   # or a custom prefix:
#   INSTALL_MINICONDA=1 MINICONDA_PREFIX=/opt/miniconda3 bash script/setup_runpod_env.sh
# (Linux x86_64 / aarch64 only; on Windows/macOS install Miniconda manually.)
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
  echo "==> Downloading Miniconda -> ${prefix} (batch install, -b)…"
  tmp="$(mktemp)"
  wget -q "$url" -O "$tmp"
  bash "$tmp" -b -p "$prefix"
  rm -f "$tmp"
  echo "==> Miniconda installed. Add to your shell: export PATH=\"${prefix}/bin:\$PATH\" or source conda.sh"
}

if ! command -v conda &>/dev/null; then
  if [ "${INSTALL_MINICONDA:-0}" = "1" ] && [ "$(uname -s)" = "Linux" ]; then
    if [ -x "${MINICONDA_PREFIX}/bin/conda" ]; then
      echo "==> Using existing Miniconda at ${MINICONDA_PREFIX}"
      # shellcheck source=/dev/null
      source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
    else
      install_miniconda_linux "$MINICONDA_PREFIX"
      export PATH="${MINICONDA_PREFIX}/bin:${PATH}"
      # shellcheck source=/dev/null
      source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
    fi
  else
    echo "conda not found on PATH. Options:" >&2
    echo "  - Install Miniconda/Anaconda: https://docs.conda.io/en/latest/miniconda.html" >&2
    echo "  - On Linux, re-run with:  INSTALL_MINICONDA=1 bash script/setup_runpod_env.sh" >&2
    exit 1
  fi
else
  # Load conda (required for `conda activate` in a non-interactive script)
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
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
