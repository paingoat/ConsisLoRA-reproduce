#!/usr/bin/env bash
# One-time development / RunPod setup for ConsisLoRA (aligns with README + batch notebook).
#
# - `conda create` is intentionally WITHOUT `-y` so you can confirm when conda prompts.
# - Assumes `conda` is on PATH. Typical RunPod: `source ~/miniconda3/etc/profile.d/conda.sh`
#   (path may differ).
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

if ! command -v conda &>/dev/null; then
  echo "conda not found on PATH. Install Miniconda/Anaconda, then re-run this script." >&2
  exit 1
fi
# Load conda (required for `conda activate` in a non-interactive script)
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

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
