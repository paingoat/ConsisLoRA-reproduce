#!/usr/bin/env bash
# Generate `notebooks/consislora_runpod_batch.ipynb` from the Jupytext source
# `notebooks/consislora_runpod_batch.py` (percent format).
#
# Requires: pip install jupytext
# Run from anywhere: bash script/jupytext_to_ipynb.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v jupytext &>/dev/null; then
  echo "jupytext not found. Install with: pip install jupytext" >&2
  exit 1
fi

jupytext --to ipynb notebooks/consislora_runpod_batch.py -o notebooks/consislora_runpod_batch.ipynb
echo "Wrote notebooks/consislora_runpod_batch.ipynb"
