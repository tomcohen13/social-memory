#!/usr/bin/env bash
# Sets up an isolated uv virtualenv for VideoLLaMA2
# Usage: bash scripts/setup_videollama2_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-videollama2"
VL2_DIR="$REPO_ROOT/vendor/VideoLLaMA2"

echo "==> Creating isolated venv at $VENV_DIR"
uv venv "$VENV_DIR" --python 3.12

echo "==> Cloning VideoLLaMA2 (if not already present)"
mkdir -p "$REPO_ROOT/vendor"
if [ ! -d "$VL2_DIR" ]; then
    git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2 "$VL2_DIR"
else
    echo "    Already cloned, skipping."
fi

echo "==> Installing VideoLLaMA2 into isolated venv"
uv pip install --python "$VENV_DIR/bin/python" -e "$VL2_DIR"

echo "==> Installing flash-attn (no build isolation, may take a few minutes)"
uv pip install --python "$VENV_DIR/bin/python" \
    flash-attn==2.5.8 --no-build-isolation

echo "==> Installing social-memory into isolated venv (for pipeline imports)"
uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_ROOT"

echo ""
echo "Done. To run a VideoLLaMA2 pipeline:"
echo "  source $VENV_DIR/bin/activate"
echo "  python scripts/run_pipeline.py --pipeline video --model videollama2:DAMO-NLP-SG/VideoLLaMA2.1-7B-AV --split demo"
