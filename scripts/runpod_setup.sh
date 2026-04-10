#!/bin/bash
# =============================================================================
# RunPod Setup Script for social-memory
#
# Prerequisites: Set these as RunPod environment variables (secrets):
#   - OPENAI_API_KEY
#   - HF_TOKEN
#   - ANTHROPIC_API_KEY
#   - GH_TOKEN  (GitHub personal access token for cloning private repos)
# =============================================================================

set -e

REPO_URL="https://github.com/tomcohen13/social-memory.git"
BRANCH="sheer/huggingface-model-loading"
WORKSPACE="/workspace/social-memory"

# =============================================================================
# Cache symlinks (persist across pod restarts)
# =============================================================================

for dir in huggingface pip; do
    if [ -d "/workspace/.cache/$dir" ] && [ ! -L "/root/.cache/$dir" ]; then
        rm -rf "/root/.cache/$dir" 2>/dev/null
        mkdir -p /root/.cache
        ln -s "/workspace/.cache/$dir" "/root/.cache/$dir"
        echo "Restored $dir cache symlink"
    fi
done

# =============================================================================
# Git + SSH
# =============================================================================

git config --global user.name "antbaez"
git config --global user.email "acbaez01@gmail.com"

# Use GH_TOKEN for HTTPS cloning (no SSH key needed)
if [ -n "$GH_TOKEN" ]; then
    git config --global credential.helper store
    echo "https://antbaez:${GH_TOKEN}@github.com" > ~/.git-credentials
    echo "Git credentials configured"
else
    echo "WARNING: GH_TOKEN not set — clone may fail for private repos"
fi

# =============================================================================
# Clone repo
# =============================================================================

if [ ! -d "$WORKSPACE" ]; then
    echo "Cloning $REPO_URL ($BRANCH)..."
    git clone --branch "$BRANCH" "$REPO_URL" "$WORKSPACE"
else
    echo "Repo already exists, pulling latest..."
    cd "$WORKSPACE" && git pull origin "$BRANCH"
fi

cd "$WORKSPACE"

# =============================================================================
# .env file (from RunPod environment variables)
# =============================================================================

echo "Creating .env file..."
cat > .env <<EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
HF_TOKEN=${HF_TOKEN}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
EOF
echo ".env file created"

# =============================================================================
# Python environment + dependencies
# =============================================================================

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Creating venv and installing dependencies..."
uv venv .venv --python 3.12
uv pip install -e "." --python .venv/bin/python

echo "Dependencies installed"

# =============================================================================
# VS Code extensions
# =============================================================================

CODE=$(find /root/.vscode-server -name "code" 2>/dev/null | head -1)

if [ -n "$CODE" ]; then
    extensions=(
        "anthropic.claude-code"
        "ms-python.python"
        "ms-python.vscode-pylance"
        "ms-python.vscode-python-envs"
        "ms-python.debugpy"
    )

    echo "Installing VS Code extensions..."
    for ext in "${extensions[@]}"; do
        "$CODE" --install-extension "$ext"
    done
    echo "Extensions installed"
else
    echo "VS Code server not found — skipping extensions"
fi

# =============================================================================
# Claude Code
# =============================================================================

curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo ""
echo "============================================"
echo "Setup complete!"
echo "  Repo: $WORKSPACE"
echo "  Run:  source ~/.bashrc && cd $WORKSPACE"
echo "============================================"
