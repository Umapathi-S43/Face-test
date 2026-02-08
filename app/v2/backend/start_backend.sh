#!/bin/bash
# PlasticVision Pro v2 — Backend Start Script
# Usage: ./start_backend.sh [--port 8000]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V2_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR"

echo "═══════════════════════════════════════════════"
echo "🎭 PlasticVision Pro v2 — GPU Backend"
echo "═══════════════════════════════════════════════"

# Try to find and activate a virtual environment
VENV_PATHS=(
    "$HOME/face-swap/facefusion/venv"
    "$V2_DIR/venv"
    "$HOME/.venv"
)

for venv in "${VENV_PATHS[@]}"; do
    if [ -f "$venv/bin/activate" ]; then
        echo "📦 Activating: $venv"
        source "$venv/bin/activate"
        break
    fi
done

# Navigate to backend directory
cd "$BACKEND_DIR"

echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(which python)"

# Start
exec python run_backend.py "$@"
