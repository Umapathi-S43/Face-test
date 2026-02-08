#!/bin/bash
# PlasticVision Pro v2 — GPU Server Deploy Script
# Usage: bash deploy_gpu.sh
set -e

echo "═══════════════════════════════════════════════"
echo "🎭 PlasticVision Pro v2 — GPU Deployment"
echo "═══════════════════════════════════════════════"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
MODELS_DIR="$PROJECT_DIR/models"

# === Check GPU ===
echo ""
echo "🔍 Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  No nvidia-smi found. CPU mode will be used."
fi

# === Create directories ===
echo ""
echo "📁 Creating directories..."
mkdir -p "$MODELS_DIR/models"
mkdir -p "$BACKEND_DIR/_uploads"
mkdir -p "$BACKEND_DIR/_outputs"
mkdir -p "$BACKEND_DIR/gfpgan/weights"

# === Python venv ===
echo ""
echo "🐍 Setting up Python environment..."
if [ ! -d "$PROJECT_DIR/venv" ]; then
    python3 -m venv "$PROJECT_DIR/venv"
    echo "  ✅ Created venv"
else
    echo "  ✅ venv exists"
fi
source "$PROJECT_DIR/venv/bin/activate"

# === Install dependencies ===
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "  ✅ Dependencies installed"

# === Download models ===
echo ""
echo "📥 Downloading AI models (this takes a few minutes first time)..."

# buffalo_l (InsightFace face analysis)
BUFFALO_DIR="$MODELS_DIR/models/buffalo_l"
if [ ! -d "$BUFFALO_DIR" ] || [ -z "$(ls -A $BUFFALO_DIR 2>/dev/null)" ]; then
    echo "  ⬇️  Downloading buffalo_l face analysis model..."
    BUFFALO_ZIP="$MODELS_DIR/models/buffalo_l.zip"
    wget -q --show-progress -O "$BUFFALO_ZIP" \
        "https://huggingface.co/datasets/deepinsight/insightface/resolve/main/models/buffalo_l.zip"
    mkdir -p "$BUFFALO_DIR"
    unzip -o -q "$BUFFALO_ZIP" -d "$MODELS_DIR/models/"
    rm -f "$BUFFALO_ZIP"
    echo "  ✅ buffalo_l ready"
else
    echo "  ✅ buffalo_l exists"
fi

# inswapper_128.onnx (face swap model - FP32, 529MB)
if [ ! -f "$MODELS_DIR/inswapper_128.onnx" ]; then
    echo "  ⬇️  Downloading inswapper_128.onnx (529MB)..."
    wget -q --show-progress -O "$MODELS_DIR/inswapper_128.onnx" \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
    echo "  ✅ inswapper_128.onnx ready"
else
    echo "  ✅ inswapper_128.onnx exists ($(du -sh $MODELS_DIR/inswapper_128.onnx | cut -f1))"
fi

# inswapper_128_fp16.onnx (face swap model - FP16, 264MB — 2x faster on GPU)
if [ ! -f "$MODELS_DIR/inswapper_128_fp16.onnx" ]; then
    echo "  ⬇️  Downloading inswapper_128_fp16.onnx (264MB)..."
    wget -q --show-progress -O "$MODELS_DIR/inswapper_128_fp16.onnx" \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
    echo "  ✅ inswapper_128_fp16.onnx ready"
else
    echo "  ✅ inswapper_128_fp16.onnx exists ($(du -sh $MODELS_DIR/inswapper_128_fp16.onnx | cut -f1))"
fi

# GFPGANv1.4.pth (face enhancement, 332MB)
if [ ! -f "$MODELS_DIR/GFPGANv1.4.pth" ]; then
    echo "  ⬇️  Downloading GFPGANv1.4.pth (332MB)..."
    wget -q --show-progress -O "$MODELS_DIR/GFPGANv1.4.pth" \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    echo "  ✅ GFPGANv1.4.pth ready"
else
    echo "  ✅ GFPGANv1.4.pth exists ($(du -sh $MODELS_DIR/GFPGANv1.4.pth | cut -f1))"
fi

# GFPGAN auxiliary weights (detection + parsing)
if [ ! -f "$BACKEND_DIR/gfpgan/weights/detection_Resnet50_Final.pth" ]; then
    echo "  ⬇️  Downloading GFPGAN detection weights..."
    wget -q --show-progress -O "$BACKEND_DIR/gfpgan/weights/detection_Resnet50_Final.pth" \
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    echo "  ✅ detection_Resnet50_Final.pth ready"
else
    echo "  ✅ GFPGAN detection weights exist"
fi

if [ ! -f "$BACKEND_DIR/gfpgan/weights/parsing_parsenet.pth" ]; then
    echo "  ⬇️  Downloading GFPGAN parsing weights..."
    wget -q --show-progress -O "$BACKEND_DIR/gfpgan/weights/parsing_parsenet.pth" \
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
    echo "  ✅ parsing_parsenet.pth ready"
else
    echo "  ✅ GFPGAN parsing weights exist"
fi

# === Create symlinks ===
echo ""
echo "🔗 Creating model symlinks..."
ln -sf "$MODELS_DIR/inswapper_128.onnx" "$BACKEND_DIR/inswapper_128.onnx"
ln -sf "$MODELS_DIR/inswapper_128_fp16.onnx" "$BACKEND_DIR/inswapper_128_fp16.onnx"
ln -sf "$MODELS_DIR/GFPGANv1.4.pth" "$BACKEND_DIR/GFPGANv1.4.pth"
ln -sf "$MODELS_DIR/models/buffalo_l" "$BACKEND_DIR/models/models/buffalo_l"
echo "  ✅ Symlinks created"

# === Verify ===
echo ""
echo "✅ Verification:"
echo "  Backend:   $(ls $BACKEND_DIR/*.py | wc -l | tr -d ' ') Python files"
echo "  Models:    $(ls $MODELS_DIR/*.onnx $MODELS_DIR/*.pth 2>/dev/null | wc -l | tr -d ' ') model files"
echo "  Buffalo_l: $(ls $BUFFALO_DIR/*.onnx 2>/dev/null | wc -l | tr -d ' ') ONNX models"
echo "  GFPGAN:    $(ls $BACKEND_DIR/gfpgan/weights/*.pth 2>/dev/null | wc -l | tr -d ' ') weight files"

# === Start server ===
echo ""
echo "═══════════════════════════════════════════════"
echo "🚀 Starting backend server..."
echo "═══════════════════════════════════════════════"
cd "$BACKEND_DIR"
python run_backend.py --skip-download --port 8000 --host 0.0.0.0
