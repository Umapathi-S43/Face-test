#!/bin/bash
# PlasticVision Pro v2 — GPU Server Deploy Script
# Usage: bash deploy_gpu.sh
# NOTE: no 'set -e' — we handle errors explicitly so downloads don't kill the script

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

# Helper: download with retries using curl (more reliable than wget for HuggingFace)
download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"
    echo "  ⬇️  Downloading $name..."
    if curl -L --retry 3 --retry-delay 5 -# -o "$dest" "$url"; then
        local size=$(du -sh "$dest" 2>/dev/null | cut -f1)
        echo "  ✅ $name ready ($size)"
        return 0
    else
        echo "  ❌ Failed to download $name from $url"
        rm -f "$dest"
        return 1
    fi
}

# === Download models ===
echo ""
echo "📥 Downloading AI models (this takes a few minutes first time)..."

# buffalo_l (InsightFace face analysis)
BUFFALO_DIR="$MODELS_DIR/models/buffalo_l"
if [ ! -f "$BUFFALO_DIR/det_10g.onnx" ]; then
    BUFFALO_ZIP="$MODELS_DIR/models/buffalo_l.zip"
    if download_file \
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l.zip" \
        "$BUFFALO_ZIP" "buffalo_l face analysis model (~150MB)"; then
        mkdir -p "$BUFFALO_DIR"
        unzip -o -q "$BUFFALO_ZIP" -d "$MODELS_DIR/models/"
        # Handle case where zip extracts into buffalo_l/buffalo_l/
        if [ -d "$BUFFALO_DIR/buffalo_l" ]; then
            mv "$BUFFALO_DIR/buffalo_l/"* "$BUFFALO_DIR/" 2>/dev/null
            rmdir "$BUFFALO_DIR/buffalo_l" 2>/dev/null
        fi
        rm -f "$BUFFALO_ZIP"
    fi
    # Verify extraction worked
    if [ -f "$BUFFALO_DIR/det_10g.onnx" ]; then
        echo "  ✅ buffalo_l ready ($(ls $BUFFALO_DIR/*.onnx | wc -l | tr -d ' ') models)"
    else
        echo "  ⚠️  buffalo_l download/extract may have failed."
        echo "     Falling back to insightface auto-download on first run..."
        echo "     Files found: $(ls $BUFFALO_DIR/ 2>/dev/null || echo 'none')"
    fi
else
    echo "  ✅ buffalo_l exists"
fi

# inswapper_128.onnx (face swap model - FP32, 529MB)
if [ ! -f "$MODELS_DIR/inswapper_128.onnx" ]; then
    download_file \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx" \
        "$MODELS_DIR/inswapper_128.onnx" "inswapper_128.onnx (529MB)"
else
    echo "  ✅ inswapper_128.onnx exists ($(du -sh $MODELS_DIR/inswapper_128.onnx | cut -f1))"
fi

# inswapper_128_fp16.onnx (face swap model - FP16, 264MB — 2x faster on GPU)
if [ ! -f "$MODELS_DIR/inswapper_128_fp16.onnx" ]; then
    download_file \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx" \
        "$MODELS_DIR/inswapper_128_fp16.onnx" "inswapper_128_fp16.onnx (264MB)"
else
    echo "  ✅ inswapper_128_fp16.onnx exists ($(du -sh $MODELS_DIR/inswapper_128_fp16.onnx | cut -f1))"
fi

# GFPGANv1.4.pth (face enhancement, 332MB)
if [ ! -f "$MODELS_DIR/GFPGANv1.4.pth" ]; then
    download_file \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
        "$MODELS_DIR/GFPGANv1.4.pth" "GFPGANv1.4.pth (332MB)"
else
    echo "  ✅ GFPGANv1.4.pth exists ($(du -sh $MODELS_DIR/GFPGANv1.4.pth | cut -f1))"
fi

# GFPGAN auxiliary weights (detection + parsing)
if [ ! -f "$BACKEND_DIR/gfpgan/weights/detection_Resnet50_Final.pth" ]; then
    download_file \
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
        "$BACKEND_DIR/gfpgan/weights/detection_Resnet50_Final.pth" "GFPGAN detection weights"
else
    echo "  ✅ GFPGAN detection weights exist"
fi

if [ ! -f "$BACKEND_DIR/gfpgan/weights/parsing_parsenet.pth" ]; then
    download_file \
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" \
        "$BACKEND_DIR/gfpgan/weights/parsing_parsenet.pth" "GFPGAN parsing weights"
else
    echo "  ✅ GFPGAN parsing weights exist"
fi

# === Verify ===
echo ""
echo "✅ Verification:"
echo "  Backend:   $(ls $BACKEND_DIR/*.py 2>/dev/null | wc -l | tr -d ' ') Python files"
echo "  Models:    $(ls $MODELS_DIR/*.onnx $MODELS_DIR/*.pth 2>/dev/null | wc -l | tr -d ' ') model files"
echo "  Buffalo_l: $(ls $BUFFALO_DIR/*.onnx 2>/dev/null | wc -l | tr -d ' ') ONNX models"
echo "  GFPGAN:    $(ls $BACKEND_DIR/gfpgan/weights/*.pth 2>/dev/null | wc -l | tr -d ' ') weight files"
echo ""
echo "  Model dir contents:"
ls -lh "$MODELS_DIR/" 2>/dev/null
echo "  Buffalo_l contents:"
ls "$BUFFALO_DIR/" 2>/dev/null

# === Start server ===
echo ""
echo "═══════════════════════════════════════════════"
echo "🚀 Starting backend server..."
echo "═══════════════════════════════════════════════"
cd "$BACKEND_DIR"
python run_backend.py --skip-download --port 8000 --host 0.0.0.0
