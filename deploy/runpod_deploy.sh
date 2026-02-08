#!/bin/bash
# =============================================================================
# RunPod Deployment Script for Face-Swap Application
# RTX 5090 x2 GPU Instance
# =============================================================================

set -e

echo "=============================================="
echo "🚀 Face-Swap RunPod Deployment"
echo "=============================================="

# Step 1: System Update
echo ""
echo "📦 Step 1: Updating system packages..."
apt-get update && apt-get upgrade -y
apt-get install -y git wget curl python3-pip python3-venv ffmpeg libgl1-mesa-glx libglib2.0-0

# Step 2: Clone Repository
echo ""
echo "📥 Step 2: Cloning repository..."
cd /workspace
if [ -d "face-swap" ]; then
    echo "Repository exists, pulling latest..."
    cd face-swap && git pull origin main
else
    git clone https://github.com/Umapathi-S43/Face-test.git face-swap
    cd face-swap
fi

# Step 3: Create Virtual Environment
echo ""
echo "🐍 Step 3: Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Step 4: Install PyTorch with CUDA support
echo ""
echo "🔥 Step 4: Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 5: Install ONNX Runtime with CUDA
echo ""
echo "⚡ Step 5: Installing ONNX Runtime GPU..."
pip install onnxruntime-gpu

# Step 6: Install other dependencies
echo ""
echo "📚 Step 6: Installing dependencies..."
pip install \
    insightface==0.7.3 \
    opencv-python==4.8.1.78 \
    gradio==4.44.1 \
    numpy==1.26.4 \
    Pillow \
    gfpgan \
    basicsr \
    facexlib \
    realesrgan

# Step 7: Download Models
echo ""
echo "🧠 Step 7: Downloading models..."
cd /workspace/face-swap/app/models

# Download InSwapper model
if [ ! -f "inswapper_128.onnx" ]; then
    echo "Downloading InSwapper model..."
    wget -O inswapper_128.onnx "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
fi

# Download GFPGAN model
if [ ! -f "GFPGANv1.4.pth" ]; then
    echo "Downloading GFPGAN model..."
    wget -O GFPGANv1.4.pth "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
fi

# Step 8: Update face_swap_engine.py for CUDA
echo ""
echo "🔧 Step 8: Configuring for CUDA GPU..."
cd /workspace/face-swap/app

# Create GPU-optimized configuration
cat > gpu_config.py << 'EOF'
# GPU Configuration for RunPod
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both RTX 5090 GPUs

# ONNX Runtime providers priority
EXECUTION_PROVIDERS = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB limit per GPU
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]
EOF

# Step 9: Create startup script
echo ""
echo "📝 Step 9: Creating startup script..."
cat > /workspace/face-swap/start_gpu.sh << 'EOF'
#!/bin/bash
cd /workspace/face-swap
source venv/bin/activate

# Set GPU environment
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

# Kill any existing process on port 7860
lsof -ti:7860 | xargs kill -9 2>/dev/null || true
sleep 2

echo "=============================================="
echo "🚀 Starting Face-Swap with RTX 5090 x2"
echo "=============================================="

cd app
python run.py --server-name 0.0.0.0 --server-port 7860
EOF
chmod +x /workspace/face-swap/start_gpu.sh

# Step 10: Verify Installation
echo ""
echo "✅ Step 10: Verifying installation..."
cd /workspace/face-swap
source venv/bin/activate
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

import onnxruntime as ort
print(f'ONNX Runtime version: {ort.__version__}')
print(f'Available providers: {ort.get_available_providers()}')
"

echo ""
echo "=============================================="
echo "✅ Deployment Complete!"
echo "=============================================="
echo ""
echo "To start the application, run:"
echo "  cd /workspace/face-swap && ./start_gpu.sh"
echo ""
echo "The app will be available at:"
echo "  http://[YOUR_RUNPOD_IP]:7860"
echo ""
echo "Or use the RunPod proxy URL from your dashboard"
echo "=============================================="
