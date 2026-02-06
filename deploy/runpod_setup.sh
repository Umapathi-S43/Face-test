#!/bin/bash
# ===========================================
# Deep-Live-Cam GPU Deployment Script
# For RunPod / Vast.ai / Any CUDA GPU Server
# ===========================================

echo "🚀 Deep-Live-Cam GPU Deployment"
echo "================================"

# Update system
apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    wget

# Clone Deep-Live-Cam
cd /workspace
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ONNX Runtime with CUDA
pip install onnxruntime-gpu==1.17.0

# Install other dependencies
pip install numpy==1.26.4 opencv-python==4.9.0.80 pillow insightface onnx psutil
pip install customtkinter tk tensorflow
pip install "git+https://github.com/xinntao/BasicSR.git@master"
pip install "git+https://github.com/TencentARC/GFPGAN.git@master"
pip install opennsfw2 gradio

# Download models
mkdir -p models
cd models
wget -O inswapper_128_fp16.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
wget -O GFPGANv1.4.pth "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth"
cd ..

echo ""
echo "✅ Installation Complete!"
echo ""
echo "To run with CUDA GPU:"
echo "  python run.py --execution-provider cuda"
echo ""
echo "For web interface, run the Gradio app"
