#!/bin/bash
# PlasticVision Pro - RunPod Startup Script
# Optimized for 2x RTX 5090 GPUs

echo "========================================"
echo "💉 PlasticVision Pro - GPU Deployment"
echo "========================================"

# Navigate to app directory
cd /workspace/face-swap/app

# Activate virtual environment if exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Check GPU status
echo ""
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Check CUDA availability
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')" 2>/dev/null || echo "⚠️ PyTorch CUDA check skipped"
echo ""

# Kill any existing processes on port 7860
pkill -f "python.*main" 2>/dev/null
sleep 2

# Start the optimized app
echo "🚀 Starting PlasticVision Pro..."
echo ""

# Use the GPU-optimized version
if [ -f "main_gpu.py" ]; then
    python main_gpu.py
else
    # Fallback to standard
    python main.py
fi
