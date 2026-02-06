# ===========================================

# 🚀 QUICK GPU DEPLOYMENT GUIDE

# ===========================================

## Option 1: RunPod (FASTEST - 2 minutes)

### Step 1: Create Account

1. Go to https://runpod.io
2. Sign up and add $10-20 credits

### Step 2: Deploy GPU Pod

1. Click "Deploy" → "GPU Pods"
2. Select: **RTX 4090** ($0.44/hr) or **RTX 3090** ($0.31/hr)
3. Choose template: **RunPod Pytorch 2.1**
4. Set storage: 50GB
5. Click "Deploy"

### Step 3: Connect & Setup

```bash
# SSH into your pod (connection details in RunPod dashboard)
# Or use the web terminal

# Run setup script
cd /workspace
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam

# Create venv
python -m venv venv
source venv/bin/activate

# Install with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.17.0
pip install numpy==1.26.4 opencv-python pillow insightface gradio
pip install "git+https://github.com/TencentARC/GFPGAN.git@master"

# Download models
mkdir -p models && cd models
wget https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx
wget https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth
cd ..
```

### Step 4: Run Server

```bash
# Copy the gpu_server.py to your pod
python gpu_server.py
```

Access via the public Gradio link or: `http://your-pod-ip:7860`

---

## Option 2: Vast.ai (CHEAPEST)

### Step 1: Create Account

1. Go to https://vast.ai
2. Sign up and add credits

### Step 2: Rent GPU

1. Search for: RTX 4090 or RTX 3090
2. Filter by: "PyTorch", "Reliable"
3. Select cheapest option (~$0.30-0.50/hr)
4. Click "Rent"

### Step 3: Same setup as RunPod

---

## Option 3: Google Colab Pro ($10/month)

### Quick Notebook:

```python
# Cell 1: Setup
!git clone https://github.com/hacksider/Deep-Live-Cam.git
%cd Deep-Live-Cam
!pip install torch torchvision onnxruntime-gpu insightface gradio gfpgan

# Cell 2: Download models
!mkdir -p models
!wget -O models/inswapper_128_fp16.onnx https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx
!wget -O models/GFPGANv1.4.pth https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth

# Cell 3: Run
!python gpu_server.py
```

---

## Performance Comparison

| GPU          | FPS (Video) | FPS (Realtime) | Cost/hr |
| ------------ | ----------- | -------------- | ------- |
| RTX 4090     | 60+ FPS     | 45+ FPS        | $0.44   |
| RTX 3090     | 45+ FPS     | 35+ FPS        | $0.31   |
| A100         | 80+ FPS     | 60+ FPS        | $1.10   |
| T4 (Colab)   | 20+ FPS     | 15+ FPS        | ~$0.10  |
| M1 Pro (CPU) | 15 FPS      | 10 FPS         | Local   |

---

## Files for Deployment

Upload these to your GPU server:

- `gpu_server.py` - Web server with Gradio UI
- `runpod_setup.sh` - Automated setup script

Both files are in: `~/face-swap/deploy/`
