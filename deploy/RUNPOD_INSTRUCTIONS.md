# 🚀 RunPod Deployment - Complete Guide

## Your RunPod Instance

- **Pod ID:** e3k0rc4uqhr1w9
- **GPU:** RTX 5090 x2
- **Cost:** $1.79/hr

---

## Option 1: Web Terminal (Easiest)

1. Go to [RunPod Dashboard](https://runpod.io/console/pods)
2. Click on your pod `e3k0rc4uqhr1w9`
3. Click the **Terminal** icon (or "Connect" → "Start Web Terminal")
4. **Copy and paste this entire block:**

```bash
cd /workspace && \
apt-get update && apt-get install -y git wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 && \
git clone https://github.com/Umapathi-S43/Face-test.git face-swap && \
cd face-swap && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install onnxruntime-gpu insightface==0.7.3 opencv-python==4.8.1.78 gradio==4.44.1 numpy==1.26.4 Pillow gfpgan basicsr facexlib realesrgan && \
mkdir -p app/models && \
cd app/models && \
wget -O inswapper_128.onnx "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" && \
wget -O GFPGANv1.4.pth "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" && \
cd /workspace/face-swap/app && \
python run.py --server-name 0.0.0.0 --server-port 7860
```

5. Wait for "Running on http://0.0.0.0:7860"
6. Access via RunPod's **Proxy URL** (HTTP Port 7860)

---

## Option 2: SSH Access

### Step 1: Add Your SSH Key to RunPod

1. Go to RunPod → Settings → SSH Keys
2. Add this public key:

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC2fkx7/+WY8uCOtwj8JBVJBU1Dk3iUMXnqDNGW0PJer+uLyMiefBADRdNpMSLEDFgQKVpsA1RTtjEPNk5cnwDZFhcCJXXGd9nG91b+VS5ChOZaahGGxGIIQ4LFSSWuOVUc2dGMFbrtPza0j3LMVLzpvrE3b3RO30NNteuuskWpG4wA5gglGCGJ1P8nff8qsh9XvbkN9DO/1SDxmzPNLrb9G0no3JheNQwcwkwJpjAYegxIh0U0ENFtmIbXEpqnRenPf8SMpZDI9rOvHs6pjOUpEOnnTD6q72O10SlP/FxUuNfNaFb9M+B/MYs/Q0Ne6tYWoOc9aO32/FrXKwp5giCn
```

### Step 2: Get SSH Command

From RunPod dashboard → Connect → SSH over exposed TCP
Copy the command (looks like: `ssh root@IP -p PORT`)

### Step 3: Connect and Deploy

```bash
ssh root@[IP] -p [PORT]
# Then run the deployment commands above
```

---

## Accessing Your App

After deployment, get the URL from RunPod:

1. Click your pod
2. Click "Connect"
3. Look for "HTTP Service [Port 7860]"
4. Click the link (e.g., `https://e3k0rc4uqhr1w9-7860.proxy.runpod.net`)

---

## Troubleshooting

### Check GPU Status

```bash
nvidia-smi
```

### Check if App is Running

```bash
ps aux | grep python
```

### View Logs

```bash
cd /workspace/face-swap/app && cat nohup.out
```

### Restart App

```bash
pkill -f "python run.py"
cd /workspace/face-swap/app
source ../venv/bin/activate
nohup python run.py --server-name 0.0.0.0 --server-port 7860 &
```
