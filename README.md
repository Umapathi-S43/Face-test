# 🎭 Face Swap Application

Real-time face swap application with web UI, optimized for MacBook Pro M1 Pro and GPU servers.

## Features

- 📷 **Image Face Swap** - Swap faces in static images
- 🎬 **Video Face Swap** - Process video files with face swap
- 📹 **Live Webcam** - Real-time face swap with webcam
- 🎥 **Virtual Camera** - Output to Teams/Zoom/Google Meet
- 👄 **Mouth Mask** - Preserves lip movement for realistic results
- ✨ **GFPGAN Enhancement** - High-quality face enhancement

## Quick Start

### Local (Mac/Linux)

```bash
# Clone the repo
git clone https://github.com/Umapathi-S43/Face-test.git
cd Face-test

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (run once)
python app/run.py  # Will auto-download models

# Access at http://localhost:7860
```

### GPU Server (RunPod/Vast.ai)

```bash
# Use the deployment script
cd deploy
bash runpod_setup.sh

# Run with CUDA
python gpu_server.py
```

## Project Structure

```
face-swap/
├── app/
│   ├── main.py              # Gradio web UI
│   ├── face_swap_engine.py  # Core face swap logic
│   ├── webcam_manager.py    # Webcam handling
│   └── run.py               # Application launcher
├── deploy/
│   ├── gpu_server.py        # GPU-optimized server
│   ├── runpod_setup.sh      # RunPod deployment script
│   └── GPU_DEPLOYMENT_GUIDE.md
└── requirements.txt
```

## Requirements

- Python 3.10+
- OpenCV
- InsightFace
- ONNX Runtime
- Gradio
- GFPGAN (optional, for enhancement)

## Models (Auto-downloaded)

- `inswapper_128.onnx` - Face swapping model
- `GFPGANv1.4.pth` - Face enhancement model
- `buffalo_l` - Face detection model (InsightFace)

## Usage

### Web Interface

1. Open http://localhost:7860
2. Upload source face image(s)
3. Upload target image/video or use webcam
4. Click "Swap Faces"

### For Video Calls

1. Go to "Virtual Camera" tab
2. Upload source face
3. Click "Start Virtual Camera"
4. Select virtual camera in Teams/Zoom/Meet

## Performance

| Platform        | FPS   |
| --------------- | ----- |
| M1 Pro (CPU)    | 15-20 |
| RTX 4090 (CUDA) | 60+   |
| RTX 3090 (CUDA) | 45+   |

## License

For educational and research purposes only.

## Credits

- [InsightFace](https://github.com/deepinsight/insightface)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)
