# 💉 SurgeryPreview - Plastic Surgery Visualization Tool

Real-time plastic surgery preview application that allows doctors to showcase expected surgical outcomes to patients using their webcam.

## Overview

SurgeryPreview is a medical visualization tool designed for plastic surgeons and cosmetic clinics. It enables:
- **Doctors** to upload expected post-surgery face images
- **Patients** to see the expected outcome overlaid on their live webcam feed
- Real-time preview of surgical results before the procedure

## Features

- 📷 **Image Preview** - Upload expected result image and see it applied to patient photos
- 📹 **Live Webcam Preview** - Real-time visualization of expected surgical outcomes
- 🎥 **Virtual Camera** - Share preview in video calls (Teams/Zoom/Google Meet)
- 👄 **Mouth Mask** - Preserves natural lip movement for realistic preview
- ✨ **Face Enhancement** - High-quality GFPGAN enhancement for better visualization

## Quick Start

### Local Installation (Mac/Linux)

```bash
# Clone the repo
git clone https://github.com/Umapathi-S43/Face-test.git
cd Face-test

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models and start (run once)
python app/run.py  # Will auto-download models

# Access at http://localhost:7860
```

### GPU Server (For Clinics)

```bash
# Use the deployment script
cd deploy
bash runpod_setup.sh

# Run with CUDA
python gpu_server.py
```

## Project Structure

```
SurgeryPreview/
├── app/
│   ├── main.py              # Gradio web UI
│   ├── face_swap_engine.py  # Core visualization engine
│   ├── webcam_manager.py    # Webcam handling
│   └── run.py               # Application launcher
├── deploy/
│   ├── gpu_server.py        # GPU-optimized server
│   ├── runpod_setup.sh      # Cloud deployment script
│   └── GPU_DEPLOYMENT_GUIDE.md
└── requirements.txt
```

## How to Use

### For Doctors/Clinicians

1. Open the web interface at http://localhost:7860
2. Go to "📷 Image Preview" or "📹 Live Webcam" tab
3. Upload the **Expected Result Image** (post-surgery visualization)
4. Have the patient use the webcam or upload their current photo
5. The expected surgical outcome will be displayed in real-time

### For Video Consultations

1. Go to "Virtual Camera" tab
2. Upload expected result image
3. Click "Start Virtual Camera"
4. Select virtual camera in Teams/Zoom/Meet for remote consultations

## Requirements

- Python 3.10+
- OpenCV
- InsightFace
- ONNX Runtime
- Gradio
- GFPGAN (for enhancement)

## Models (Auto-downloaded)

- `inswapper_128.onnx` - Face visualization model
- `GFPGANv1.4.pth` - Face enhancement model
- `buffalo_l` - Face detection model (InsightFace)

## Performance

| Platform        | FPS   |
| --------------- | ----- |
| M1 Pro (CPU)    | 15-20 |
| RTX 4090 (CUDA) | 60+   |
| RTX 3090 (CUDA) | 45+   |

## Disclaimer

⚠️ **Medical Disclaimer**: This tool is for visualization purposes only. Actual surgical results may vary. This is not a medical device and should not be used for medical diagnosis or treatment decisions.

## License

For educational, research, and clinical visualization purposes only.

## Credits

- [InsightFace](https://github.com/deepinsight/insightface)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
