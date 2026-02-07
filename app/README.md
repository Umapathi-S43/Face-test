# 💉 SurgeryPreview - Plastic Surgery Visualization Tool

Real-time plastic surgery visualization application optimized for MacBook Pro M1 Pro.

## 🚀 Quick Start

```bash
# Make the start script executable
chmod +x start.sh

# Run the application
./start.sh
```

Or manually:

```bash
# Activate virtual environment
source ~/face-swap/facefusion/venv/bin/activate

# Run the application
python run.py
```

## 📁 Directory Structure

```
app/
├── main.py              # Main application with Gradio UI
├── face_swap_engine.py  # Core visualization engine
├── webcam_manager.py    # Webcam and virtual camera handler
├── run.py               # Setup and launch script
├── start.sh             # Bash startup script
├── models/              # AI models (auto-downloaded)
├── uploads/             # Uploaded images
└── outputs/             # Processed outputs
```

## 📦 Required Models

Models are automatically downloaded on first run:

1. **inswapper_128.onnx** (~300MB) - Face visualization model
2. **GFPGANv1.4.pth** (~350MB) - Face enhancement model

## 🎯 Features

### Image Preview
- Upload 1-10 expected result images (post-surgery look)
- Upload patient's current photo
- Optional face enhancement (GFPGAN)
- Apply to single or all faces

### Video Preview
- Process video files with surgery visualization
- Progress tracking
- MP4 output

### Live Webcam Preview
- Real-time surgery preview from webcam
- Preview window for patient consultations
- Virtual camera output (if available)

## ⚙️ Configuration

### For Best Quality
- Use 5-10 clear photos of expected result
- Include different angles and expressions
- Good lighting in photos

### For Best Performance
- Disable face enhancement for faster processing
- Close other applications
- Use lower resolution webcam if needed

## 🔧 Troubleshooting

### Models not downloading
```bash
# Manually download models:
cd ~/face-swap/app/models

# Download inswapper model
curl -L -o inswapper_128.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"

# Download GFPGAN model
curl -L -o GFPGANv1.4.pth "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
```

### Camera not detected
- Check camera permissions in System Preferences
- Try different camera index (0, 1, 2)

### Low FPS
- Disable face enhancement
- Close other applications
- Reduce webcam resolution

## 📋 System Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- 16GB RAM recommended
- Webcam for live mode

## 🎥 Using with Zoom/Teams/Meet (Remote Consultations)

1. Start the webcam surgery preview
2. Open OBS Studio
3. Add Window Capture → Select "Surgery Preview" window
4. Click "Start Virtual Camera" in OBS
5. In Zoom/Teams, select "OBS Virtual Camera" as your camera

## ⚠️ Disclaimer

This tool is for visualization purposes only. Actual surgical results may vary. 
This is not a medical device and should not be used for medical diagnosis.

## 📝 License

For educational, clinical visualization, and personal use only.
Please obtain consent when using this technology with patients.
