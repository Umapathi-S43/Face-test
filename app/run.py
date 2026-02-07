#!/usr/bin/env python3
"""
SurgeryPreview - Plastic Surgery Visualization Tool
Quick Start Script - Downloads models and starts the application
"""

import os
import sys
import urllib.request
from pathlib import Path
import subprocess

# Configuration
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
VENV_DIR = Path.home() / "face-swap" / "facefusion" / "venv"


def download_file(url: str, path: Path, description: str):
    """Download a file with progress"""
    if path.exists():
        print(f"  ✅ {description} already exists")
        return True
    
    print(f"  ⬇️ Downloading {description}...")
    print(f"     URL: {url}")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r     Progress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(path), reporthook=progress_hook)
        print(f"\n  ✅ Downloaded {description}")
        return True
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        return False


def download_models():
    """Download required AI models"""
    print("\n📦 Checking AI Models...")
    MODELS_DIR.mkdir(exist_ok=True)
    
    models = [
        {
            "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
            "path": MODELS_DIR / "inswapper_128.onnx",
            "description": "InSwapper Face Swap Model (300MB)"
        },
        {
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            "path": MODELS_DIR / "GFPGANv1.4.pth",
            "description": "GFPGAN Face Enhancement Model (350MB)"
        }
    ]
    
    all_success = True
    for model in models:
        if not download_file(model["url"], model["path"], model["description"]):
            all_success = False
    
    return all_success


def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n🔍 Checking Dependencies...")
    
    required = [
        "cv2",
        "numpy",
        "gradio",
        "insightface",
        "onnxruntime",
        "torch"
    ]
    
    missing = []
    for module in required:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - MISSING")
            missing.append(module)
    
    return len(missing) == 0


def setup_environment():
    """Setup the Python environment"""
    print("\n🔧 Setting up environment...")
    
    # Create necessary directories
    (APP_DIR / "uploads").mkdir(exist_ok=True)
    (APP_DIR / "outputs").mkdir(exist_ok=True)
    
    print("  ✅ Directories created")


def main():
    """Main entry point"""
    print("=" * 60)
    print("💉 SurgeryPreview - Plastic Surgery Visualization Tool")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    # Download models
    if not download_models():
        print("\n⚠️ Some models failed to download. The app may not work properly.")
        print("   Please manually download the models to:", MODELS_DIR)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please run:")
        print("   cd ~/face-swap/facefusion && source venv/bin/activate")
        print("   pip install insightface gfpgan")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🚀 Starting SurgeryPreview Application...")
    print("=" * 60)
    print("\n📍 Open your browser at: http://localhost:7860")
    print("   Press Ctrl+C to stop\n")
    
    # Import and run the main app
    from main import create_ui
    
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
