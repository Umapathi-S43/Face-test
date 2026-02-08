#!/usr/bin/env python3
"""
PlasticVision Pro v2 — Backend Entry Point
Downloads models if needed, then starts the FastAPI server.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

REQUIRED_MODELS = [
    {
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
        "path": MODELS_DIR / "inswapper_128.onnx",
        "name": "InSwapper Face Swap Model (529MB)",
    },
    {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "path": MODELS_DIR / "GFPGANv1.4.pth",
        "name": "GFPGAN Face Enhancement (332MB)",
    },
]


def download_file(url: str, path: Path, name: str) -> bool:
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {name} ({size_mb:.0f}MB)")
        return True
    print(f"  ⬇️ Downloading {name}...")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        def progress(count, block, total):
            pct = int(count * block * 100 / total) if total > 0 else 0
            print(f"\r     {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, str(path), reporthook=progress)
        print(f"\n  ✅ Downloaded {name}")
        return True
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        return False


def check_dependencies():
    print("\n🔍 Checking dependencies...")
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "insightface": "insightface",
        "onnxruntime": "onnxruntime (or onnxruntime-gpu)",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} — install: pip install {pkg}")
            missing.append(pkg)
    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="PlasticVision Pro v2 — GPU Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    args = parser.parse_args()

    print("=" * 60)
    print("🎭 PlasticVision Pro v2 — GPU Backend")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Download models
    if not args.skip_download:
        print(f"\n📦 Checking AI models ({MODELS_DIR})...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for model in REQUIRED_MODELS:
            download_file(model["url"], model["path"], model["name"])

    # Copy models from existing app dir if available
    existing_models = Path(__file__).parent.parent.parent / "models"
    if existing_models.exists() and existing_models != MODELS_DIR:
        for model in REQUIRED_MODELS:
            if not model["path"].exists():
                src = existing_models / model["path"].name
                if src.exists():
                    print(f"  📋 Copying {model['path'].name} from existing app...")
                    import shutil
                    shutil.copy2(str(src), str(model["path"]))

    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print(f"   WebSocket:  ws://{args.host}:{args.port}/ws/stream")
    print(f"   API docs:   http://{args.host}:{args.port}/docs")
    print("=" * 60)

    import uvicorn
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ws_max_size=50 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
