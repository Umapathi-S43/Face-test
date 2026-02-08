#!/usr/bin/env python3
"""
PlasticVision Pro - Live Real-Time Face Swap
Uses the SAME working face_swap_engine.py that works locally.
Engine auto-detects GPU (CUDA/CoreML/CPU).
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import time
from typing import Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

# Use the SAME working engine as main.py (auto-detects GPU)
from face_swap_engine import FaceSwapEngine

face_swap_engine: Optional[FaceSwapEngine] = None
source_faces_loaded = False
current_settings = {
    "use_mouth_mask": True,
    "sharpness": 0.3,
    "enhance": False,
}

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
MODELS_DIR = APP_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

_frame_count = 0


def normalize_source_paths(source_images) -> List[str]:
    if source_images is None:
        return []
    source_paths = []
    if isinstance(source_images, str):
        source_paths = [source_images]
    elif isinstance(source_images, list):
        for item in source_images:
            if isinstance(item, str):
                source_paths.append(item)
            elif hasattr(item, "name"):
                source_paths.append(item.name)
            elif isinstance(item, dict):
                if "name" in item:
                    source_paths.append(item["name"])
                elif "path" in item:
                    source_paths.append(item["path"])
            elif hasattr(item, "path"):
                source_paths.append(item.path)
    elif hasattr(source_images, "name"):
        source_paths = [source_images.name]
    elif hasattr(source_images, "path"):
        source_paths = [source_images.path]
    return [p for p in source_paths if p and os.path.exists(p)]


def initialize_engine():
    global face_swap_engine
    if face_swap_engine is None:
        print("Initializing FaceSwapEngine (auto-detects GPU)...")
        face_swap_engine = FaceSwapEngine(models_dir=str(MODELS_DIR))
        face_swap_engine.initialize()
    return face_swap_engine


def get_gpu_status() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                gpus.append(f"{name} ({mem}GB)")
            return f"{gpu_count}x GPU: " + ", ".join(gpus)
        return "CPU Mode"
    except Exception:
        return "CPU Mode"


def load_source_faces(source_images) -> str:
    global source_faces_loaded, face_swap_engine
    source_paths = normalize_source_paths(source_images)
    if not source_paths:
        source_faces_loaded = False
        return "Please upload source face images"
    try:
        engine = initialize_engine()
        print(f"Loading {len(source_paths)} face(s) from: {source_paths}")
        if engine.load_source_faces(source_paths):
            source_faces_loaded = True
            return f"Loaded {len(source_paths)} source face(s) - Ready for live preview!"
        else:
            source_faces_loaded = False
            return "Failed to detect faces in uploaded images"
    except Exception as e:
        source_faces_loaded = False
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def update_settings(mouth_mask: bool, sharpness: float, enhance: bool):
    global current_settings
    current_settings["use_mouth_mask"] = mouth_mask
    current_settings["sharpness"] = sharpness
    current_settings["enhance"] = enhance
    return f"Settings: Lip Sync={mouth_mask}, Sharpness={sharpness}, HD={enhance}"


def process_webcam_frame(frame):
    """
    Process a single webcam frame from browser.
    Uses the EXACT same swap_face_frame() that works locally.
    Gradio sends RGB, swap_face_frame expects RGB, returns RGB.
    """
    global face_swap_engine, source_faces_loaded, _frame_count

    if frame is None:
        return None

    if not source_faces_loaded or face_swap_engine is None:
        display = frame.copy()
        cv2.putText(display, "Upload source faces first!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return display

    if face_swap_engine.source_face is None:
        display = frame.copy()
        cv2.putText(display, "Click Load Faces first!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return display

    try:
        _frame_count += 1
        start_time = time.time()

        # Gradio sends RGB. swap_face_frame expects RGB. Direct pass - no conversion.
        result = face_swap_engine.swap_face_frame(
            frame,
            enhance=current_settings["enhance"],
            swap_all=False,
            use_mouth_mask=current_settings["use_mouth_mask"],
            use_color_transfer=True,
            sharpness=current_settings["sharpness"],
            opacity=1.0,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        swapped = result is not frame

        if _frame_count % 30 == 1:
            status = "SWAPPED" if swapped else "NO_FACE"
            print(f"Frame #{_frame_count}: {status} | {elapsed_ms:.0f}ms")

        display = result.copy()
        color = (0, 255, 0) if swapped else (255, 165, 0)
        label = "LIVE" if swapped else "No face"
        cv2.rectangle(display, (0, 0), (140, 30), (0, 0, 0), -1)
        cv2.putText(display, label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return display

    except Exception as e:
        if _frame_count % 30 == 1:
            print(f"Frame error: {e}")
            import traceback
            traceback.print_exc()
        return frame


def process_image_swap(source_images, target_image, enhance: bool = True):
    try:
        engine = initialize_engine()
        source_paths = normalize_source_paths(source_images)
        if not source_paths:
            return None, "Upload source face images"
        if target_image is None:
            return None, "Upload target image"
        result = engine.swap_face(
            source_paths=source_paths,
            target_path=target_image,
            enhance=enhance,
            swap_all=False,
        )
        if result is not None:
            output_path = str(OUTPUT_DIR / f"result_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, "Face swap completed!"
        else:
            return None, "Face swap failed"
    except Exception as e:
        return None, f"Error: {str(e)}"


def process_video_swap(
    source_images,
    target_video,
    enhance: bool = False,
    mouth_mask: bool = True,
    sharpness: float = 0.3,
    progress=gr.Progress(),
):
    try:
        engine = initialize_engine()
        source_paths = normalize_source_paths(source_images)
        if not source_paths:
            return None, "Upload source face images"
        if target_video is None:
            return None, "Upload target video"
        output_path = str(OUTPUT_DIR / f"video_{int(time.time())}.mp4")
        success = engine.swap_face_video(
            source_paths=source_paths,
            target_video_path=target_video,
            output_path=output_path,
            enhance=enhance,
            swap_all=False,
            use_mouth_mask=mouth_mask,
            sharpness=sharpness,
            progress_callback=lambda p: progress(p, desc="Processing..."),
        )
        if success:
            return output_path, "Video processed!"
        else:
            return None, "Video processing failed"
    except Exception as e:
        return None, f"Error: {str(e)}"


# =====================================================
# UI
# =====================================================


def create_ui():
    gpu_status = get_gpu_status()

    with gr.Blocks(
        title="PlasticVision Pro - Live", theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            f"""
        # PlasticVision Pro - Live Face Swap
        **{gpu_status}** | Real-Time Face Swap with GPU Acceleration

        **Steps:**
        1. Upload source face images (the face you want to appear)
        2. Click Load Faces
        3. Allow camera access when prompted
        4. See real-time face swap!
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                source_upload = gr.File(
                    label="Upload Source Faces (1-10 images)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                load_btn = gr.Button(
                    "Load Faces", variant="primary", size="lg"
                )
                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload faces and click Load Faces",
                )
                gr.Markdown("### Quality Settings")
                mouth_mask = gr.Checkbox(
                    label="Natural Lip Movement",
                    value=True,
                    info="Preserves original lip sync",
                )
                sharpness = gr.Slider(
                    label="Sharpness",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                )
                enhance = gr.Checkbox(
                    label="HD Enhancement (slower)",
                    value=False,
                    info="Enable for best quality, disable for speed",
                )
                apply_settings_btn = gr.Button("Apply Settings")
                settings_status = gr.Textbox(
                    label="Settings",
                    interactive=False,
                    value="Default settings active",
                )

            with gr.Column(scale=2):
                gr.Markdown("### Live Output")
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Your Camera (processed in real-time)",
                    height=500,
                    mirror_webcam=True,
                )
                gr.Markdown(
                    """
                **Tips:**
                - Good lighting = better face detection
                - Face the camera directly
                - Disable HD Enhancement for faster processing
                """
                )

        load_btn.click(
            fn=load_source_faces,
            inputs=[source_upload],
            outputs=[load_status],
        )
        apply_settings_btn.click(
            fn=update_settings,
            inputs=[mouth_mask, sharpness, enhance],
            outputs=[settings_status],
        )
        webcam_input.stream(
            fn=process_webcam_frame,
            inputs=[webcam_input],
            outputs=[webcam_input],
        )

    return app


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PlasticVision Pro - Live Real-Time Face Swap")
    print("=" * 60)
    print(f"Models: {MODELS_DIR}")
    print(f"GPU: {get_gpu_status()}")
    print("=" * 60)

    app = create_ui()
    app.queue(max_size=20)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
