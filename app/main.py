#!/usr/bin/env python3
"""
PlasticVision Pro - Face Swap Application
Single unified app that works EVERYWHERE:
  - Local MacBook (CPU/CoreML)
  - RunPod GPU server (CUDA)
  - Any machine with a browser

Auto-detects:
  - GPU vs CPU (CUDA > CoreML > CPU)
  - Local vs server (auto share=True on server)
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import time
from typing import Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Single engine - auto-detects GPU
from face_swap_engine import FaceSwapEngine

# =====================================================
# GLOBALS
# =====================================================

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


def is_server() -> bool:
    """Detect if running on a headless server (no display)."""
    if sys.platform == "darwin":
        return False
    if os.environ.get("RUNPOD_POD_ID"):
        return True
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def get_gpu_status() -> str:
    """Get GPU info string."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            try:
                import torch
                if torch.cuda.is_available():
                    gpus = []
                    for i in range(torch.cuda.device_count()):
                        name = torch.cuda.get_device_name(i)
                        mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                        gpus.append(f"{name} ({mem}GB)")
                    return f"GPU: {', '.join(gpus)}"
            except ImportError:
                return "GPU: CUDA (via ONNX)"
        if "CoreMLExecutionProvider" in providers:
            return "Apple Silicon (CPU mode)"
        return "CPU Mode"
    except Exception:
        return "CPU Mode"


# =====================================================
# UTILITIES
# =====================================================


def normalize_source_paths(source_images) -> List[str]:
    """Normalize Gradio file uploads to a list of file paths."""
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
    """Initialize the face swap engine (singleton)."""
    global face_swap_engine
    if face_swap_engine is None:
        print("Initializing FaceSwapEngine (auto-detects GPU)...")
        face_swap_engine = FaceSwapEngine(models_dir=str(MODELS_DIR))
        face_swap_engine.initialize()
    return face_swap_engine


# =====================================================
# FACE LOADING & SETTINGS
# =====================================================


def load_source_faces(source_images) -> str:
    """Load source face images into the engine."""
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
            return f"Loaded {len(source_paths)} source face(s) - Ready!"
        else:
            source_faces_loaded = False
            return "Failed to detect faces in uploaded images"
    except Exception as e:
        source_faces_loaded = False
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def update_settings(mouth_mask: bool, sharpness: float, enhance: bool):
    """Update quality settings."""
    global current_settings
    current_settings["use_mouth_mask"] = mouth_mask
    current_settings["sharpness"] = sharpness
    current_settings["enhance"] = enhance
    return f"Settings: Lip Sync={mouth_mask}, Sharpness={sharpness}, HD={enhance}"


# =====================================================
# LIVE WEBCAM (Browser-based, works everywhere)
# =====================================================


def process_webcam_frame(frame):
    """
    Process a single webcam frame from browser.
    Gradio sends RGB numpy array, swap_face_frame expects RGB, returns RGB.
    Works identically on local and server.
    """
    global face_swap_engine, source_faces_loaded, _frame_count

    _frame_count += 1

    if _frame_count <= 3 or _frame_count % 60 == 0:
        print(f"[STREAM] frame #{_frame_count} | "
              f"type={type(frame).__name__} | "
              f"shape={frame.shape if isinstance(frame, np.ndarray) else 'N/A'} | "
              f"faces_loaded={source_faces_loaded}")

    if frame is None:
        return None

    # Ensure numpy array
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

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
        start_time = time.time()

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

        if _frame_count <= 5 or _frame_count % 30 == 1:
            status = "SWAPPED" if swapped else "NO_FACE"
            print(f"[SWAP] Frame #{_frame_count}: {status} | {elapsed_ms:.0f}ms")

        display = result.copy()
        color = (0, 255, 0) if swapped else (255, 165, 0)
        label = "LIVE" if swapped else "No face"
        cv2.rectangle(display, (0, 0), (140, 30), (0, 0, 0), -1)
        cv2.putText(display, label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return display

    except Exception as e:
        print(f"[ERROR] Frame #{_frame_count}: {e}")
        import traceback
        traceback.print_exc()
        return frame


# =====================================================
# IMAGE SWAP
# =====================================================


def process_image_swap(source_images, target_image, enhance: bool = True):
    """Swap face in a single image."""
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
            return None, "Face swap failed - no faces detected"
    except Exception as e:
        return None, f"Error: {str(e)}"


# =====================================================
# VIDEO SWAP
# =====================================================


def process_video_swap(
    source_images,
    target_video,
    enhance: bool = False,
    mouth_mask: bool = True,
    sharpness: float = 0.3,
    progress=gr.Progress(),
):
    """Swap face in a video file."""
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
# FACE DETECTION PREVIEW
# =====================================================


def detect_faces_preview(image: str) -> Tuple[Optional[str], str]:
    """Detect and preview faces in an image."""
    try:
        engine = initialize_engine()
        if not image:
            return None, "Please upload an image"
        result, face_count = engine.detect_and_draw_faces(image)
        if result is not None:
            output_path = str(OUTPUT_DIR / f"faces_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, f"Detected {face_count} face(s)"
        else:
            return None, "No faces detected"
    except Exception as e:
        return None, f"Error: {str(e)}"


# =====================================================
# UNIFIED UI
# =====================================================


def create_ui():
    """Create the Gradio web interface - works on local AND server."""
    gpu_status = get_gpu_status()
    server_mode = is_server()

    with gr.Blocks(
        title="PlasticVision Pro",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px !important; margin: auto; }",
    ) as app:
        gr.Markdown(
            f"""
        # PlasticVision Pro - Face Swap
        **{gpu_status}** | {'Server Mode' if server_mode else 'Local Mode'}

        Upload source face images, then use Live Preview, Image Swap, or Video Swap.
        """
        )

        # ---- Source Face Upload (shared across all tabs) ----
        with gr.Row():
            with gr.Column(scale=1):
                source_upload = gr.File(
                    label="Upload Source Faces (1-10 images)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                load_btn = gr.Button("Load Faces", variant="primary", size="lg")
                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload faces and click Load Faces",
                )
            with gr.Column(scale=1):
                gr.Markdown("### Quality Settings")
                mouth_mask = gr.Checkbox(
                    label="Natural Lip Movement", value=True,
                    info="Preserves original lip sync",
                )
                sharpness = gr.Slider(
                    label="Sharpness",
                    minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                )
                enhance = gr.Checkbox(
                    label="HD Enhancement (slower)", value=False,
                    info="GFPGAN face enhancement",
                )
                apply_btn = gr.Button("Apply Settings")
                settings_status = gr.Textbox(
                    label="Settings", interactive=False, value="Default",
                )

        load_btn.click(
            fn=load_source_faces,
            inputs=[source_upload],
            outputs=[load_status],
        )
        apply_btn.click(
            fn=update_settings,
            inputs=[mouth_mask, sharpness, enhance],
            outputs=[settings_status],
        )

        # ---- Tabs ----
        with gr.Tabs():
            # ======== TAB 1: LIVE WEBCAM ========
            with gr.TabItem("Live Preview"):
                gr.Markdown(
                    """
                ### Real-Time Face Swap
                1. Load source faces above and click Load Faces
                2. The webcam below will show your face-swapped live stream
                """
                )
                # Use gr.Interface with live=True — works on ALL Gradio versions
                # This is the ONLY reliable way for continuous webcam streaming in Gradio 4.x
                live_interface = gr.Interface(
                    fn=process_webcam_frame,
                    inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam"),
                    outputs=gr.Image(label="Face Swap Output", type="numpy"),
                    live=True,
                    title=None,
                    description=None,
                )

            # ======== TAB 2: IMAGE SWAP ========
            with gr.TabItem("Image Swap"):
                gr.Markdown("### Swap face in a single image")
                with gr.Row():
                    target_image = gr.Image(
                        label="Target Image (face to replace)",
                        type="filepath",
                    )
                    swap_output = gr.Image(label="Result")
                img_enhance = gr.Checkbox(label="HD Enhancement", value=True)
                swap_btn = gr.Button("Swap Face", variant="primary")
                swap_status = gr.Textbox(label="Status", interactive=False)

                swap_btn.click(
                    fn=process_image_swap,
                    inputs=[source_upload, target_image, img_enhance],
                    outputs=[swap_output, swap_status],
                )

            # ======== TAB 3: VIDEO SWAP ========
            with gr.TabItem("Video Swap"):
                gr.Markdown("### Swap face in a video")
                target_video = gr.Video(label="Target Video")
                with gr.Row():
                    vid_enhance = gr.Checkbox(label="HD Enhancement", value=False)
                    vid_mouth = gr.Checkbox(label="Lip Sync", value=True)
                    vid_sharp = gr.Slider(
                        label="Sharpness",
                        minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                    )
                vid_btn = gr.Button("Process Video", variant="primary")
                vid_output = gr.Video(label="Result")
                vid_status = gr.Textbox(label="Status", interactive=False)

                vid_btn.click(
                    fn=process_video_swap,
                    inputs=[source_upload, target_video, vid_enhance, vid_mouth, vid_sharp],
                    outputs=[vid_output, vid_status],
                )

            # ======== TAB 4: FACE DETECTION ========
            with gr.TabItem("Face Detection"):
                gr.Markdown("### Detect faces in an image")
                detect_input = gr.Image(label="Upload Image", type="filepath")
                detect_btn = gr.Button("Detect Faces", variant="primary")
                detect_output = gr.Image(label="Detected Faces")
                detect_status = gr.Textbox(label="Status", interactive=False)

                detect_btn.click(
                    fn=detect_faces_preview,
                    inputs=[detect_input],
                    outputs=[detect_output, detect_status],
                )

        gr.Markdown(
            """
        ---
        **Tips:** Good lighting = better detection. Face the camera directly.
        Disable HD Enhancement for faster live preview.
        """
        )

    return app


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    server_mode = is_server()

    print("=" * 60)
    print("PlasticVision Pro - Face Swap")
    print("=" * 60)
    print(f"Mode:   {'SERVER' if server_mode else 'LOCAL'}")
    print(f"GPU:    {get_gpu_status()}")
    print(f"Models: {MODELS_DIR}")
    print("=" * 60)

    app = create_ui()
    app.queue(max_size=20)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=server_mode,  # Auto-enable share on server
        show_error=True,
    )
