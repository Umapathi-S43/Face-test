#!/usr/bin/env python3
"""
PlasticVision Pro - Face Swap Application

Architecture designed for Gradio 4.x AND 5.x compatibility:
  - Live webcam uses gr.Interface(live=True) with streaming=True
    This is the ONLY reliable way for continuous webcam streaming in Gradio 4.x
  - Other features (image swap, video swap, face detection) are separate Interfaces
  - All combined using gr.TabbedInterface (works in ALL Gradio versions)
  - The live webcam Interface is NOT nested inside gr.Blocks (critical!)

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
_source_image_paths: List[str] = []


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
                    return f"🟢 GPU: {', '.join(gpus)}"
            except ImportError:
                return "🟢 GPU: CUDA (via ONNX)"
        if "CoreMLExecutionProvider" in providers:
            return "🟡 Apple Silicon (CPU mode)"
        return "🔴 CPU Mode"
    except Exception:
        return "🔴 CPU Mode"


# =====================================================
# ENGINE MANAGEMENT
# =====================================================


def initialize_engine():
    """Initialize the face swap engine (singleton)."""
    global face_swap_engine
    if face_swap_engine is None:
        print("🚀 Initializing FaceSwapEngine (auto-detects GPU)...")
        face_swap_engine = FaceSwapEngine(models_dir=str(MODELS_DIR))
        face_swap_engine.initialize()
        print("✅ Engine ready!")
    return face_swap_engine


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


# =====================================================
# FACE LOADING (used by Setup tab)
# =====================================================


def load_source_faces(source_images) -> str:
    """Load source face images into the engine and store paths globally."""
    global source_faces_loaded, face_swap_engine, _source_image_paths
    source_paths = normalize_source_paths(source_images)
    if not source_paths:
        source_faces_loaded = False
        _source_image_paths = []
        return "❌ Please upload source face images"
    try:
        engine = initialize_engine()
        print(f"📸 Loading {len(source_paths)} face(s) from: {source_paths}")
        if engine.load_source_faces(source_paths):
            source_faces_loaded = True
            _source_image_paths = source_paths
            return f"✅ Loaded {len(source_paths)} source face(s) — Ready for live swap!"
        else:
            source_faces_loaded = False
            _source_image_paths = []
            return "❌ Failed to detect faces in uploaded images. Try clearer photos."
    except Exception as e:
        source_faces_loaded = False
        _source_image_paths = []
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


def update_settings(mouth_mask: bool, sharpness: float, enhance: bool) -> str:
    """Update quality settings."""
    global current_settings
    current_settings["use_mouth_mask"] = mouth_mask
    current_settings["sharpness"] = sharpness
    current_settings["enhance"] = enhance
    return f"✅ Lip Sync={mouth_mask}, Sharpness={sharpness:.1f}, HD={enhance}"


# =====================================================
# TAB 1: SETUP — Load faces & configure settings
# =====================================================


def create_setup_interface():
    """Create the Setup tab using gr.Blocks."""
    gpu_status = get_gpu_status()
    server_mode = is_server()

    with gr.Blocks() as setup_app:
        gr.Markdown(
            f"""
## 🎭 Setup — Load Source Faces
**{gpu_status}** | {'🌐 Server Mode' if server_mode else '💻 Local Mode'}

**Step 1:** Upload clear photos of the face you want to swap TO  
**Step 2:** Click "Load Faces" to load them into GPU memory  
**Step 3:** Switch to "Live Preview" tab for real-time face swap
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                source_upload = gr.File(
                    label="📸 Upload Source Face Photos (1-10 clear face images)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                load_btn = gr.Button("🔥 Load Faces into GPU", variant="primary", size="lg")
                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="⏳ Upload face photos and click Load Faces",
                    lines=2,
                )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Quality Settings")
                mouth_mask = gr.Checkbox(
                    label="Natural Lip Movement", value=True,
                    info="Preserves original lip movement for realistic result",
                )
                sharpness = gr.Slider(
                    label="Sharpness", minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                )
                enhance = gr.Checkbox(
                    label="HD Enhancement (slower)", value=False,
                    info="GFPGAN face enhancement — disable for faster live preview",
                )
                apply_btn = gr.Button("Apply Settings")
                settings_status = gr.Textbox(
                    label="Current Settings", interactive=False, value="Default",
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

    return setup_app


# =====================================================
# TAB 2: LIVE PREVIEW — Continuous webcam face swap
# =====================================================


def process_webcam_frame(frame):
    """
    Process a single webcam frame from browser.

    This function is called continuously by gr.Interface(live=True)
    when streaming=True on the webcam Image input.

    Gradio sends RGB numpy array.
    face_swap_engine.swap_face_frame() expects RGB, returns RGB.
    """
    global face_swap_engine, source_faces_loaded, _frame_count

    _frame_count += 1

    if frame is None:
        return None

    # Ensure numpy array
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    # Log periodically
    if _frame_count <= 3 or _frame_count % 100 == 0:
        print(f"[LIVE] frame #{_frame_count} | "
              f"shape={frame.shape} | "
              f"faces_loaded={source_faces_loaded}")

    # If no source faces loaded, show original with overlay message
    if not source_faces_loaded or face_swap_engine is None:
        display = frame.copy()
        h, w = display.shape[:2]
        # Semi-transparent overlay bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        cv2.putText(display, "Go to Setup tab -> Load Faces first!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        return display

    if face_swap_engine.source_face is None:
        display = frame.copy()
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        cv2.putText(display, "No source face detected. Upload clearer photos.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
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

        if _frame_count <= 5 or _frame_count % 60 == 0:
            status = "✅ SWAPPED" if swapped else "⚠️ NO_FACE_IN_FRAME"
            print(f"[SWAP] Frame #{_frame_count}: {status} | {elapsed_ms:.0f}ms | "
                  f"FPS≈{1000/max(elapsed_ms,1):.1f}")

        # Add minimal status overlay
        display = result.copy()
        h, w = display.shape[:2]
        color = (0, 255, 0) if swapped else (255, 165, 0)
        label = "LIVE" if swapped else "No face"
        fps_text = f"{1000/max(elapsed_ms,1):.0f}fps"

        # Small status bar at top
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (180, 28), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"{label} | {fps_text}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return display

    except Exception as e:
        if _frame_count <= 10 or _frame_count % 60 == 0:
            print(f"[ERROR] Frame #{_frame_count}: {e}")
            import traceback
            traceback.print_exc()
        return frame


def create_live_interface():
    """
    Create the Live Preview tab as a standalone gr.Interface.

    KEY INSIGHT: gr.Interface(live=True) with gr.Image(streaming=True)
    is the ONLY reliable way for continuous webcam streaming in Gradio 4.x.
    It must NOT be nested inside gr.Blocks — it must be a standalone Interface.
    """
    # Build Interface kwargs — handle Gradio 4.x vs 5.x flagging param
    import inspect
    iface_params = inspect.signature(gr.Interface.__init__).parameters
    iface_kwargs = dict(
        fn=process_webcam_frame,
        inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Input"),
        outputs=gr.Image(label="Face Swap Output", type="numpy"),
        live=True,
        title="🎭 Live Face Swap",
        description=(
            "**Real-time face swap from your webcam!**\n\n"
            "1. First go to **Setup** tab and load source face images\n"
            "2. Allow webcam access when prompted\n"
            "3. Face swap happens automatically on every frame\n\n"
            "💡 *Disable HD Enhancement in Setup for faster FPS*"
        ),
    )
    # Gradio 5.x uses flagging_mode, Gradio 4.x uses allow_flagging
    if "flagging_mode" in iface_params:
        iface_kwargs["flagging_mode"] = "never"
    elif "allow_flagging" in iface_params:
        iface_kwargs["allow_flagging"] = "never"

    live_app = gr.Interface(**iface_kwargs)
    return live_app


# =====================================================
# TAB 3: IMAGE SWAP
# =====================================================


def process_image_swap(source_images, target_image, enhance: bool = True):
    """Swap face in a single image."""
    try:
        engine = initialize_engine()
        source_paths = normalize_source_paths(source_images)
        if not source_paths:
            return None, "❌ Upload source face images"
        if target_image is None:
            return None, "❌ Upload target image"

        result = engine.swap_face(
            source_paths=source_paths,
            target_path=target_image,
            enhance=enhance,
            swap_all=False,
        )
        if result is not None:
            output_path = str(OUTPUT_DIR / f"result_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, "✅ Face swap completed!"
        else:
            return None, "❌ No faces detected in target image"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


def create_image_swap_interface():
    """Create the Image Swap tab using gr.Blocks."""
    with gr.Blocks() as image_app:
        gr.Markdown(
            """
## 🖼️ Image Face Swap
Upload a source face and a target image to swap faces.
"""
        )
        with gr.Row():
            with gr.Column():
                source_imgs = gr.File(
                    label="📸 Source Face Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
            with gr.Column():
                target_image = gr.Image(
                    label="🎯 Target Image (face to replace)",
                    type="filepath",
                )

        img_enhance = gr.Checkbox(label="HD Enhancement", value=True)
        swap_btn = gr.Button("🔄 Swap Face", variant="primary", size="lg")

        with gr.Row():
            swap_output = gr.Image(label="✨ Result")
        swap_status = gr.Textbox(label="Status", interactive=False)

        swap_btn.click(
            fn=process_image_swap,
            inputs=[source_imgs, target_image, img_enhance],
            outputs=[swap_output, swap_status],
        )

    return image_app


# =====================================================
# TAB 4: VIDEO SWAP
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
            return None, "❌ Upload source face images"
        if target_video is None:
            return None, "❌ Upload target video"

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
            return output_path, "✅ Video processed!"
        else:
            return None, "❌ Video processing failed"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


def create_video_swap_interface():
    """Create the Video Swap tab using gr.Blocks."""
    with gr.Blocks() as video_app:
        gr.Markdown(
            """
## 🎬 Video Face Swap
Upload a source face and a target video to swap faces frame-by-frame.
"""
        )
        source_imgs = gr.File(
            label="📸 Source Face Images",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        )
        target_video = gr.Video(label="🎯 Target Video")
        with gr.Row():
            vid_enhance = gr.Checkbox(label="HD Enhancement", value=False)
            vid_mouth = gr.Checkbox(label="Lip Sync", value=True)
            vid_sharp = gr.Slider(
                label="Sharpness", minimum=0.0, maximum=1.0, value=0.3, step=0.1,
            )
        vid_btn = gr.Button("🔄 Process Video", variant="primary", size="lg")
        vid_output = gr.Video(label="✨ Result")
        vid_status = gr.Textbox(label="Status", interactive=False)

        vid_btn.click(
            fn=process_video_swap,
            inputs=[source_imgs, target_video, vid_enhance, vid_mouth, vid_sharp],
            outputs=[vid_output, vid_status],
        )

    return video_app


# =====================================================
# TAB 5: FACE DETECTION
# =====================================================


def detect_faces_preview(image: str) -> Tuple[Optional[str], str]:
    """Detect and preview faces in an image."""
    try:
        engine = initialize_engine()
        if not image:
            return None, "❌ Please upload an image"
        result, face_count = engine.detect_and_draw_faces(image)
        if result is not None:
            output_path = str(OUTPUT_DIR / f"faces_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, f"✅ Detected {face_count} face(s)"
        else:
            return None, "❌ No faces detected"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_face_detection_interface():
    """Create the Face Detection tab using gr.Blocks."""
    with gr.Blocks() as detect_app:
        gr.Markdown(
            """
## 🔍 Face Detection
Upload an image to detect and visualize all faces found.
"""
        )
        detect_input = gr.Image(label="Upload Image", type="filepath")
        detect_btn = gr.Button("🔍 Detect Faces", variant="primary")
        detect_output = gr.Image(label="Detected Faces")
        detect_status = gr.Textbox(label="Status", interactive=False)

        detect_btn.click(
            fn=detect_faces_preview,
            inputs=[detect_input],
            outputs=[detect_output, detect_status],
        )

    return detect_app


# =====================================================
# MAIN UI — Combine everything with TabbedInterface
# =====================================================


def create_ui():
    """
    Create the complete Gradio application.

    Uses gr.TabbedInterface to combine:
    - Setup (load faces, settings)
    - Live Preview (continuous webcam streaming) — standalone gr.Interface
    - Image Swap
    - Video Swap
    - Face Detection

    This architecture works on ALL Gradio versions (4.x and 5.x)
    because the live webcam Interface is NOT nested inside Blocks.
    """
    setup_app = create_setup_interface()
    live_app = create_live_interface()
    image_app = create_image_swap_interface()
    video_app = create_video_swap_interface()
    detect_app = create_face_detection_interface()

    app = gr.TabbedInterface(
        interface_list=[setup_app, live_app, image_app, video_app, detect_app],
        tab_names=["⚙️ Setup", "🎭 Live Preview", "🖼️ Image Swap", "🎬 Video Swap", "🔍 Face Detection"],
        title="PlasticVision Pro — Real-Time Face Swap",
    )

    return app


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    server_mode = is_server()

    print("=" * 60)
    print("🎭 PlasticVision Pro - Face Swap")
    print("=" * 60)
    print(f"Mode:   {'🌐 SERVER' if server_mode else '💻 LOCAL'}")
    print(f"GPU:    {get_gpu_status()}")
    print(f"Models: {MODELS_DIR}")
    print("=" * 60)

    app = create_ui()
    app.queue(max_size=20)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=server_mode,
        show_error=True,
    )
