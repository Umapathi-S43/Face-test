#!/usr/bin/env python3
"""
💉 PlasticVision Pro - Plastic Surgery Visualization Platform
GPU-Optimized for Real-Time Performance

Features:
- Multi-GPU support (2x RTX 5090)
- Ultra-low latency video streaming
- Adaptive FPS optimization
- Professional medical visualization
- Natural lip movement preservation
- High-quality face enhancement
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import threading
import time
from typing import Optional, List, Tuple, Any
import tempfile
import shutil
from queue import Queue, Empty
from collections import deque

# GPU optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import GPU-optimized engine (fallback to standard if not available)
try:
    from face_swap_engine_gpu import FaceSwapEngineGPU as FaceSwapEngine
    print("🚀 Using GPU-optimized engine")
except ImportError:
    from face_swap_engine import FaceSwapEngine
    print("💻 Using standard engine")

# Virtual camera support
try:
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False
    print("⚠️ pyvirtualcam not available (normal for cloud deployment)")

# Global instances
face_swap_engine: Optional[FaceSwapEngine] = None
webcam_manager = None
streaming_active = False

# Paths
APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
MODELS_DIR = APP_DIR / "models"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize_source_paths(source_images) -> List[str]:
    """Normalize source images input to list of file paths"""
    if source_images is None:
        return []
    
    source_paths = []
    if isinstance(source_images, str):
        source_paths = [source_images]
    elif isinstance(source_images, list):
        for item in source_images:
            if isinstance(item, str):
                source_paths.append(item)
            elif hasattr(item, 'name'):
                source_paths.append(item.name)
            elif isinstance(item, dict) and 'name' in item:
                source_paths.append(item['name'])
    elif hasattr(source_images, 'name'):
        source_paths = [source_images.name]
    
    return [p for p in source_paths if os.path.exists(p)]


def initialize_engine():
    """Initialize the face swap engine with GPU support"""
    global face_swap_engine
    if face_swap_engine is None:
        face_swap_engine = FaceSwapEngine(models_dir=str(MODELS_DIR))
        face_swap_engine.initialize()
    return face_swap_engine


def get_gpu_status() -> str:
    """Get GPU status for display"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                gpus.append(f"{name} ({mem}GB)")
            return f"🚀 {gpu_count}x GPU: " + ", ".join(gpus)
        else:
            return "💻 CPU Mode"
    except:
        return "💻 CPU Mode"


# =====================================================
# IMAGE PROCESSING
# =====================================================

def process_face_swap(
    source_images,
    target_image: str,
    enhance_face: bool = True,
    swap_all_faces: bool = False
) -> Tuple[str, str]:
    """Process single image face swap"""
    try:
        engine = initialize_engine()
        source_paths = normalize_source_paths(source_images)
        
        if not source_paths:
            return None, "❌ Please upload source face images"
        
        if not target_image:
            return None, "❌ Please upload a target image"
        
        result = engine.swap_face(
            source_paths=source_paths,
            target_path=target_image,
            enhance=enhance_face,
            swap_all=swap_all_faces
        )
        
        if result is not None:
            output_path = str(OUTPUT_DIR / f"result_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, "✅ Face swap completed!"
        else:
            return None, "❌ Face swap failed - check if faces are detected"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def detect_faces_preview(image: str) -> Tuple[str, str]:
    """Detect and preview faces in an image"""
    try:
        engine = initialize_engine()
        
        if not image:
            return None, "❌ Please upload an image"
        
        result, face_count = engine.detect_and_draw_faces(image)
        
        if result is not None:
            output_path = str(OUTPUT_DIR / f"detected_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, f"✅ Detected {face_count} face(s)"
        else:
            return None, "❌ No faces detected"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# =====================================================
# VIDEO PROCESSING (GPU-OPTIMIZED)
# =====================================================

def process_video_swap(
    source_images,
    target_video: str,
    enhance_face: bool = False,
    swap_all_faces: bool = False,
    use_mouth_mask: bool = True,
    sharpness: float = 0.3,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """GPU-optimized video processing"""
    try:
        engine = initialize_engine()
        source_paths = normalize_source_paths(source_images)
        
        if not source_paths:
            return None, "❌ Please upload source face images"
        
        if not target_video:
            return None, "❌ Please upload a target video"
        
        output_path = str(OUTPUT_DIR / f"video_{int(time.time())}.mp4")
        
        result = engine.swap_face_video(
            source_paths=source_paths,
            target_video_path=target_video,
            output_path=output_path,
            enhance=enhance_face,
            swap_all=swap_all_faces,
            use_mouth_mask=use_mouth_mask,
            sharpness=sharpness,
            progress_callback=lambda p: progress(p, desc=f"Processing: {int(p*100)}%")
        )
        
        if result:
            fps = engine.get_fps() if hasattr(engine, 'get_fps') else 0
            return output_path, f"✅ Video completed! (Processing FPS: {fps:.1f})"
        else:
            return None, "❌ Video processing failed"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# =====================================================
# WEBCAM STREAMING (ULTRA-OPTIMIZED)
# =====================================================

class FrameBuffer:
    """Thread-safe frame buffer for smooth streaming"""
    def __init__(self, maxsize=3):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        self.latest_frame = None
    
    def put(self, frame):
        with self.lock:
            self.buffer.append(frame)
            self.latest_frame = frame
    
    def get_latest(self):
        with self.lock:
            return self.latest_frame


def webcam_face_swap_stream(
    source_images,
    camera_index_str: str = "0 - Default Camera",
    enhance_face: bool = False,
    use_mouth_mask: bool = True,
    sharpness: float = 0.3,
    target_fps: float = 30.0
):
    """
    Ultra-optimized webcam streaming with GPU acceleration
    Uses frame buffering and adaptive FPS for smooth streaming
    """
    global webcam_manager, streaming_active
    
    # Parse camera index
    camera_index = int(camera_index_str.split(" - ")[0]) if camera_index_str else 0
    
    source_paths = normalize_source_paths(source_images)
    
    if not source_paths:
        yield None, "❌ Please upload source face images first"
        return
    
    print(f"📸 Source images: {len(source_paths)}")
    print(f"🎛️ Settings: mouth_mask={use_mouth_mask}, sharpness={sharpness}, enhance={enhance_face}")
    
    cap = None
    try:
        # Initialize engine
        print("🔄 Initializing engine...")
        engine = initialize_engine()
        
        if engine is None:
            yield None, "❌ Failed to initialize engine"
            return
        
        # Load source faces
        yield None, f"🔄 Loading {len(source_paths)} source face(s)..."
        
        if not engine.load_source_faces(source_paths):
            yield None, "❌ Failed to load source faces - ensure images contain clear faces"
            return
        
        yield None, "✅ Source faces loaded! Opening camera..."
        
        # Open camera with optimized settings
        print(f"📹 Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            yield None, f"❌ Could not open camera {camera_index}"
            return
        
        # Optimized camera settings for GPU streaming
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Faster codec
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Camera: {width}x{height} @ {actual_fps}fps")
        
        webcam_manager = cap
        streaming_active = True
        
        # Performance tracking
        frame_times = deque(maxlen=30)
        frame_count = 0
        start_time = time.time()
        fps_display = 0.0
        last_fps_update = time.time()
        
        # Adaptive frame timing
        target_frame_time = 1.0 / target_fps
        
        while streaming_active and cap is not None and cap.isOpened():
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # GPU-accelerated face swap
            process_start = time.time()
            try:
                result = engine.swap_face_frame(
                    frame_rgb,
                    enhance=enhance_face,
                    swap_all=False,
                    use_mouth_mask=use_mouth_mask,
                    use_color_transfer=True,
                    sharpness=sharpness,
                    opacity=1.0
                )
            except Exception as e:
                print(f"⚠️ Swap error: {e}")
                result = frame_rgb
            
            process_time = time.time() - process_start
            
            # Update FPS
            frame_times.append(process_time)
            frame_count += 1
            
            current_time = time.time()
            if current_time - last_fps_update >= 0.5:
                elapsed = current_time - start_time
                fps_display = frame_count / elapsed if elapsed > 0 else 0
                last_fps_update = current_time
            
            # Check if face was swapped
            face_swapped = result is not frame_rgb
            
            # Create display frame with overlays
            display_frame = result.copy()
            display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            # Status bar
            status_color = (0, 255, 0) if face_swapped else (0, 165, 255)
            cv2.rectangle(display_bgr, (0, 0), (320, 100), (0, 0, 0), -1)
            cv2.putText(display_bgr, f"FPS: {fps_display:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if face_swapped:
                cv2.putText(display_bgr, "✓ Face Swapped", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if use_mouth_mask:
                    cv2.putText(display_bgr, "Lip Sync: ON", (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(display_bgr, "Looking for face...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # GPU indicator
            gpu_status = "GPU" if hasattr(engine, 'gpu_info') and engine.gpu_info.get('cuda_available') else "CPU"
            cv2.putText(display_bgr, gpu_status, (width - 60, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            
            status_text = (
                f"{'✅ Face swap active' if face_swapped else '⚠️ No face detected'} | "
                f"FPS: {fps_display:.1f} | "
                f"Latency: {process_time*1000:.0f}ms | "
                f"{gpu_status}"
            )
            
            yield display_rgb, status_text
            
            # Adaptive frame pacing for smooth streaming
            loop_time = time.time() - loop_start
            sleep_time = max(0, target_frame_time - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        yield None, "✅ Stream stopped"
        
    except GeneratorExit:
        print("🛑 Stream generator closed")
    except Exception as e:
        print(f"❌ Stream error: {e}")
        import traceback
        traceback.print_exc()
        yield None, f"❌ Error: {str(e)}"
    finally:
        streaming_active = False
        if cap is not None:
            cap.release()
            print("📹 Camera released")


def stop_webcam_stream():
    """Stop the webcam stream"""
    global webcam_manager, streaming_active
    
    streaming_active = False
    
    if webcam_manager is not None:
        if isinstance(webcam_manager, cv2.VideoCapture):
            webcam_manager.release()
        webcam_manager = None
        return "✅ Stream stopped"
    
    return "⚠️ Stream was not running"


# =====================================================
# VIRTUAL CAMERA (FOR LOCAL DEPLOYMENT)
# =====================================================

virtual_cam_running = False
virtual_cam_thread = None

def start_virtual_camera(
    source_images,
    camera_index_str: str = "0 - Default Camera",
    enhance_face: bool = False,
    use_mouth_mask: bool = True,
    sharpness: float = 0.3
):
    """Start virtual camera for Teams/Zoom/Meet"""
    global virtual_cam_running, virtual_cam_thread, webcam_manager
    
    if not VIRTUAL_CAM_AVAILABLE:
        return "❌ pyvirtualcam not available (not needed for cloud deployment)"
    
    camera_index = int(camera_index_str.split(" - ")[0]) if camera_index_str else 0
    source_paths = normalize_source_paths(source_images)
    
    if not source_paths:
        return "❌ Please upload source face images"
    
    try:
        engine = initialize_engine()
        if not engine.load_source_faces(source_paths):
            return "❌ Failed to load source faces"
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return f"❌ Could not open camera {camera_index}"
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        webcam_manager = cap
        virtual_cam_running = True
        
        def virtual_cam_loop():
            global virtual_cam_running
            try:
                with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.RGB) as vcam:
                    print(f"📺 Virtual camera: {vcam.device}")
                    
                    while virtual_cam_running and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        result = engine.swap_face_frame(
                            frame_rgb,
                            enhance=enhance_face,
                            swap_all=False,
                            use_mouth_mask=use_mouth_mask,
                            sharpness=sharpness
                        )
                        
                        vcam.send(result)
                        vcam.sleep_until_next_frame()
                        
            except Exception as e:
                print(f"Virtual camera error: {e}")
            finally:
                cap.release()
        
        virtual_cam_thread = threading.Thread(target=virtual_cam_loop, daemon=True)
        virtual_cam_thread.start()
        
        return f"✅ Virtual camera started ({width}x{height})"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def stop_virtual_camera():
    """Stop virtual camera"""
    global virtual_cam_running, webcam_manager
    
    virtual_cam_running = False
    
    if webcam_manager is not None:
        if isinstance(webcam_manager, cv2.VideoCapture):
            webcam_manager.release()
        webcam_manager = None
    
    return "✅ Virtual camera stopped"


# =====================================================
# GRADIO UI
# =====================================================

def create_ui():
    """Create professional Gradio interface"""
    
    gpu_status = get_gpu_status()
    
    custom_css = """
    .gradio-container { 
        max-width: 1200px !important; 
        margin: auto;
    }
    .main-header { 
        text-align: center; 
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        margin-bottom: 5px;
    }
    .status-bar {
        background: #1a1a2e;
        padding: 10px;
        border-radius: 5px;
        color: #00ff88;
        font-family: monospace;
    }
    """
    
    with gr.Blocks(
        title="💉 PlasticVision Pro",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        gr.HTML(f"""
        <div class="main-header">
            <h1>💉 PlasticVision Pro</h1>
            <p style="font-size: 1.1em; margin: 0;">GPU-Accelerated Plastic Surgery Visualization</p>
            <p style="font-size: 0.9em; opacity: 0.8; margin-top: 5px;">{gpu_status}</p>
        </div>
        """)
        
        with gr.Tabs():
            # ===== LIVE PREVIEW TAB =====
            with gr.TabItem("📹 Live Preview", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        source_images = gr.File(
                            label="🎯 Upload Expected Result Photos (1-10 images)",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                        gr.Markdown("### ⚙️ Quality Settings")
                        
                        with gr.Row():
                            mouth_mask_check = gr.Checkbox(
                                label="👄 Natural Lip Movement",
                                value=True
                            )
                            webcam_enhance = gr.Checkbox(
                                label="✨ HD Enhancement",
                                value=False,
                                info="Slower but higher quality"
                            )
                        
                        sharpness_slider = gr.Slider(
                            label="🔍 Sharpness",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1
                        )
                        
                        fps_slider = gr.Slider(
                            label="🎬 Target FPS",
                            minimum=15,
                            maximum=60,
                            value=30,
                            step=5,
                            info="Higher = smoother but more GPU usage"
                        )
                        
                        with gr.Row():
                            start_btn = gr.Button(
                                "▶️ Start Live Preview",
                                variant="primary",
                                size="lg"
                            )
                            stop_btn = gr.Button(
                                "⏹️ Stop",
                                variant="stop",
                                size="lg"
                            )
                    
                    with gr.Column(scale=2):
                        webcam_status = gr.Textbox(
                            label="📊 Status",
                            interactive=False,
                            value="Upload photos and click 'Start Live Preview'",
                            elem_classes=["status-bar"]
                        )
                        
                        webcam_preview = gr.Image(
                            label="📺 Live Preview",
                            height=500
                        )
                
                # Hidden camera selector
                camera_selector = gr.Textbox(value="0 - Default Camera", visible=False)
                
                start_btn.click(
                    fn=webcam_face_swap_stream,
                    inputs=[
                        source_images,
                        camera_selector,
                        webcam_enhance,
                        mouth_mask_check,
                        sharpness_slider,
                        fps_slider
                    ],
                    outputs=[webcam_preview, webcam_status]
                )
                
                stop_btn.click(
                    fn=stop_webcam_stream,
                    outputs=[webcam_status]
                )
            
            # ===== VIDEO PROCESSING TAB =====
            with gr.TabItem("🎬 Video Processing", id=2):
                with gr.Row():
                    with gr.Column():
                        video_source = gr.File(
                            label="🎯 Upload Result Photos",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                        target_video = gr.Video(
                            label="📹 Upload Target Video"
                        )
                        
                        with gr.Row():
                            video_mouth_mask = gr.Checkbox(
                                label="👄 Natural Lips",
                                value=True
                            )
                            video_enhance = gr.Checkbox(
                                label="✨ HD Enhancement",
                                value=False
                            )
                        
                        video_sharpness = gr.Slider(
                            label="🔍 Sharpness",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1
                        )
                        
                        process_video_btn = gr.Button(
                            "🚀 Process Video",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        video_output = gr.Video(
                            label="📹 Result Video"
                        )
                        video_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                
                process_video_btn.click(
                    fn=process_video_swap,
                    inputs=[
                        video_source,
                        target_video,
                        video_enhance,
                        gr.Checkbox(value=False, visible=False),  # swap_all
                        video_mouth_mask,
                        video_sharpness
                    ],
                    outputs=[video_output, video_status]
                )
            
            # ===== IMAGE SWAP TAB =====
            with gr.TabItem("🖼️ Image Swap", id=3):
                with gr.Row():
                    with gr.Column():
                        img_source = gr.File(
                            label="🎯 Source Face Photos",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                        img_target = gr.Image(
                            label="📷 Target Image",
                            type="filepath"
                        )
                        
                        with gr.Row():
                            img_enhance = gr.Checkbox(
                                label="✨ HD Enhancement",
                                value=True
                            )
                            img_swap_all = gr.Checkbox(
                                label="Swap All Faces",
                                value=False
                            )
                        
                        process_img_btn = gr.Button(
                            "🔄 Swap Faces",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        img_output = gr.Image(
                            label="Result"
                        )
                        img_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                
                process_img_btn.click(
                    fn=process_face_swap,
                    inputs=[img_source, img_target, img_enhance, img_swap_all],
                    outputs=[img_output, img_status]
                )
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        
        **💡 Tips for Best Results:**
        - Upload 5-10 clear, front-facing photos
        - Ensure good, even lighting
        - Face the camera directly
        - Enable "Natural Lip Movement" for realistic results
        
        **⚡ Performance:**
        - GPU acceleration enabled automatically
        - Target 30 FPS for smooth preview
        - Use HD Enhancement for final outputs only
        
        </center>
        """)
    
    return app


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("=" * 50)
    print("💉 PlasticVision Pro - Starting...")
    print("=" * 50)
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎮 GPU Status: {get_gpu_status()}")
    print("=" * 50)
    
    app = create_ui()
    app.queue()  # Enable queue for handling multiple users
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Enable public URL
        show_error=True
    )
