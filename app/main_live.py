#!/usr/bin/env python3
"""
💉 PlasticVision Pro - Live Real-Time Face Swap
GPU-Optimized for Ultra-Low Latency (10-100ms)

Features:
- Browser webcam → GPU processing → Live output
- Multi-GPU support (2x RTX 5090)
- Sub-100ms latency processing
- Real-time face swap streaming
- Natural lip movement preservation
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import time
from typing import Optional, List, Tuple
from collections import deque

# GPU optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import GPU-optimized engine
try:
    from face_swap_engine_gpu import FaceSwapEngineGPU as FaceSwapEngine
    print("🚀 Using GPU-optimized engine")
except ImportError:
    from face_swap_engine import FaceSwapEngine
    print("💻 Using standard engine")

# Global instances
face_swap_engine: Optional[FaceSwapEngine] = None
source_faces_loaded = False
current_settings = {
    'use_mouth_mask': True,
    'sharpness': 0.3,
    'enhance': False
}

# Performance tracking
frame_times = deque(maxlen=60)
last_fps = 0.0

# Paths
APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
MODELS_DIR = APP_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


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
    """Initialize the face swap engine"""
    global face_swap_engine
    if face_swap_engine is None:
        face_swap_engine = FaceSwapEngine(models_dir=str(MODELS_DIR))
        face_swap_engine.initialize()
    return face_swap_engine


def get_gpu_status() -> str:
    """Get GPU status"""
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
        return "💻 CPU Mode"
    except:
        return "💻 CPU Mode"


def load_source_faces(source_images) -> str:
    """Load source faces for face swap"""
    global source_faces_loaded, face_swap_engine
    
    source_paths = normalize_source_paths(source_images)
    
    if not source_paths:
        source_faces_loaded = False
        return "❌ Please upload source face images"
    
    try:
        engine = initialize_engine()
        
        if engine.load_source_faces(source_paths):
            source_faces_loaded = True
            return f"✅ Loaded {len(source_paths)} source face(s) - Ready for live preview!"
        else:
            source_faces_loaded = False
            return "❌ Failed to detect faces in uploaded images"
    except Exception as e:
        source_faces_loaded = False
        return f"❌ Error: {str(e)}"


def update_settings(mouth_mask: bool, sharpness: float, enhance: bool):
    """Update processing settings"""
    global current_settings
    current_settings['use_mouth_mask'] = mouth_mask
    current_settings['sharpness'] = sharpness
    current_settings['enhance'] = enhance
    return f"⚙️ Settings updated: Lip Sync={mouth_mask}, Sharpness={sharpness}, HD={enhance}"


def process_webcam_frame(frame):
    """
    Process a single frame from browser webcam
    This is called for each frame captured from user's browser
    Target: <100ms latency per frame
    """
    global face_swap_engine, source_faces_loaded, frame_times, last_fps
    
    if frame is None:
        return None
    
    # If source faces not loaded, return original with overlay
    if not source_faces_loaded or face_swap_engine is None:
        # Add "Upload faces first" message
        display = frame.copy()
        cv2.putText(display, "Upload source faces first!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return display
    
    start_time = time.time()
    
    try:
        # Convert to RGB if needed (Gradio sends RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = frame
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # GPU-accelerated face swap
        result = face_swap_engine.swap_face_frame(
            frame_rgb,
            enhance=current_settings['enhance'],
            swap_all=False,
            use_mouth_mask=current_settings['use_mouth_mask'],
            use_color_transfer=True,
            sharpness=current_settings['sharpness'],
            opacity=1.0
        )
        
        # Calculate processing time
        process_time = time.time() - start_time
        frame_times.append(process_time)
        
        # Update FPS every 10 frames
        if len(frame_times) >= 10:
            avg_time = sum(frame_times) / len(frame_times)
            last_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Add performance overlay
        display = result.copy()
        latency_ms = process_time * 1000
        
        # Status bar background
        cv2.rectangle(display, (0, 0), (250, 70), (0, 0, 0), -1)
        
        # FPS and latency
        color = (0, 255, 0) if latency_ms < 100 else (0, 165, 255) if latency_ms < 200 else (0, 0, 255)
        cv2.putText(display, f"FPS: {last_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Latency: {latency_ms:.0f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Face swap status
        face_detected = result is not frame_rgb
        if face_detected:
            cv2.putText(display, "LIVE", (180, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return frame


def process_image_swap(source_images, target_image, enhance: bool = True):
    """Process single image face swap"""
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
            swap_all=False
        )
        
        if result is not None:
            output_path = str(OUTPUT_DIR / f"result_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, "✅ Face swap completed!"
        else:
            return None, "❌ Face swap failed"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def process_video_swap(source_images, target_video, enhance: bool = False, 
                       mouth_mask: bool = True, sharpness: float = 0.3,
                       progress=gr.Progress()):
    """Process video face swap with progress"""
    try:
        engine = initialize_engine()
        source_paths = normalize_source_paths(source_images)
        
        if not source_paths:
            return None, "❌ Upload source face images"
        
        if target_video is None:
            return None, "❌ Upload target video"
        
        output_path = str(OUTPUT_DIR / f"video_{int(time.time())}.mp4")
        
        result = engine.swap_face_video(
            source_paths=source_paths,
            target_video_path=target_video,
            output_path=output_path,
            enhance=enhance,
            swap_all=False,
            use_mouth_mask=mouth_mask,
            sharpness=sharpness,
            progress_callback=lambda p: progress(p, desc=f"Processing: {int(p*100)}%")
        )
        
        if result:
            fps = engine.get_fps() if hasattr(engine, 'get_fps') else 0
            return output_path, f"✅ Video complete! ({fps:.1f} FPS)"
        else:
            return None, "❌ Video processing failed"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# =====================================================
# GRADIO UI
# =====================================================

def create_ui():
    """Create the Gradio interface with live webcam streaming"""
    
    gpu_status = get_gpu_status()
    
    css = """
    .gradio-container { max-width: 1400px !important; margin: auto; }
    .header { 
        text-align: center; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;
    }
    .header h1 { color: white !important; margin: 0; font-size: 2.5em; }
    .header p { margin: 5px 0; opacity: 0.9; }
    .live-badge { 
        background: #ff4444; color: white; padding: 3px 10px; 
        border-radius: 20px; font-weight: bold; animation: pulse 1s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    .status-box { 
        background: #1a1a2e; padding: 15px; border-radius: 10px; 
        color: #00ff88; font-family: monospace; 
    }
    """
    
    with gr.Blocks(title="PlasticVision Pro - Live", theme=gr.themes.Soft(), css=css) as app:
        
        # Header
        gr.HTML(f"""
        <div class="header">
            <h1>💉 PlasticVision Pro</h1>
            <p>Real-Time Face Swap with GPU Acceleration</p>
            <p style="font-size: 0.9em;">{gpu_status}</p>
            <p><span class="live-badge">🔴 LIVE STREAMING</span></p>
        </div>
        """)
        
        with gr.Tabs():
            # ========== LIVE PREVIEW TAB ==========
            with gr.TabItem("📹 Live Preview (Browser Webcam)", id=1):
                gr.Markdown("""
                ### 🎥 Real-Time Face Swap
                Your browser webcam → GPU Processing → Live Output
                
                **Steps:**
                1. Upload source face images (the face you want to appear)
                2. Click "Load Faces" 
                3. Allow camera access when prompted
                4. See real-time face swap!
                """)
                
                with gr.Row():
                    # Left panel - Controls
                    with gr.Column(scale=1):
                        source_upload = gr.File(
                            label="🎯 Upload Source Faces (1-10 images)",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                        load_btn = gr.Button("📥 Load Faces", variant="primary", size="lg")
                        load_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Upload faces and click 'Load Faces'",
                            elem_classes=["status-box"]
                        )
                        
                        gr.Markdown("### ⚙️ Quality Settings")
                        
                        mouth_mask = gr.Checkbox(
                            label="👄 Natural Lip Movement",
                            value=True,
                            info="Preserves original lip sync"
                        )
                        
                        sharpness = gr.Slider(
                            label="🔍 Sharpness",
                            minimum=0.0, maximum=1.0, value=0.3, step=0.1
                        )
                        
                        enhance = gr.Checkbox(
                            label="✨ HD Enhancement (slower)",
                            value=False,
                            info="Enable for best quality, disable for speed"
                        )
                        
                        apply_settings_btn = gr.Button("Apply Settings")
                        settings_status = gr.Textbox(
                            label="Settings",
                            interactive=False,
                            value="Default settings active"
                        )
                    
                    # Right panel - Live video
                    with gr.Column(scale=2):
                        gr.Markdown("### 📺 Live Output")
                        
                        # Webcam input with streaming
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Your Camera (processed in real-time)",
                            height=500,
                            mirror_webcam=True
                        )
                        
                        gr.Markdown("""
                        **💡 Performance Tips:**
                        - Target: <100ms latency for smooth real-time
                        - Disable HD Enhancement for faster processing
                        - Good lighting = better face detection
                        """)
                
                # Event handlers
                load_btn.click(
                    fn=load_source_faces,
                    inputs=[source_upload],
                    outputs=[load_status]
                )
                
                apply_settings_btn.click(
                    fn=update_settings,
                    inputs=[mouth_mask, sharpness, enhance],
                    outputs=[settings_status]
                )
                
                # Real-time webcam processing
                webcam_input.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_input],
                    outputs=[webcam_input]
                )
            
            # ========== VIDEO PROCESSING TAB ==========
            with gr.TabItem("🎬 Video Processing", id=2):
                with gr.Row():
                    with gr.Column():
                        vid_source = gr.File(
                            label="🎯 Source Face Images",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                        vid_target = gr.Video(label="📹 Target Video")
                        
                        with gr.Row():
                            vid_mouth = gr.Checkbox(label="👄 Lip Sync", value=True)
                            vid_enhance = gr.Checkbox(label="✨ HD", value=False)
                        
                        vid_sharp = gr.Slider(
                            label="Sharpness", minimum=0.0, maximum=1.0, value=0.3, step=0.1
                        )
                        
                        vid_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                    
                    with gr.Column():
                        vid_output = gr.Video(label="📹 Result")
                        vid_status = gr.Textbox(label="Status", interactive=False)
                
                vid_btn.click(
                    fn=process_video_swap,
                    inputs=[vid_source, vid_target, vid_enhance, vid_mouth, vid_sharp],
                    outputs=[vid_output, vid_status]
                )
            
            # ========== IMAGE SWAP TAB ==========
            with gr.TabItem("🖼️ Image Swap", id=3):
                with gr.Row():
                    with gr.Column():
                        img_source = gr.File(
                            label="🎯 Source Faces",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                        img_target = gr.Image(label="📷 Target Image", type="filepath")
                        
                        img_enhance = gr.Checkbox(label="✨ HD Enhancement", value=True)
                        
                        img_btn = gr.Button("🔄 Swap Faces", variant="primary")
                    
                    with gr.Column():
                        img_output = gr.Image(label="Result")
                        img_status = gr.Textbox(label="Status", interactive=False)
                
                img_btn.click(
                    fn=process_image_swap,
                    inputs=[img_source, img_target, img_enhance],
                    outputs=[img_output, img_status]
                )
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        
        **⚡ Performance:** Browser webcam streams to GPU server for real-time processing.
        Target latency: <100ms for smooth live preview.
        
        **💡 Tips:** Upload 5-10 clear face photos for best results. Good lighting helps!
        
        </center>
        """)
    
    return app


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("💉 PlasticVision Pro - Live Real-Time Face Swap")
    print("=" * 60)
    print(f"📁 Upload dir: {UPLOAD_DIR}")
    print(f"📁 Output dir: {OUTPUT_DIR}")
    print(f"🎮 {get_gpu_status()}")
    print("=" * 60)
    
    app = create_ui()
    app.queue(max_size=20)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
