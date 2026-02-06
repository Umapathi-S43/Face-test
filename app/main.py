#!/usr/bin/env python3
"""
🎭 Real-Time Face Swap Application
Complete Face Swap System with Web UI
Optimized for MacBook Pro M1 Pro
Features:
- Real-time webcam face swap
- Virtual camera output for Teams/Zoom/Meet
- Mouth mask for lip sync
- Color transfer for lighting matching
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

# Set environment variables for Apple Silicon optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from face_swap_engine import FaceSwapEngine
from webcam_manager import WebcamManager

# Try to import pyvirtualcam for virtual camera support
try:
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False
    print("⚠️ pyvirtualcam not available, virtual camera output disabled")

# Global instances
face_swap_engine: Optional[FaceSwapEngine] = None
webcam_manager: Optional[WebcamManager] = None

# Paths
APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
MODELS_DIR = APP_DIR / "models"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def normalize_source_paths(source_images) -> List[str]:
    """
    Normalize source images input from Gradio to a list of file paths.
    Handles various Gradio file upload formats.
    """
    if source_images is None:
        return []
    
    source_paths = []
    if isinstance(source_images, str):
        # Single file path
        source_paths = [source_images]
    elif isinstance(source_images, list):
        for item in source_images:
            if isinstance(item, str):
                source_paths.append(item)
            elif hasattr(item, 'name'):
                # Gradio UploadFile object
                source_paths.append(item.name)
            elif isinstance(item, dict) and 'name' in item:
                source_paths.append(item['name'])
    elif hasattr(source_images, 'name'):
        source_paths = [source_images.name]
    
    # Filter to only existing files
    valid_paths = [p for p in source_paths if os.path.exists(p)]
    return valid_paths


def initialize_engine():
    """Initialize the face swap engine"""
    global face_swap_engine
    if face_swap_engine is None:
        face_swap_engine = FaceSwapEngine(models_dir=str(MODELS_DIR))
        face_swap_engine.initialize()
    return face_swap_engine


def process_face_swap(
    source_images,
    target_image: str,
    enhance_face: bool = True,
    swap_all_faces: bool = False
) -> Tuple[str, str]:
    """
    Process face swap with multiple source images
    
    Args:
        source_images: List of paths to source face images (Person A)
        target_image: Path to target image (Person B)
        enhance_face: Whether to apply GFPGAN enhancement
        swap_all_faces: Whether to swap all detected faces
    
    Returns:
        Tuple of (output_path, status_message)
    """
    try:
        engine = initialize_engine()
        
        # Normalize source paths
        source_paths = normalize_source_paths(source_images)
        
        if not source_paths or len(source_paths) == 0:
            return None, "❌ Please upload at least one source face image"
        
        if not target_image:
            return None, "❌ Please upload a target image"
        
        # Process source images to create averaged face embedding
        status = f"📸 Processing {len(source_paths)} source images..."
        
        # Perform face swap
        result = engine.swap_face(
            source_paths=source_paths,
            target_path=target_image,
            enhance=enhance_face,
            swap_all=swap_all_faces
        )
        
        if result is not None:
            # Save output
            output_path = str(OUTPUT_DIR / f"swapped_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, "✅ Face swap completed successfully!"
        else:
            return None, "❌ Face swap failed. Please check if faces are detected in both images."
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def process_video_swap(
    source_images,
    target_video: str,
    enhance_face: bool = True,
    swap_all_faces: bool = False,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Process face swap on video"""
    try:
        engine = initialize_engine()
        
        # Normalize source paths
        source_paths = normalize_source_paths(source_images)
        
        if not source_paths or len(source_paths) == 0:
            return None, "❌ Please upload at least one source face image"
        
        if not target_video:
            return None, "❌ Please upload a target video"
        
        output_path = str(OUTPUT_DIR / f"swapped_video_{int(time.time())}.mp4")
        
        # Process video with progress
        result = engine.swap_face_video(
            source_paths=source_paths,
            target_video_path=target_video,
            output_path=output_path,
            enhance=enhance_face,
            swap_all=swap_all_faces,
            progress_callback=lambda p: progress(p, desc="Processing video...")
        )
        
        if result:
            return output_path, "✅ Video face swap completed!"
        else:
            return None, "❌ Video processing failed"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def start_webcam_swap(
    source_images,
    camera_index: int = 0,
    enhance_face: bool = False,
    swap_all_faces: bool = False
) -> str:
    """Start real-time webcam face swap"""
    global webcam_manager
    
    try:
        engine = initialize_engine()
        
        # Normalize source paths
        source_paths = normalize_source_paths(source_images)
        
        if not source_paths or len(source_paths) == 0:
            return "❌ Please upload at least one source face image first"
        
        # Initialize webcam manager
        webcam_manager = WebcamManager(
            engine=engine,
            source_paths=source_paths,
            camera_index=camera_index,
            enhance=enhance_face,
            swap_all=swap_all_faces
        )
        
        # Start webcam processing in background
        webcam_manager.start()
        
        return "✅ Webcam face swap started! Open OBS and capture the preview window."
        
    except Exception as e:
        return f"❌ Error starting webcam: {str(e)}"


def stop_webcam_swap() -> str:
    """Stop webcam face swap"""
    global webcam_manager
    
    if webcam_manager:
        webcam_manager.stop()
        webcam_manager = None
        return "✅ Webcam face swap stopped"
    return "⚠️ Webcam was not running"


def get_webcam_frame():
    """Get current webcam frame for display"""
    global webcam_manager
    
    if webcam_manager and webcam_manager.is_running:
        frame = webcam_manager.get_current_frame()
        if frame is not None:
            return frame
    return None


def webcam_face_swap_stream(
    source_images,
    camera_index_str: str = "0 - Default Camera",
    enhance_face: bool = False,
    use_mouth_mask: bool = True,
    sharpness: float = 0.3
):
    """Generator function for streaming webcam face swap to Gradio with quality options"""
    global webcam_manager
    
    # Parse camera index
    camera_index = int(camera_index_str.split(" - ")[0]) if camera_index_str else 0
    
    # Use the helper function to normalize paths
    source_paths = normalize_source_paths(source_images)
    
    if not source_paths or len(source_paths) == 0:
        yield None, "❌ Please upload source face images first (no valid files found)"
        return
    
    print(f"📸 Source image paths: {source_paths}")
    print(f"🎛️ Quality settings: mouth_mask={use_mouth_mask}, sharpness={sharpness}, enhance={enhance_face}")
    
    cap = None
    try:
        # Initialize engine first
        print("🔄 Initializing face swap engine...")
        engine = initialize_engine()
        
        if engine is None:
            yield None, "❌ Failed to initialize face swap engine"
            return
        
        # Load source faces with detailed feedback
        print(f"📂 Loading {len(source_paths)} source face(s)...")
        yield None, f"🔄 Loading {len(source_paths)} source face(s)..."
        
        if not engine.load_source_faces(source_paths):
            yield None, "❌ Failed to load source faces - please ensure source images contain clear, visible faces"
            return
        
        yield None, "✅ Source faces loaded! Opening camera..."
        
        # Open camera
        print(f"📹 Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            yield None, f"❌ Could not open camera {camera_index}. Please check camera permissions."
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Camera opened: {width}x{height}")
        
        # Store cap reference for stopping
        webcam_manager = cap
        
        frame_count = 0
        start_time = time.time()
        fps_update_time = time.time()
        fps_display = 0
        
        while cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply face swap with quality settings
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
                print(f"⚠️ Face swap error: {e}")
                result = frame_rgb
            
            # Calculate FPS (update every second)
            frame_count += 1
            current_time = time.time()
            if current_time - fps_update_time >= 1.0:
                elapsed = current_time - start_time
                fps_display = frame_count / elapsed if elapsed > 0 else 0
                fps_update_time = current_time
            
            # Determine if face swap was applied
            face_swapped = result is not frame_rgb
            
            if face_swapped:
                status = f"✅ Face swap active | FPS: {fps_display:.1f} | Mouth Mask: {'ON' if use_mouth_mask else 'OFF'}"
            else:
                status = f"⚠️ No face detected | FPS: {fps_display:.1f}"
            
            # Add status overlay to the frame
            display_frame = result.copy()
            display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            # Add FPS and status overlay
            cv2.putText(display_bgr, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if face_swapped:
                cv2.putText(display_bgr, "Face Swapped!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if use_mouth_mask:
                    cv2.putText(display_bgr, "Lip Sync: ON", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display_bgr, "Looking for face...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            
            yield display_rgb, status
            
            # Control frame rate for streaming
            time.sleep(0.033)  # ~30 FPS
        
        yield None, "✅ Webcam stopped"
        
    except GeneratorExit:
        print("🛑 Webcam stream generator closed")
    except Exception as e:
        print(f"❌ Webcam error: {e}")
        import traceback
        traceback.print_exc()
        yield None, f"❌ Error: {str(e)}"
    finally:
        if cap is not None:
            cap.release()
            print("📹 Camera released")


# Global virtual camera state
virtual_cam_running = False
virtual_cam_thread = None

def start_virtual_camera(
    source_images,
    camera_index_str: str = "0 - Default Camera",
    enhance_face: bool = False,
    use_mouth_mask: bool = True,
    sharpness: float = 0.3
):
    """Start virtual camera output for Teams/Zoom/Meet"""
    global virtual_cam_running, virtual_cam_thread, webcam_manager
    
    if not VIRTUAL_CAM_AVAILABLE:
        return "❌ pyvirtualcam not available. Please use OBS Virtual Camera instead."
    
    # Parse camera index
    camera_index = int(camera_index_str.split(" - ")[0]) if camera_index_str else 0
    
    # Normalize source paths
    source_paths = normalize_source_paths(source_images)
    
    if not source_paths:
        return "❌ Please upload source face images first"
    
    try:
        engine = initialize_engine()
        if not engine.load_source_faces(source_paths):
            return "❌ Failed to load source faces"
        
        # Open camera
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
                    print(f"📺 Virtual camera started: {vcam.device}")
                    
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
        
        return f"✅ Virtual camera started! Select '{width}x{height}' virtual camera in Teams/Zoom/Meet"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def stop_virtual_camera():
    """Stop virtual camera output"""
    global virtual_cam_running, webcam_manager
    
    virtual_cam_running = False
    
    if webcam_manager is not None:
        if isinstance(webcam_manager, cv2.VideoCapture):
            webcam_manager.release()
        webcam_manager = None
    
    return "✅ Virtual camera stopped"


def stop_webcam_stream():
    """Stop the webcam stream"""
    global webcam_manager
    
    if webcam_manager is not None:
        if isinstance(webcam_manager, cv2.VideoCapture):
            webcam_manager.release()
        elif hasattr(webcam_manager, 'stop'):
            webcam_manager.stop()
        webcam_manager = None
        return "✅ Webcam stopped"
    return "⚠️ Webcam was not running"


def detect_faces_preview(image: str) -> Tuple[str, str]:
    """Detect and preview faces in an image"""
    try:
        engine = initialize_engine()
        
        if not image:
            return None, "❌ Please upload an image"
        
        # Detect faces and draw boxes
        result, face_count = engine.detect_and_draw_faces(image)
        
        if result is not None:
            output_path = str(OUTPUT_DIR / f"faces_preview_{int(time.time())}.png")
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return output_path, f"✅ Detected {face_count} face(s)"
        else:
            return None, "❌ No faces detected"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_ui():
    """Create the Gradio web interface"""
    
    with gr.Blocks(
        title="🎭 Real-Time Face Swap",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .main-header { text-align: center; margin-bottom: 20px; }
        """
    ) as app:
        
        gr.Markdown("""
        # 🎭 Real-Time Face Swap Application
        
        **Optimized for MacBook Pro M1 Pro** | Powered by InsightFace + GFPGAN
        
        ---
        """)
        
        with gr.Tabs():
            # ============ IMAGE SWAP TAB ============
            with gr.TabItem("📷 Image Face Swap"):
                gr.Markdown("### Swap faces between images")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        source_images = gr.File(
                            label="📸 Source Face Images (Person A) - Upload 1-10 images",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        gr.Markdown("*Upload multiple photos of the same person for better quality*")
                        
                    with gr.Column(scale=1):
                        target_image = gr.Image(
                            label="🎯 Target Image (Person B)",
                            type="filepath"
                        )
                
                with gr.Row():
                    enhance_check = gr.Checkbox(
                        label="✨ Enhance Face (GFPGAN)",
                        value=True
                    )
                    swap_all_check = gr.Checkbox(
                        label="👥 Swap All Faces",
                        value=False
                    )
                
                swap_btn = gr.Button("🔄 Swap Faces", variant="primary", size="lg")
                
                with gr.Row():
                    output_image = gr.Image(label="📤 Result")
                    status_text = gr.Textbox(label="Status", interactive=False)
                
                swap_btn.click(
                    fn=process_face_swap,
                    inputs=[source_images, target_image, enhance_check, swap_all_check],
                    outputs=[output_image, status_text]
                )
            
            # ============ VIDEO SWAP TAB ============
            with gr.TabItem("🎬 Video Face Swap"):
                gr.Markdown("### Swap faces in a video")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_source_images = gr.File(
                            label="📸 Source Face Images (Person A)",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                    with gr.Column(scale=1):
                        target_video = gr.Video(
                            label="🎯 Target Video (Person B)"
                        )
                
                with gr.Row():
                    video_enhance_check = gr.Checkbox(
                        label="✨ Enhance Face (slower)",
                        value=False
                    )
                    video_swap_all_check = gr.Checkbox(
                        label="👥 Swap All Faces",
                        value=False
                    )
                
                video_swap_btn = gr.Button("🔄 Process Video", variant="primary", size="lg")
                
                with gr.Row():
                    output_video = gr.Video(label="📤 Result Video")
                    video_status = gr.Textbox(label="Status", interactive=False)
                
                video_swap_btn.click(
                    fn=process_video_swap,
                    inputs=[video_source_images, target_video, video_enhance_check, video_swap_all_check],
                    outputs=[output_video, video_status]
                )
            
            # ============ WEBCAM TAB ============
            with gr.TabItem("📹 Live Webcam"):
                gr.Markdown("""
                ### Real-Time Webcam Face Swap
                
                **Enhanced with Deep-Live-Cam style quality features:**
                - 🎯 **Mouth Mask** - Preserves lip movement for realistic results
                - 🎨 **Color Transfer** - Matches lighting between faces
                - ✨ **Sharpening** - Crisp, high-quality output
                
                **Instructions:**
                1. Upload source face images (person you want to look like)
                2. Adjust quality settings as needed
                3. Click "▶️ Start Face Swap"
                """)
                
                with gr.Row():
                    webcam_source_images = gr.File(
                        label="📸 Source Face Images (Person you want to look like)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath"
                    )
                
                with gr.Row():
                    camera_dropdown = gr.Dropdown(
                        label="📷 Camera",
                        choices=["0 - Default Camera", "1 - Camera 1", "2 - Camera 2"],
                        value="0 - Default Camera"
                    )
                
                gr.Markdown("### 🎛️ Quality Settings")
                
                with gr.Row():
                    mouth_mask_check = gr.Checkbox(
                        label="👄 Mouth Mask (Lip Sync)",
                        value=True,
                        info="Preserves original mouth movement for realistic lip sync"
                    )
                    webcam_enhance = gr.Checkbox(
                        label="✨ GFPGAN Enhance",
                        value=False,
                        info="Higher quality but reduces FPS significantly"
                    )
                
                with gr.Row():
                    sharpness_slider = gr.Slider(
                        label="🔍 Sharpness",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        info="Higher = sharper face (0 = off)"
                    )
                
                with gr.Row():
                    start_webcam_btn = gr.Button("▶️ Start Face Swap", variant="primary", size="lg")
                    stop_webcam_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                
                webcam_status = gr.Textbox(label="Status", interactive=False)
                webcam_preview = gr.Image(label="📺 Live Face Swap Preview", height=480)
                
                # Use generator for streaming with quality options
                start_webcam_btn.click(
                    fn=webcam_face_swap_stream,
                    inputs=[webcam_source_images, camera_dropdown, webcam_enhance, mouth_mask_check, sharpness_slider],
                    outputs=[webcam_preview, webcam_status]
                )
                
                stop_webcam_btn.click(
                    fn=stop_webcam_stream,
                    outputs=[webcam_status]
                )
            
            # ============ VIRTUAL CAMERA TAB (for Teams/Zoom/Meet) ============
            with gr.TabItem("🎥 Virtual Camera (Zoom/Teams/Meet)"):
                vc_available = "✅ Native virtual camera available!" if VIRTUAL_CAM_AVAILABLE else "⚠️ Using OBS method (pyvirtualcam not installed)"
                
                gr.Markdown(f"""
                ### 🎥 Virtual Camera for Video Calls
                
                **Status:** {vc_available}
                
                **Use face swap in video conferencing apps:**
                - 💼 Microsoft Teams
                - 🎦 Zoom
                - 📹 Google Meet
                - 🎬 OBS Studio
                - 📱 Discord
                
                **Method 1 - Native Virtual Camera (if available):**
                1. Click "Start Virtual Camera"
                2. In Teams/Zoom/Meet, select the virtual camera device
                
                **Method 2 - OBS Virtual Camera:**
                1. Install [OBS Studio](https://obsproject.com/)
                2. Add "Window Capture" source → select this preview window
                3. Start "Virtual Camera" in OBS
                4. In Teams/Zoom/Meet, select "OBS Virtual Camera"
                """)
                
                with gr.Row():
                    vc_source_images = gr.File(
                        label="📸 Source Face Images",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath"
                    )
                
                with gr.Row():
                    vc_camera_dropdown = gr.Dropdown(
                        label="📷 Input Camera",
                        choices=["0 - Default Camera", "1 - Camera 1", "2 - Camera 2"],
                        value="0 - Default Camera"
                    )
                
                gr.Markdown("### 🎛️ Quality Settings")
                
                with gr.Row():
                    vc_mouth_mask = gr.Checkbox(label="👄 Lip Sync (Mouth Mask)", value=True)
                    vc_enhance = gr.Checkbox(label="✨ Face Enhance", value=False)
                
                with gr.Row():
                    vc_sharpness = gr.Slider(label="🔍 Sharpness", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
                
                gr.Markdown("### 🎬 Virtual Camera Controls")
                
                with gr.Row():
                    vc_native_btn = gr.Button("🎥 Start Native Virtual Camera", variant="primary", size="lg")
                    vc_preview_btn = gr.Button("📺 Start Preview Only", variant="secondary", size="lg")
                    vc_stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                
                vc_status = gr.Textbox(label="Status", interactive=False, value="Ready")
                
                gr.Markdown("### 📺 Preview Window (Capture this in OBS if using Method 2)")
                vc_preview = gr.Image(label="Virtual Camera Preview", height=480)
                
                # Native virtual camera (direct output)
                vc_native_btn.click(
                    fn=start_virtual_camera,
                    inputs=[vc_source_images, vc_camera_dropdown, vc_enhance, vc_mouth_mask, vc_sharpness],
                    outputs=[vc_status]
                )
                
                # Preview only (for OBS capture)
                vc_preview_btn.click(
                    fn=webcam_face_swap_stream,
                    inputs=[vc_source_images, vc_camera_dropdown, vc_enhance, vc_mouth_mask, vc_sharpness],
                    outputs=[vc_preview, vc_status]
                )
                
                vc_stop_btn.click(
                    fn=stop_virtual_camera,
                    outputs=[vc_status]
                )
            
            # ============ FACE DETECTION TAB ============
            with gr.TabItem("🔍 Face Detection"):
                gr.Markdown("### Detect and preview faces in an image")
                
                detect_input = gr.Image(
                    label="📷 Upload Image",
                    type="filepath"
                )
                
                detect_btn = gr.Button("🔍 Detect Faces", variant="primary")
                
                with gr.Row():
                    detect_output = gr.Image(label="📤 Detected Faces")
                    detect_status = gr.Textbox(label="Status", interactive=False)
                
                detect_btn.click(
                    fn=detect_faces_preview,
                    inputs=[detect_input],
                    outputs=[detect_output, detect_status]
                )
        
        gr.Markdown("""
        ---
        ### 📋 Tips for Best Results:
        - Use **5-10 clear photos** of the source face from different angles
        - Ensure good **lighting** in both source and target images
        - For webcam: Use good lighting and face the camera directly
        - **Face Enhancement** improves quality but is slower
        
        ### ⚙️ Performance (M1 Pro):
        - Image swap: ~1-2 seconds
        - Video: ~15-20 FPS processing
        - Webcam: ~15-20 FPS real-time
        """)
    
    return app


if __name__ == "__main__":
    print("🎭 Starting Face Swap Application...")
    print("📁 Upload directory:", UPLOAD_DIR)
    print("📁 Output directory:", OUTPUT_DIR)
    
    # Create and launch the app
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
