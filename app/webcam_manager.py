#!/usr/bin/env python3
"""
Webcam Manager
Real-time webcam face swap with virtual camera output
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, List, Any
from collections import deque

# Try to import pyvirtualcam for virtual camera output
try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False
    print("⚠️ pyvirtualcam not available, virtual camera output disabled")


class WebcamManager:
    """
    Manages real-time webcam capture and face swap processing
    """
    
    def __init__(
        self,
        engine: Any,
        source_paths: List[str],
        camera_index: int = 0,
        enhance: bool = False,
        swap_all: bool = False,
        target_fps: int = 30
    ):
        self.engine = engine
        self.source_paths = source_paths
        self.camera_index = camera_index
        self.enhance = enhance
        self.swap_all = swap_all
        self.target_fps = target_fps
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Virtual camera
        self.virtual_cam = None
        
        # Threading
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        
        # Frame buffers
        self.raw_frame: Optional[np.ndarray] = None
        self.processed_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # FPS tracking
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        
        # Preview window name
        self.window_name = "Face Swap Preview"
    
    def start(self) -> bool:
        """Start webcam capture and processing"""
        try:
            # Load source faces
            if not self.engine.load_source_faces(self.source_paths):
                print("❌ Failed to load source faces")
                return False
            
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"❌ Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual dimensions
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Camera opened: {width}x{height}")
            
            # Try to initialize virtual camera
            if PYVIRTUALCAM_AVAILABLE:
                try:
                    self.virtual_cam = pyvirtualcam.Camera(
                        width=width,
                        height=height,
                        fps=self.target_fps,
                        fmt=pyvirtualcam.PixelFormat.RGB
                    )
                    print(f"📺 Virtual camera started: {self.virtual_cam.device}")
                except Exception as e:
                    print(f"⚠️ Could not start virtual camera: {e}")
                    self.virtual_cam = None
            
            # Start processing
            self.is_running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.process_thread.start()
            
            # Start preview window in main context
            self._start_preview()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start webcam: {e}")
            self.stop()
            return False
    
    def _capture_loop(self):
        """Continuously capture frames from webcam"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.frame_lock:
                        self.raw_frame = frame_rgb
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
    
    def _process_loop(self):
        """Process frames with face swap"""
        while self.is_running:
            try:
                # Get raw frame
                with self.frame_lock:
                    frame = self.raw_frame
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                start_time = time.time()
                
                # Process frame
                result = self.engine.swap_face_frame(
                    frame,
                    enhance=self.enhance,
                    swap_all=self.swap_all
                )
                
                # Update processed frame
                with self.frame_lock:
                    self.processed_frame = result
                
                # Send to virtual camera
                if self.virtual_cam:
                    try:
                        self.virtual_cam.send(result)
                        self.virtual_cam.sleep_until_next_frame()
                    except Exception as e:
                        pass
                
                # Track FPS
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 0:
                    avg_time = sum(self.frame_times) / len(self.frame_times)
                    self.fps = 1.0 / avg_time if avg_time > 0 else 0
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def _start_preview(self):
        """Start the preview window"""
        print("📺 Starting preview window...")
        print("   Press 'q' to quit")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        while self.is_running:
            try:
                with self.frame_lock:
                    frame = self.processed_frame
                
                if frame is not None:
                    # Convert RGB to BGR for OpenCV display
                    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Add FPS overlay
                    fps_text = f"FPS: {self.fps:.1f}"
                    cv2.putText(
                        display, fps_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )
                    
                    cv2.imshow(self.window_name, display)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Preview error: {e}")
                time.sleep(0.1)
        
        self.stop()
    
    def stop(self):
        """Stop webcam capture and processing"""
        print("⏹️ Stopping webcam...")
        self.is_running = False
        
        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        # Release resources
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.virtual_cam:
            self.virtual_cam.close()
            self.virtual_cam = None
        
        cv2.destroyAllWindows()
        print("✅ Webcam stopped")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current processed frame"""
        with self.frame_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps


class VirtualCameraOutput:
    """
    Standalone virtual camera output for OBS integration
    """
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.is_running = False
    
    def start(self) -> bool:
        """Start virtual camera"""
        if not PYVIRTUALCAM_AVAILABLE:
            print("❌ pyvirtualcam not available")
            return False
        
        try:
            self.cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.RGB
            )
            self.is_running = True
            print(f"📺 Virtual camera started: {self.cam.device}")
            return True
        except Exception as e:
            print(f"❌ Could not start virtual camera: {e}")
            return False
    
    def send_frame(self, frame: np.ndarray):
        """Send a frame to the virtual camera"""
        if self.cam and self.is_running:
            # Resize if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.cam.send(frame)
    
    def stop(self):
        """Stop virtual camera"""
        if self.cam:
            self.cam.close()
            self.cam = None
        self.is_running = False
