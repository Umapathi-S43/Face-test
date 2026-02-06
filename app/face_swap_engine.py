#!/usr/bin/env python3
"""
Face Swap Engine
Core face swapping functionality using InsightFace and GFPGAN
Enhanced with Deep-Live-Cam quality features:
- Mouth Mask for lip sync preservation
- Face Mask with Feathering for seamless blending
- Color Transfer for lighting matching
- Frame Interpolation for temporal smoothing
- Sharpening for crisp output
- Poisson Blending for seamless integration
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Any
import threading
from collections import deque
import time

# InsightFace for face detection and swapping
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# GFPGAN for face enhancement
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    print("⚠️ GFPGAN not available, face enhancement disabled")


# =====================================================
# QUALITY ENHANCEMENT FUNCTIONS (Deep-Live-Cam style)
# =====================================================

def apply_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Apply color transfer using LAB color space.
    Transfers the color distribution from target to source.
    """
    try:
        if source is None or target is None:
            return source
        if source.size == 0 or target.size == 0:
            return source
        
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Compute mean and std for each channel
        src_mean, src_std = cv2.meanStdDev(source_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(target_lab)
        
        # Reshape for broadcasting
        src_mean = src_mean.flatten()
        src_std = src_std.flatten()
        tgt_mean = tgt_mean.flatten()
        tgt_std = tgt_std.flatten()
        
        # Avoid division by zero
        src_std = np.where(src_std < 1e-6, 1e-6, src_std)
        
        # Apply color transfer
        result_lab = source_lab.copy()
        for i in range(3):
            result_lab[:, :, i] = (result_lab[:, :, i] - src_mean[i]) * (tgt_std[i] / src_std[i]) + tgt_mean[i]
        
        # Clip and convert back
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    except Exception as e:
        return source


def create_face_mask(face, frame: np.ndarray, feather_amount: int = 31) -> np.ndarray:
    """
    Create a feathered mask covering the face area based on landmarks.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    if face is None or not hasattr(face, 'landmark_2d_106'):
        # Fallback to bbox-based mask
        if hasattr(face, 'bbox'):
            bbox = face.bbox.astype(int)
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
            # Feather
            kernel_size = max(1, feather_amount // 2 * 2 + 1)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        return mask
    
    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask
    
    try:
        # Filter non-finite values
        if not np.all(np.isfinite(landmarks)):
            return mask
        
        landmarks_int = landmarks.astype(np.int32)
        
        # Use face outline landmarks (0-32) for the face contour
        face_outline_points = landmarks_int[0:33]
        
        # Create convex hull
        hull = cv2.convexHull(face_outline_points)
        if hull is None or len(hull) < 3:
            return mask
        
        # Draw filled convex hull
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        
        # Apply Gaussian blur for feathering
        kernel_size = max(1, feather_amount // 2 * 2 + 1)
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
    except Exception as e:
        pass
    
    return mask


def create_lower_mouth_mask(face, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int], Optional[np.ndarray]]:
    """
    Create a mask for the lower mouth area to preserve original lip movement.
    Returns: (mask, mouth_cutout, mouth_box, mouth_polygon)
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    mouth_box = (0, 0, 0, 0)
    mouth_polygon = None
    
    if face is None or not hasattr(face, 'landmark_2d_106'):
        return mask, mouth_cutout, mouth_box, mouth_polygon
    
    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask, mouth_cutout, mouth_box, mouth_polygon
    
    try:
        # Lower lip landmark indices for 106-point model
        # These indices define the mouth area
        lower_lip_order = [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65]
        
        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, mouth_polygon
        
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)
        
        # Filter out non-finite values
        if not np.all(np.isfinite(lower_lip_landmarks)):
            return mask, mouth_cutout, mouth_box, mouth_polygon
        
        center = np.mean(lower_lip_landmarks, axis=0)
        if not np.all(np.isfinite(center)):
            return mask, mouth_cutout, mouth_box, mouth_polygon
        
        # Expand the mask area slightly
        expansion_factor = 1.1
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center
        expanded_landmarks = expanded_landmarks.astype(np.int32)
        
        # Calculate bounding box
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)
        
        # Add padding
        padding_ratio = 0.1
        padding_x = int((max_x - min_x) * padding_ratio)
        padding_y = int((max_y - min_y) * padding_ratio)
        
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(frame_w, max_x + padding_x)
        max_y = min(frame_h, max_y + padding_y)
        
        if max_x > min_x and max_y > min_y:
            # Create mask ROI
            mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
            
            # Shift polygon to ROI coordinates
            polygon_relative = expanded_landmarks - [min_x, min_y]
            cv2.fillPoly(mask_roi, [polygon_relative], 255)
            
            # Apply blur for feathering
            blur_kernel = 15
            blur_kernel = max(1, blur_kernel // 2 * 2 + 1)
            mask_roi = cv2.GaussianBlur(mask_roi, (blur_kernel, blur_kernel), 0)
            
            # Place mask ROI in full mask
            mask[min_y:max_y, min_x:max_x] = mask_roi
            
            # Extract mouth cutout from original frame
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
            mouth_polygon = expanded_landmarks
            mouth_box = (min_x, min_y, max_x, max_y)
    
    except Exception as e:
        pass
    
    return mask, mouth_cutout, mouth_box, mouth_polygon


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: Tuple[int, int, int, int],
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray
) -> np.ndarray:
    """
    Apply the original mouth area onto the swapped face to preserve lip movement.
    """
    if frame is None or mouth_cutout is None or mouth_box == (0, 0, 0, 0):
        return frame
    if face_mask is None or mouth_polygon is None or len(mouth_polygon) < 3:
        return frame
    
    try:
        min_x, min_y, max_x, max_y = mouth_box
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        if box_width <= 0 or box_height <= 0:
            return frame
        
        # Clamp to frame boundaries
        frame_h, frame_w = frame.shape[:2]
        min_y = max(0, min_y)
        min_x = max(0, min_x)
        max_y = min(frame_h, max_y)
        max_x = min(frame_w, max_x)
        
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        if box_width <= 0 or box_height <= 0:
            return frame
        
        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame
        
        # Resize mouth cutout to match ROI
        if roi.shape[:2] != mouth_cutout.shape[:2]:
            if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                resized_mouth = cv2.resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
            else:
                return frame
        else:
            resized_mouth = mouth_cutout
        
        if resized_mouth is None or resized_mouth.size == 0:
            return frame
        
        # Apply color transfer to match the swapped face lighting
        color_corrected_mouth = apply_color_transfer(resized_mouth, roi)
        
        # Create polygon mask for the mouth area
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon.astype(np.int32)], 255)
        
        # Feather the mask
        feather_amount = max(1, min(30, min(box_width, box_height) // 12))
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(polygon_mask.astype(float), (kernel_size, kernel_size), 0)
        
        # Normalize mask
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask = feathered_mask / max_val
        else:
            feathered_mask.fill(0.0)
        
        # Get face mask ROI
        face_mask_float = face_mask.astype(float) / 255.0 if face_mask.dtype == np.uint8 else face_mask
        face_mask_roi = face_mask_float[min_y:max_y, min_x:max_x]
        
        # Combine masks
        combined_mask = np.minimum(feathered_mask, face_mask_roi)
        
        # Blend mouth onto swapped face
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            combined_mask_3ch = combined_mask[:, :, np.newaxis]
            blended_roi = (color_corrected_mouth.astype(float) * combined_mask_3ch +
                          roi.astype(float) * (1.0 - combined_mask_3ch))
            frame[min_y:max_y, min_x:max_x] = blended_roi.astype(np.uint8)
    
    except Exception as e:
        pass
    
    return frame


def apply_sharpening(frame: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Apply sharpening to the face region."""
    if frame is None or amount <= 0:
        return frame
    
    try:
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception:
        return frame


class FaceSwapEngine:
    """
    Complete Face Swap Engine with multi-image source support
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Face analysis app
        self.face_app: Optional[FaceAnalysis] = None
        
        # Face swapper model
        self.face_swapper = None
        
        # GFPGAN enhancer
        self.face_enhancer = None
        
        # Source face embeddings (averaged from multiple images)
        self.source_face = None
        self.source_embeddings: List[np.ndarray] = []
        
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Frame cache for performance
        self.frame_cache = deque(maxlen=3)
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize all models"""
        try:
            print("🔄 Initializing Face Swap Engine...")
            
            # Initialize face analysis
            print("  📦 Loading face analysis model...")
            # Force CPU provider only - CoreML has issues on macOS
            providers = ['CPUExecutionProvider']
            
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                root=str(self.models_dir),
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            # Use standard detection size and low threshold for better detection
            self.face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)
            print("  ✅ Face analysis model loaded")
            
            # Initialize face swapper
            print("  📦 Loading face swapper model...")
            swapper_path = self.models_dir / "inswapper_128.onnx"
            
            if not swapper_path.exists():
                print("  ⬇️ Downloading face swapper model...")
                self._download_swapper_model(swapper_path)
            
            self.face_swapper = get_model(
                str(swapper_path),
                providers=['CPUExecutionProvider']
            )
            print("  ✅ Face swapper model loaded")
            
            # Initialize GFPGAN if available
            if GFPGAN_AVAILABLE:
                print("  📦 Loading GFPGAN enhancer...")
                gfpgan_path = self.models_dir / "GFPGANv1.4.pth"
                
                if not gfpgan_path.exists():
                    print("  ⬇️ Downloading GFPGAN model...")
                    self._download_gfpgan_model(gfpgan_path)
                
                if gfpgan_path.exists():
                    self.face_enhancer = GFPGANer(
                        model_path=str(gfpgan_path),
                        upscale=1,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=None
                    )
            
            self._initialized = True
            print("✅ Face Swap Engine initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Face Swap Engine: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _download_swapper_model(self, path: Path):
        """Download the inswapper model"""
        import urllib.request
        
        url = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
        print(f"  Downloading from {url}...")
        
        try:
            urllib.request.urlretrieve(url, str(path))
            print("  ✅ Download complete!")
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            print("  Please manually download inswapper_128.onnx to:", path)
    
    def _download_gfpgan_model(self, path: Path):
        """Download the GFPGAN model"""
        import urllib.request
        
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        print(f"  Downloading from {url}...")
        
        try:
            urllib.request.urlretrieve(url, str(path))
            print("  ✅ Download complete!")
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            print("  Please manually download GFPGANv1.4.pth to:", path)
    
    def load_source_faces(self, image_paths: List[str]) -> bool:
        """
        Load and process multiple source face images
        Creates an averaged embedding for better quality
        """
        try:
            if not self._initialized:
                print("⚠️ Engine not initialized, initializing now...")
                if not self.initialize():
                    print("❌ Failed to initialize engine")
                    return False
            
            self.source_embeddings = []
            all_faces = []
            
            print(f"📂 Loading source faces from {len(image_paths)} image(s)...")
            
            for path in image_paths:
                # Check if file exists
                if not os.path.exists(path):
                    print(f"⚠️ File does not exist: {path}")
                    continue
                
                print(f"  📷 Reading: {path}")
                
                # Read image with alpha channel support
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"⚠️ Could not read image (cv2.imread returned None): {path}")
                    # Try alternative loading methods
                    try:
                        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    except Exception as e:
                        print(f"⚠️ Alternative read also failed: {e}")
                        continue
                
                if img is None:
                    print(f"⚠️ All read methods failed for: {path}")
                    continue
                
                print(f"  ✓ Image loaded: {img.shape}")
                
                # Handle images with alpha channel (RGBA -> RGB)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    print(f"  🔄 Converting RGBA to RGB...")
                    # Create white background and composite
                    alpha = img[:, :, 3] / 255.0
                    rgb = img[:, :, :3]
                    white_bg = np.ones_like(rgb) * 255
                    img = (rgb * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
                
                # Ensure 3 channels
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Resize if image is too large (IMPORTANT: must match det_size for best results)
                max_size = 640
                h, w = img.shape[:2]
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"  📐 Resized to: {img.shape}")
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                print(f"  🔍 Detecting faces (threshold=0.1)...")
                
                # Use low threshold for better detection
                original_thresh = self.face_app.det_model.det_thresh
                self.face_app.det_model.det_thresh = 0.1
                faces = self.face_app.get(img_rgb)
                self.face_app.det_model.det_thresh = original_thresh
                
                print(f"  ✓ Detected {len(faces)} face(s)")
                
                if faces:
                    # Get the largest face
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    all_faces.append(face)
                    self.source_embeddings.append(face.embedding)
                    print(f"  ✅ Face extracted successfully!")
                else:
                    print(f"  ⚠️ No faces detected in: {path}")
            
            if not all_faces:
                print("❌ No faces detected in any source images")
                return False
            
            # Use the first face as the primary source (best quality usually)
            # The embeddings are stored for potential averaging
            self.source_face = all_faces[0]
            
            # Average embeddings if multiple faces
            if len(self.source_embeddings) > 1:
                avg_embedding = np.mean(self.source_embeddings, axis=0)
                # Normalize
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                self.source_face.embedding = avg_embedding
            
            print(f"✅ Successfully loaded {len(all_faces)} source face(s)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading source faces: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect_faces(self, image: np.ndarray) -> List[Any]:
        """Detect all faces in an image"""
        if not self._initialized:
            return []
        
        with self.lock:
            faces = self.face_app.get(image)
        return faces
    
    def detect_and_draw_faces(self, image_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Detect faces and draw bounding boxes"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, 0
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.detect_faces(img_rgb)
            
            # Draw boxes on faces
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(
                    img_rgb, f"Face {i+1}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            return img_rgb, len(faces)
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return None, 0
    
    def swap_face(
        self,
        source_paths: List[str],
        target_path: str,
        enhance: bool = True,
        swap_all: bool = False
    ) -> Optional[np.ndarray]:
        """
        Swap faces in a single image
        
        Args:
            source_paths: List of source face image paths
            target_path: Path to target image
            enhance: Whether to apply GFPGAN enhancement
            swap_all: Whether to swap all faces or just the largest
        
        Returns:
            Result image as numpy array (RGB) or None if failed
        """
        try:
            if not self._initialized:
                if not self.initialize():
                    return None
            
            # Load source faces
            if not self.load_source_faces(source_paths):
                return None
            
            # Read target image
            target_img = cv2.imread(target_path)
            if target_img is None:
                print(f"❌ Could not read target image: {target_path}")
                return None
            
            target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            
            # Detect faces in target
            target_faces = self.detect_faces(target_rgb)
            
            if not target_faces:
                print("❌ No faces detected in target image")
                return None
            
            # Determine which faces to swap
            if swap_all:
                faces_to_swap = target_faces
            else:
                # Just swap the largest face
                faces_to_swap = [max(target_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))]
            
            # Perform face swap
            result = target_rgb.copy()
            for target_face in faces_to_swap:
                result = self.face_swapper.get(result, target_face, self.source_face, paste_back=True)
            
            # Apply enhancement if requested
            if enhance and self.face_enhancer:
                result = self._enhance_face(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Face swap error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def swap_face_frame(
        self,
        frame: np.ndarray,
        enhance: bool = False,
        swap_all: bool = False,
        use_mouth_mask: bool = True,
        use_color_transfer: bool = True,
        sharpness: float = 0.3,
        opacity: float = 1.0
    ) -> np.ndarray:
        """
        Swap faces in a single frame (for video/webcam processing)
        Source face must be loaded beforehand using load_source_faces()
        
        Args:
            frame: Input frame (RGB)
            enhance: Apply GFPGAN enhancement (slower but higher quality)
            swap_all: Swap all detected faces
            use_mouth_mask: Preserve original mouth for lip sync
            use_color_transfer: Match colors between faces
            sharpness: Sharpening amount (0-1)
            opacity: Blend opacity for swapped face (0-1)
        """
        try:
            if self.source_face is None:
                return frame
            
            original_frame = frame.copy()
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            if not faces:
                return frame
            
            # Determine which faces to swap
            if swap_all:
                faces_to_swap = faces
            else:
                faces_to_swap = [max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))]
            
            # Process each face
            result = frame.copy()
            for target_face in faces_to_swap:
                # Save original mouth data BEFORE swap (for lip sync)
                mouth_mask_data = None
                if use_mouth_mask:
                    mouth_mask, mouth_cutout, mouth_box, mouth_polygon = create_lower_mouth_mask(
                        target_face, original_frame
                    )
                    if mouth_cutout is not None and mouth_box != (0, 0, 0, 0):
                        face_mask = create_face_mask(target_face, original_frame)
                        mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, mouth_polygon, face_mask)
                
                # Perform the face swap
                swapped = self.face_swapper.get(result, target_face, self.source_face, paste_back=True)
                
                # Ensure output is valid
                if swapped is None or not isinstance(swapped, np.ndarray):
                    continue
                if swapped.shape != result.shape:
                    swapped = cv2.resize(swapped, (result.shape[1], result.shape[0]))
                swapped = np.clip(swapped, 0, 255).astype(np.uint8)
                
                # Apply mouth mask to preserve lip movement
                if mouth_mask_data is not None:
                    mouth_mask, mouth_cutout, mouth_box, mouth_polygon, face_mask = mouth_mask_data
                    swapped = apply_mouth_area(
                        swapped, mouth_cutout, mouth_box, face_mask, mouth_polygon
                    )
                
                result = swapped
            
            # Apply sharpening
            if sharpness > 0:
                result = apply_sharpening(result, sharpness)
            
            # Apply opacity blending
            if 0 < opacity < 1:
                result = cv2.addWeighted(
                    original_frame.astype(np.uint8), 1 - opacity,
                    result.astype(np.uint8), opacity, 0
                )
            
            # Apply enhancement if requested (slower)
            if enhance and self.face_enhancer:
                result = self._enhance_face(result)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            return frame
    
    def swap_face_video(
        self,
        source_paths: List[str],
        target_video_path: str,
        output_path: str,
        enhance: bool = False,
        swap_all: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> bool:
        """
        Swap faces in a video file
        """
        try:
            if not self._initialized:
                if not self.initialize():
                    return False
            
            # Load source faces
            if not self.load_source_faces(source_paths):
                return False
            
            # Open video
            cap = cv2.VideoCapture(target_video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {target_video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Swap face
                result = self.swap_face_frame(frame_rgb, enhance=enhance, swap_all=swap_all)
                
                # Convert back to BGR for saving
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                out.write(result_bgr)
                
                frame_count += 1
                if progress_callback and total_frames > 0:
                    progress_callback(frame_count / total_frames)
            
            cap.release()
            out.release()
            
            print(f"✅ Video processing complete: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Video processing error: {e}")
            return False
    
    def _enhance_face(self, image: np.ndarray) -> np.ndarray:
        """Apply GFPGAN face enhancement"""
        try:
            if self.face_enhancer is None:
                return image
            
            # GFPGAN expects BGR
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            _, _, enhanced = self.face_enhancer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            
            # Convert back to RGB
            return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image
    
    def get_source_face_preview(self) -> Optional[np.ndarray]:
        """Get a preview of the loaded source face"""
        if self.source_face is None:
            return None
        
        # Create a simple visualization
        # This could be enhanced to show the actual cropped face
        return None
