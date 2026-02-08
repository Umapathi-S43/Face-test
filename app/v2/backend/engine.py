#!/usr/bin/env python3
"""
FaceSwapEngine v2 — Core face swap engine
Ported from v1 face_swap_engine.py with all quality features intact.

Models: InsightFace buffalo_l + inswapper_128.onnx + GFPGANv1.4.pth
Quality: Mouth mask, color transfer, face mask feathering, sharpening, GFPGAN
"""

import os
import cv2
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Any
import threading
import time
import urllib.request

import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except Exception as _gfpgan_err:
    GFPGAN_AVAILABLE = False
    print(f"⚠️ GFPGAN not available — HD enhancement disabled ({_gfpgan_err})")


# ═══════════════════════════════════════════════════════
# QUALITY FUNCTIONS (ported from v1 — Deep-Live-Cam style)
# ═══════════════════════════════════════════════════════

def apply_color_transfer(source: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Transfer color distribution from target to source using LAB color space.
    If mask is provided, computes LAB stats only from masked (face) pixels
    instead of the entire frame — critical for accurate skin tone matching
    when background dominates the image.
    """
    try:
        if source is None or target is None or source.size == 0 or target.size == 0:
            return source
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        if mask is not None and mask.shape[:2] == source.shape[:2]:
            # Compute stats from face pixels only (mask > 128 = face region)
            face_pixels = mask > 128
            if np.count_nonzero(face_pixels) > 100:  # Need enough pixels for stable stats
                src_mean = np.array([source_lab[:, :, i][face_pixels].mean() for i in range(3)])
                src_std = np.array([max(source_lab[:, :, i][face_pixels].std(), 1e-6) for i in range(3)])
                tgt_mean = np.array([target_lab[:, :, i][face_pixels].mean() for i in range(3)])
                tgt_std = np.array([max(target_lab[:, :, i][face_pixels].std(), 1e-6) for i in range(3)])
            else:
                # Not enough face pixels, fall back to global stats
                src_mean, src_std = cv2.meanStdDev(source_lab)
                tgt_mean, tgt_std = cv2.meanStdDev(target_lab)
                src_mean, src_std = src_mean.flatten(), src_std.flatten()
                tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()
                src_std = np.where(src_std < 1e-6, 1e-6, src_std)
        else:
            src_mean, src_std = cv2.meanStdDev(source_lab)
            tgt_mean, tgt_std = cv2.meanStdDev(target_lab)
            src_mean, src_std = src_mean.flatten(), src_std.flatten()
            tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()
            src_std = np.where(src_std < 1e-6, 1e-6, src_std)

        result_lab = source_lab.copy()
        for i in range(3):
            result_lab[:, :, i] = (result_lab[:, :, i] - src_mean[i]) * (tgt_std[i] / src_std[i]) + tgt_mean[i]
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return source


def create_face_mask(face, frame: np.ndarray, feather_amount: int = 0, expand: float = 0.10) -> np.ndarray:
    """
    Create a feathered mask covering the face area based on landmarks.
    feather_amount: 0 = auto-proportional to face size (~6% of face width).
    expand: expand convex hull outward by this fraction (0.10 = 10%) for
            better coverage on profile/turned faces.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if face is None:
        return mask
    if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
        if hasattr(face, 'bbox'):
            bbox = face.bbox.astype(int)
            face_w = bbox[2] - bbox[0]
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
            fa = feather_amount if feather_amount > 0 else max(5, int(face_w * 0.06))
            ks = max(1, fa // 2 * 2 + 1)
            mask = cv2.GaussianBlur(mask, (ks, ks), 0)
        return mask
    landmarks = face.landmark_2d_106
    if not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask
    try:
        if not np.all(np.isfinite(landmarks)):
            return mask
        # Use ALL 106 landmarks for convex hull — covers full face including forehead
        pts = landmarks.astype(np.float32)
        hull = cv2.convexHull(pts)
        if hull is None or len(hull) < 3:
            return mask
        # Expand hull outward by `expand` fraction from centroid
        # Prevents tight clipping on profile/angled faces
        if expand > 0:
            hull_pts = hull.reshape(-1, 2).astype(np.float32)
            centroid = hull_pts.mean(axis=0)
            hull_pts = centroid + (hull_pts - centroid) * (1.0 + expand)
            hull = hull_pts.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        # Auto-proportional feather: ~6% of face width for natural blending
        # Small faces get small feather, large faces get large feather
        face_w = int(landmarks[:, 0].max() - landmarks[:, 0].min())
        fa = feather_amount if feather_amount > 0 else max(5, int(face_w * 0.06))
        ks = max(1, fa // 2 * 2 + 1)
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)
    except Exception:
        pass
    return mask


def create_lower_mouth_mask(
    face, frame: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int], Optional[np.ndarray]]:
    """Create a mask for the lower mouth area to preserve original lip movement."""
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
        lower_lip_order = [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65]
        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, mouth_polygon
        lower_lip_pts = landmarks[lower_lip_order].astype(np.float32)
        if not np.all(np.isfinite(lower_lip_pts)):
            return mask, mouth_cutout, mouth_box, mouth_polygon
        center = np.mean(lower_lip_pts, axis=0)
        if not np.all(np.isfinite(center)):
            return mask, mouth_cutout, mouth_box, mouth_polygon
        # 1.3x expansion captures full mouth opening + chin area (1.1x was cutting lips)
        expanded = ((lower_lip_pts - center) * 1.3 + center).astype(np.int32)
        min_x, min_y = np.min(expanded, axis=0)
        max_x, max_y = np.max(expanded, axis=0)
        pad_x = int((max_x - min_x) * 0.15)
        pad_y = int((max_y - min_y) * 0.15)
        fh, fw = frame.shape[:2]
        min_x = max(0, min_x - pad_x)
        min_y = max(0, min_y - pad_y)
        max_x = min(fw, max_x + pad_x)
        max_y = min(fh, max_y + pad_y)
        if max_x > min_x and max_y > min_y:
            roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
            rel = expanded - [min_x, min_y]
            cv2.fillPoly(roi, [rel], 255)
            # Larger feather (25px) for invisible mouth-to-face blending
            bk = max(1, 25 // 2 * 2 + 1)
            roi = cv2.GaussianBlur(roi, (bk, bk), 0)
            mask[min_y:max_y, min_x:max_x] = roi
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
            mouth_polygon = expanded
            mouth_box = (min_x, min_y, max_x, max_y)
    except Exception:
        pass
    return mask, mouth_cutout, mouth_box, mouth_polygon


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: Tuple[int, int, int, int],
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    """Blend the original mouth area onto the swapped face for lip sync."""
    if frame is None or mouth_cutout is None or mouth_box == (0, 0, 0, 0):
        return frame
    if face_mask is None or mouth_polygon is None or len(mouth_polygon) < 3:
        return frame
    try:
        min_x, min_y, max_x, max_y = mouth_box
        fh, fw = frame.shape[:2]
        min_y, min_x = max(0, min_y), max(0, min_x)
        max_y, max_x = min(fh, max_y), min(fw, max_x)
        bw, bh = max_x - min_x, max_y - min_y
        if bw <= 0 or bh <= 0:
            return frame
        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame
        if roi.shape[:2] != mouth_cutout.shape[:2]:
            if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                resized = cv2.resize(mouth_cutout, (bw, bh), interpolation=cv2.INTER_LINEAR)
            else:
                return frame
        else:
            resized = mouth_cutout
        if resized is None or resized.size == 0:
            return frame
        color_corrected = apply_color_transfer(resized, roi)
        pmask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adj = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(pmask, [adj.astype(np.int32)], 255)
        feather = max(1, min(30, min(bw, bh) // 12))
        ks = 2 * feather + 1
        fmask = cv2.GaussianBlur(pmask.astype(float), (ks, ks), 0)
        mx = fmask.max()
        if mx > 1e-6:
            fmask /= mx
        else:
            fmask.fill(0.0)
        # Use a thresholded (hard-edged) face mask for mouth region so mouth
        # restoration doesn't fade at the feathered boundary of the face mask.
        # The mouth is always well inside the face, so a threshold >= 0.5 keeps
        # it fully opaque where it matters.
        fm_float = face_mask.astype(float) / 255.0 if face_mask.dtype == np.uint8 else face_mask
        fm_roi = fm_float[min_y:max_y, min_x:max_x]
        fm_roi_hard = np.where(fm_roi >= 0.5, 1.0, fm_roi * 2.0)  # Boost interior to 1.0
        combined = np.minimum(fmask, fm_roi_hard)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            c3 = combined[:, :, np.newaxis]
            blended = color_corrected.astype(float) * c3 + roi.astype(float) * (1.0 - c3)
            frame[min_y:max_y, min_x:max_x] = blended.astype(np.uint8)
    except Exception:
        pass
    return frame


def apply_sharpening(frame: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Apply unsharp-mask sharpening."""
    if frame is None or amount <= 0:
        return frame
    try:
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception:
        return frame


# ═══════════════════════════════════════════════════════
# FACE SWAP ENGINE
# ═══════════════════════════════════════════════════════

class FaceSwapEngine:
    """
    Complete face swap engine with all quality features.
    Thread-safe singleton — shared across all WebSocket connections.
    """

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.face_app: Optional[FaceAnalysis] = None
        self.face_app_live: Optional[FaceAnalysis] = None  # 320x320 for fast live detection
        self.face_swapper = None
        self.face_enhancer = None
        self._has_cuda = False

        # Per-session source face storage
        # Key: session_id, Value: (source_face, source_embeddings, timestamp)
        self._session_faces: dict = {}
        self._session_lock = threading.Lock()
        self._session_ttl = 3600  # 1 hour TTL for session face data

        # Global source face (for non-session calls)
        self.source_face = None
        self.source_embeddings: List[np.ndarray] = []

        self.lock = threading.Lock()
        self._live_lock = threading.Lock()  # Separate lock for live detector (no contention with quality path)
        self._initialized = False

        # Settings
        self.settings = {
            "mouth_mask": True,
            "sharpness": 0.3,
            "enhance": False,
            "opacity": 1.0,
            "swap_all": False,
        }

    # ───────────────────────────────────────
    # Initialization
    # ───────────────────────────────────────

    def initialize(self) -> bool:
        """Initialize all AI models. Auto-detects GPU."""
        try:
            print("🔄 Initializing Face Swap Engine...")
            import onnxruntime as ort
            available = ort.get_available_providers()
            print(f"  ONNX providers: {available}")

            if 'CUDAExecutionProvider' in available:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    }),
                    'CPUExecutionProvider',
                ]
                self._has_cuda = True
                print("  🚀 Using CUDA GPU")
            else:
                providers = ['CPUExecutionProvider']
                self._has_cuda = False
                print("  💻 Using CPU")

            # Face analysis (buffalo_l)
            print("  📦 Loading face analysis (buffalo_l)...")
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                root=str(self.models_dir),
                providers=providers,
                allowed_modules=['detection', 'recognition'],
            )
            # Use 640x640 for high-quality (image/video swap)
            self.face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

            # Create a second analyzer for live streaming with 320x320
            # 320x320 is ~4x faster than 640x640 for face detection
            print("  📦 Loading fast face detector (320x320 for live)...")
            self.face_app_live = FaceAnalysis(
                name='buffalo_l',
                root=str(self.models_dir),
                providers=providers,
                allowed_modules=['detection', 'recognition'],
            )
            self.face_app_live.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
            print("  ✅ Face analysis ready")

            # Face swapper (inswapper_128)
            # Prefer FP16 model on GPU (2x faster, half memory)
            print("  📦 Loading face swapper (inswapper_128)...")
            fp16_path = self.models_dir / "inswapper_128_fp16.onnx"
            fp32_path = self.models_dir / "inswapper_128.onnx"

            if self._has_cuda and fp16_path.exists():
                swapper_path = fp16_path
                print("  ⚡ Using FP16 model (2x faster on GPU)")
            elif fp32_path.exists():
                swapper_path = fp32_path
            else:
                swapper_path = fp32_path
                self._download_model(
                    "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
                    swapper_path,
                    "inswapper_128.onnx",
                )
            self.face_swapper = get_model(str(swapper_path), providers=providers)
            print("  ✅ Face swapper ready")

            # GFPGAN (optional)
            if GFPGAN_AVAILABLE:
                print("  📦 Loading GFPGAN enhancer...")
                gfpgan_path = self.models_dir / "GFPGANv1.4.pth"
                if not gfpgan_path.exists():
                    self._download_model(
                        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                        gfpgan_path,
                        "GFPGANv1.4.pth",
                    )
                if gfpgan_path.exists():
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.face_enhancer = GFPGANer(
                        model_path=str(gfpgan_path),
                        upscale=1, arch='clean', channel_multiplier=2,
                        bg_upsampler=None, device=device,
                    )
                    print(f"  ✅ GFPGAN ready ({device})")

            self._initialized = True
            print("✅ Engine initialized!")
            return True
        except Exception as e:
            print(f"❌ Engine init failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def _download_model(self, url: str, path: Path, name: str):
        """Download a model file with progress."""
        print(f"  ⬇️ Downloading {name}...")
        try:
            def hook(count, block, total):
                pct = int(count * block * 100 / total) if total > 0 else 0
                print(f"\r  Downloading {name}: {pct}%", end="", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(path), reporthook=hook)
            print(f"\n  ✅ Downloaded {name}")
        except Exception as e:
            print(f"\n  ❌ Download failed: {e}")

    # ───────────────────────────────────────
    # GPU Info
    # ───────────────────────────────────────

    def get_gpu_status(self) -> str:
        """Return a human-readable GPU status string."""
        try:
            import onnxruntime as ort
            provs = ort.get_available_providers()
            if "CUDAExecutionProvider" in provs:
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
                    pass
                return "🟢 GPU: CUDA (via ONNX)"
            if "CoreMLExecutionProvider" in provs:
                return "🟡 Apple Silicon (CPU mode)"
        except Exception:
            pass
        return "🔴 CPU Mode"

    # ───────────────────────────────────────
    # Source Face Loading
    # ───────────────────────────────────────

    def load_source_faces(self, image_paths: List[str], session_id: Optional[str] = None) -> bool:
        """
        Load source face(s) from image files.
        If session_id given, stores per-session. Otherwise stores globally.
        Accepts any image format OpenCV supports (JPEG, PNG, WebP, BMP, TIFF, etc.)
        """
        try:
            if not self._initialized:
                if not self.initialize():
                    return False

            embeddings = []
            all_faces = []

            print(f"📂 Loading {len(image_paths)} source face(s)...")

            for path in image_paths:
                if not os.path.exists(path):
                    print(f"  ⚠️ File not found: {path}")
                    continue

                # Read image — supports any format (JPEG, PNG, WebP, BMP, TIFF, etc.)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    try:
                        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    except Exception:
                        continue
                if img is None:
                    print(f"  ⚠️ Cannot read: {path}")
                    continue

                # RGBA → RGB
                if len(img.shape) == 3 and img.shape[2] == 4:
                    alpha = img[:, :, 3] / 255.0
                    rgb = img[:, :, :3]
                    white = np.ones_like(rgb) * 255
                    img = (rgb * alpha[:, :, np.newaxis] + white * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

                # Grayscale → BGR
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Resize large images
                h, w = img.shape[:2]
                if max(h, w) > 640:
                    scale = 640 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                # Detect faces — InsightFace expects BGR input
                orig_thresh = self.face_app.det_model.det_thresh
                self.face_app.det_model.det_thresh = 0.1
                faces = self.face_app.get(img)  # BGR input
                self.face_app.det_model.det_thresh = orig_thresh

                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    all_faces.append(face)
                    embeddings.append(face.embedding)
                    print(f"  ✅ Face extracted from: {os.path.basename(path)}")
                else:
                    print(f"  ⚠️ No face in: {os.path.basename(path)}")

            if not all_faces:
                print("❌ No faces detected in any source image")
                return False

            # Use first face, average embeddings if multiple
            source_face = all_faces[0]
            if len(embeddings) > 1:
                avg = np.mean(embeddings, axis=0)
                source_face.embedding = avg / np.linalg.norm(avg)

            # Store per-session or globally
            if session_id:
                with self._session_lock:
                    self._session_faces[session_id] = (source_face, embeddings, time.time())
                    # Purge expired sessions (older than TTL)
                    self._purge_expired_sessions()
            else:
                self.source_face = source_face
                self.source_embeddings = embeddings

            print(f"✅ Loaded {len(all_faces)} source face(s)")
            return True
        except Exception as e:
            print(f"❌ Error loading source faces: {e}")
            import traceback; traceback.print_exc()
            return False

    def get_source_face(self, session_id: Optional[str] = None):
        """Get source face for a session or global."""
        if session_id:
            with self._session_lock:
                data = self._session_faces.get(session_id)
                if data:
                    # Update access time (touch)
                    self._session_faces[session_id] = (data[0], data[1], time.time())
                    return data[0]
                return self.source_face
        return self.source_face

    def has_source_face(self, session_id: Optional[str] = None) -> bool:
        """Check if source face is loaded."""
        return self.get_source_face(session_id) is not None

    def clear_session(self, session_id: str):
        """Remove session face data."""
        with self._session_lock:
            self._session_faces.pop(session_id, None)

    def _purge_expired_sessions(self):
        """Remove sessions older than TTL. Call with _session_lock held."""
        now = time.time()
        expired = [sid for sid, data in self._session_faces.items()
                   if len(data) >= 3 and (now - data[2]) > self._session_ttl]
        for sid in expired:
            del self._session_faces[sid]
        if expired:
            print(f"🧹 Purged {len(expired)} expired session(s)")

    # ───────────────────────────────────────
    # Face Detection
    # ───────────────────────────────────────

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect all faces. Uses 640x640 det_size. Input: BGR (native OpenCV format)."""
        if not self._initialized:
            return []
        with self.lock:
            return self.face_app.get(image)

    def detect_faces_live(self, image: np.ndarray) -> list:
        """Fast face detection for live streaming. Uses 320x320 det_size (~4x faster). Input: BGR."""
        if not self._initialized:
            return []
        with self._live_lock:
            return self.face_app_live.get(image)

    def detect_and_draw_faces(self, image_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Detect faces and draw bounding boxes. Returns (BGR image, count)."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, 0
            # InsightFace expects BGR — detect directly
            faces = self.detect_faces(img)
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, f"Face {i+1}", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return img, len(faces)  # Returns BGR
        except Exception as e:
            print(f"Detection error: {e}")
            return None, 0

    # ───────────────────────────────────────
    # Single Image Swap
    # ───────────────────────────────────────

    def swap_face(
        self,
        source_paths: List[str],
        target_path: str,
        enhance: bool = True,
        swap_all: bool = False,
        session_id: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Swap faces in a single target image with full quality pipeline. Returns BGR or None."""
        try:
            if not self._initialized and not self.initialize():
                return None
            if not self.load_source_faces(source_paths, session_id=session_id):
                return None
            img = cv2.imread(target_path)
            if img is None:
                return None
            # InsightFace expects BGR — no conversion needed
            faces = self.detect_faces(img)
            if not faces:
                return None
            src = self.get_source_face(session_id)
            targets = faces if swap_all else [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

            original = img.copy()
            result = img.copy()
            settings = self.settings

            for tf in targets:
                # Save original mouth BEFORE swap (for lip sync)
                mouth_data = None
                if settings["mouth_mask"]:
                    face_mask = create_face_mask(tf, original)
                    mm, mc, mb, mp = create_lower_mouth_mask(tf, original)
                    if mc is not None and mb != (0, 0, 0, 0):
                        mouth_data = (mm, mc, mb, mp, face_mask)

                # INSwapper: paste_back=True handles affine warp + blending internally
                swapped = self.face_swapper.get(result, tf, src, paste_back=True)
                if swapped is None or not isinstance(swapped, np.ndarray):
                    continue
                if swapped.shape != result.shape:
                    swapped = cv2.resize(swapped, (result.shape[1], result.shape[0]))
                swapped = np.clip(swapped, 0, 255).astype(np.uint8)

                # Restore original mouth for lip sync
                if mouth_data:
                    mm, mc, mb, mp, fm = mouth_data
                    swapped = apply_mouth_area(swapped, mc, mb, fm, mp)

                result = swapped

            # Sharpening
            if settings["sharpness"] > 0:
                result = apply_sharpening(result, settings["sharpness"])

            # Opacity blending
            if 0 < settings["opacity"] < 1:
                result = cv2.addWeighted(
                    original.astype(np.uint8), 1 - settings["opacity"],
                    result.astype(np.uint8), settings["opacity"], 0,
                )

            # HD Enhancement
            if enhance and self.face_enhancer:
                result = self._enhance_face(result)

            return result
        except Exception as e:
            print(f"❌ swap_face error: {e}")
            import traceback; traceback.print_exc()
            return None

    # ───────────────────────────────────────
    # Frame-by-frame swap (webcam / video)
    # ───────────────────────────────────────

    def swap_face_frame(
        self,
        frame_bgr: np.ndarray,
        enhance: bool = False,
        swap_all: bool = False,
        use_mouth_mask: bool = True,
        use_color_transfer: bool = True,
        sharpness: float = 0.3,
        opacity: float = 1.0,
        session_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Swap faces in a single frame (BGR in, BGR out).
        Source face must be pre-loaded via load_source_faces().
        Uses 640x640 detection (higher quality, for video processing).
        """
        try:
            src = self.get_source_face(session_id)
            if src is None:
                return frame_bgr

            original = frame_bgr.copy()
            # InsightFace expects BGR — no conversion needed
            faces = self.detect_faces(frame_bgr)
            if not faces:
                return frame_bgr

            targets = faces if swap_all else [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

            result = frame_bgr.copy()
            for tf in targets:
                # Save mouth BEFORE swap for lip sync preservation
                mouth_data = None
                if use_mouth_mask:
                    face_mask = create_face_mask(tf, original)
                    mm, mc, mb, mp = create_lower_mouth_mask(tf, original)
                    if mc is not None and mb != (0, 0, 0, 0):
                        mouth_data = (mm, mc, mb, mp, face_mask)

                # INSwapper: paste_back=True handles affine warp + blending internally
                swapped = self.face_swapper.get(result, tf, src, paste_back=True)
                if swapped is None or not isinstance(swapped, np.ndarray):
                    continue
                if swapped.shape != result.shape:
                    swapped = cv2.resize(swapped, (result.shape[1], result.shape[0]))
                swapped = np.clip(swapped, 0, 255).astype(np.uint8)

                # Restore original mouth area (lip sync) — color-correct mouth to match swapped face
                if mouth_data:
                    mm, mc, mb, mp, fm = mouth_data
                    swapped = apply_mouth_area(swapped, mc, mb, fm, mp)

                result = swapped

            if sharpness > 0:
                result = apply_sharpening(result, sharpness)

            if 0 < opacity < 1:
                result = cv2.addWeighted(
                    original.astype(np.uint8), 1 - opacity,
                    result.astype(np.uint8), opacity, 0,
                )

            if enhance and self.face_enhancer:
                result = self._enhance_face(result)

            return result.astype(np.uint8)
        except Exception:
            return frame_bgr

    def swap_face_frame_live(
        self,
        frame_bgr: np.ndarray,
        use_mouth_mask: bool = True,
        sharpness: float = 0.3,
        opacity: float = 1.0,
        swap_all: bool = False,
        session_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        OPTIMIZED live frame swap — works entirely in BGR (native OpenCV format).
        InsightFace detection and swap both expect BGR input and produce BGR output.
        No color conversions needed at all!
        Uses 320x320 detection for ~4x faster face finding.
        No GFPGAN (too slow for live). Input/Output: BGR.
        """
        try:
            src = self.get_source_face(session_id)
            if src is None:
                return frame_bgr

            original_bgr = frame_bgr.copy()

            # FAST detection (320x320) — InsightFace expects BGR
            faces = self.detect_faces_live(frame_bgr)
            if not faces:
                return frame_bgr

            targets = faces if swap_all else [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

            result = frame_bgr
            for tf in targets:
                # Save mouth BEFORE swap for lip sync preservation
                mouth_data = None
                if use_mouth_mask:
                    face_mask = create_face_mask(tf, original_bgr)
                    mm, mc, mb, mp = create_lower_mouth_mask(tf, original_bgr)
                    if mc is not None and mb != (0, 0, 0, 0):
                        mouth_data = (mm, mc, mb, mp, face_mask)

                # INSwapper: paste_back=True handles affine warp + blending internally
                swapped = self.face_swapper.get(result, tf, src, paste_back=True)
                if swapped is None or not isinstance(swapped, np.ndarray):
                    continue
                if swapped.shape != result.shape:
                    swapped = cv2.resize(swapped, (result.shape[1], result.shape[0]))
                swapped = np.clip(swapped, 0, 255).astype(np.uint8)

                # Restore original mouth area (lip sync) — color-correct mouth to match swapped face
                if mouth_data:
                    mm, mc, mb, mp, fm = mouth_data
                    swapped = apply_mouth_area(swapped, mc, mb, fm, mp)
                result = swapped

            if sharpness > 0:
                result = apply_sharpening(result, sharpness)

            if 0 < opacity < 1 and original_bgr is not None:
                result = cv2.addWeighted(
                    original_bgr.astype(np.uint8), 1 - opacity,
                    result.astype(np.uint8), opacity, 0,
                )

            return result.astype(np.uint8)
        except Exception:
            return frame_bgr

    # ───────────────────────────────────────
    # Video Swap
    # ───────────────────────────────────────

    def swap_face_video(
        self,
        source_paths: List[str],
        target_video_path: str,
        output_path: str,
        enhance: bool = False,
        swap_all: bool = False,
        use_mouth_mask: bool = True,
        sharpness: float = 0.3,
        progress_callback: Optional[Callable[[float], None]] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Swap faces in a video file. Returns True on success."""
        try:
            if not self._initialized and not self.initialize():
                return False
            if not self.load_source_faces(source_paths, session_id=session_id):
                return False
            cap = cv2.VideoCapture(target_video_path)
            if not cap.isOpened():
                return False
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # frame is already BGR from cv2.VideoCapture — swap_face_frame expects BGR
                result = self.swap_face_frame(
                    frame, enhance=enhance, swap_all=swap_all,
                    use_mouth_mask=use_mouth_mask, sharpness=sharpness,
                    session_id=session_id,
                )
                out.write(result)  # result is BGR, cv2.VideoWriter expects BGR
                count += 1
                if progress_callback and total > 0:
                    progress_callback(count / total)
            cap.release()
            out.release()
            print(f"✅ Video done: {output_path} ({count} frames)")

            # ── Audio preservation + H.264 re-encoding ──
            self._mux_audio(target_video_path, output_path)

            return True
        except Exception as e:
            print(f"❌ Video error: {e}")
            return False

    @staticmethod
    def _mux_audio(original_video: str, swapped_video: str) -> None:
        """Copy audio track from original video into the swapped output using ffmpeg."""
        if not shutil.which("ffmpeg"):
            print("⚠️  ffmpeg not found — audio track skipped")
            return
        try:
            # Check if original actually has an audio stream
            probe = subprocess.run(
                ["ffmpeg", "-i", original_video, "-hide_banner"],
                capture_output=True, text=True, timeout=10,
            )
            if "Audio:" not in probe.stderr:
                print("ℹ️  Original video has no audio — re-encoding to H.264 only")
                # Still need to re-encode from mp4v → H.264 for browser playback
                tmp_out = swapped_video + ".h264.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", swapped_video,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-an",
                    "-movflags", "+faststart",
                    tmp_out,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if r.returncode == 0 and os.path.exists(tmp_out):
                    os.replace(tmp_out, swapped_video)
                    print("✅ Video re-encoded to H.264")
                else:
                    if os.path.exists(tmp_out):
                        os.remove(tmp_out)
                return

            # Mux: take video from swapped, audio from original → temp file
            # Re-encode video to H.264 (mp4v from OpenCV isn't browser-compatible)
            # and audio to AAC for universal browser playback
            tmp_out = swapped_video + ".muxed.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-i", swapped_video,      # video source (swapped)
                "-i", original_video,      # audio source (original)
                "-c:v", "libx264",         # re-encode to H.264 for browser compat
                "-preset", "fast",         # balance speed vs quality
                "-crf", "18",              # high quality (lower = better, 18 = visually lossless)
                "-pix_fmt", "yuv420p",     # required for browser playback
                "-c:a", "aac",             # re-encode audio to AAC
                "-b:a", "192k",
                "-map", "0:v:0",           # video from first input
                "-map", "1:a:0",           # audio from second input
                "-shortest",               # stop when shorter stream ends
                "-movflags", "+faststart", # web-friendly MP4
                tmp_out,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(tmp_out):
                os.replace(tmp_out, swapped_video)  # atomic replace
                print("✅ Audio preserved from original video")
            else:
                print(f"⚠️  Audio mux failed: {result.stderr[-200:] if result.stderr else 'unknown'}")
                # Clean up temp file if it exists
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
        except subprocess.TimeoutExpired:
            print("⚠️  Audio mux timed out — skipped")
        except Exception as e:
            print(f"⚠️  Audio mux error: {e}")

    # ───────────────────────────────────────
    # GFPGAN Enhancement
    # ───────────────────────────────────────

    def _enhance_face(self, image: np.ndarray) -> np.ndarray:
        """Apply GFPGAN face enhancement. Input/output: BGR."""
        try:
            if self.face_enhancer is None:
                return image
            # GFPGAN expects BGR input (OpenCV native)
            _, _, enhanced = self.face_enhancer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True,
            )
            return enhanced
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image
