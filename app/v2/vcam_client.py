#!/usr/bin/env python3
"""
PlasticVision Pro v2 — Virtual Camera Client

Captures webcam locally → sends frames to GPU server via WebSocket →
receives swapped frames → outputs to OBS Virtual Camera (pyvirtualcam).

The virtual camera appears as a regular webcam in Zoom, Google Meet,
Teams, Discord, FaceTime, etc.

Requirements:
  - OBS Studio 30.0+ (start Virtual Camera once to install the driver)
  - pip install pyvirtualcam websockets opencv-python numpy

Usage:
  python vcam_client.py --server http://localhost:8000 --camera 0
  python vcam_client.py --server http://your-gpu-server:8000 --session <session_id>
"""

import argparse
import asyncio
import sys
import time
import signal
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import pyvirtualcam
    VCAM_AVAILABLE = True
except ImportError:
    VCAM_AVAILABLE = False
    print("⚠️  pyvirtualcam not installed. Run: pip install pyvirtualcam")

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("⚠️  websockets not installed. Run: pip install websockets")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ═══════════════════════════════════════════════════════
# VIRTUAL CAMERA FACE SWAP CLIENT
# ═══════════════════════════════════════════════════════

class VCamClient:
    """
    Webcam → GPU Server → Virtual Camera pipeline.
    Runs entirely on the local machine, sending frames to the remote
    GPU server for face swap processing.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        jpeg_quality: int = 92,
        session_id: str = None,
        source_image: str = None,
        preview: bool = True,
        mirror: bool = False,
    ):
        self.server_url = server_url.rstrip("/")
        self.ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.session_id = session_id
        self.source_image = source_image
        self.preview = preview
        self.mirror = mirror

        # State
        self.running = False
        self.cap = None
        self.vcam = None
        self.ws = None

        # Frame tracking
        self.frame_times = deque(maxlen=60)
        self.latency_ms = 0
        self.display_fps = 0
        self.frames_sent = 0
        self.frames_received = 0

        # Latest result frame for preview
        self._result_frame = None
        self._result_lock = threading.Lock()

    def _check_server(self) -> bool:
        """Check if the GPU server is reachable."""
        if not REQUESTS_AVAILABLE:
            # Fallback: try urllib
            import urllib.request
            try:
                req = urllib.request.Request(f"{self.server_url}/status")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    import json
                    data = json.loads(resp.read())
                    print(f"  GPU: {data.get('gpu', 'unknown')}")
                    print(f"  Engine ready: {data.get('engine_ready', False)}")
                    print(f"  Faces loaded: {data.get('faces_loaded', False)}")
                    return data.get("engine_ready", False)
            except Exception as e:
                print(f"❌ Cannot reach server: {e}")
                return False

        try:
            resp = requests.get(f"{self.server_url}/status", timeout=5)
            data = resp.json()
            print(f"  GPU: {data.get('gpu', 'unknown')}")
            print(f"  Engine ready: {data.get('engine_ready', False)}")
            print(f"  Faces loaded: {data.get('faces_loaded', False)}")
            return data.get("engine_ready", False)
        except Exception as e:
            print(f"❌ Cannot reach server: {e}")
            return False

    def _upload_source_face(self, image_path: str) -> str:
        """Upload a source face image to the server. Returns session_id."""
        if not REQUESTS_AVAILABLE:
            print("⚠️  'requests' library not available for uploading. Upload faces via the web UI.")
            return None
        try:
            with open(image_path, "rb") as f:
                files = [("files", (Path(image_path).name, f, "image/jpeg"))]
                data = {}
                if self.session_id:
                    data["session_id"] = self.session_id
                resp = requests.post(
                    f"{self.server_url}/upload-source-faces",
                    files=files, data=data, timeout=30,
                )
                result = resp.json()
                if result.get("success"):
                    print(f"  ✅ {result.get('message', 'Face uploaded')}")
                    return result.get("session_id")
                else:
                    print(f"  ❌ Upload failed: {result}")
                    return None
        except Exception as e:
            print(f"  ❌ Upload error: {e}")
            return None

    def _open_camera(self) -> bool:
        """Open the webcam."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"❌ Cannot open camera {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize capture latency

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"  📹 Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

        # Update dimensions to actual
        self.width = actual_w
        self.height = actual_h
        return True

    def _open_vcam(self) -> bool:
        """Open the virtual camera output."""
        if not VCAM_AVAILABLE:
            print("⚠️  pyvirtualcam not available — virtual camera disabled")
            return False
        try:
            self.vcam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
            print(f"  📺 Virtual camera started: {self.vcam.device}")
            return True
        except Exception as e:
            print(f"⚠️  Virtual camera failed: {e}")
            print("     → Open OBS → Start Virtual Camera → restart this script")
            self.vcam = None
            return False

    async def _ws_loop(self):
        """WebSocket send/receive loop — core pipeline."""
        ws_url = f"{self.ws_url}/ws/stream"
        print(f"  🔌 Connecting WebSocket: {ws_url}")

        async with websockets.connect(
            ws_url,
            max_size=50 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            self.ws = ws
            print("  ✅ WebSocket connected")

            # Wait for server hello
            hello = await ws.recv()
            import json
            msg = json.loads(hello)
            print(f"  Server: {msg}")

            server_session = msg.get("session_id")

            # Link our session if we have one
            if self.session_id:
                await ws.send(json.dumps({
                    "action": "set_session",
                    "session_id": self.session_id,
                }))
                resp = await ws.recv()
                link_msg = json.loads(resp)
                print(f"  Session linked: {link_msg}")
            else:
                self.session_id = server_session

            # Main frame loop
            while self.running:
                t0 = time.perf_counter()

                # Capture frame
                ret, frame_bgr = self.cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                # Mirror if requested
                if self.mirror:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                # Encode as JPEG
                _, buf = cv2.imencode(
                    '.jpg', frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                jpeg_bytes = buf.tobytes()

                # Send to server
                await ws.send(jpeg_bytes)
                self.frames_sent += 1

                # Receive swapped frame
                response = await ws.recv()

                if isinstance(response, str):
                    # JSON message (status update)
                    print(f"  Server msg: {response}")
                    continue

                # Decode response JPEG → BGR
                arr = np.frombuffer(response, dtype=np.uint8)
                result_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if result_bgr is None:
                    continue

                self.frames_received += 1
                elapsed = (time.perf_counter() - t0) * 1000
                self.latency_ms = elapsed
                self.frame_times.append(elapsed)

                # Update FPS
                if len(self.frame_times) > 0:
                    avg_ms = sum(self.frame_times) / len(self.frame_times)
                    self.display_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

                # Send to virtual camera (needs RGB)
                if self.vcam:
                    try:
                        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        # Resize if dimensions don't match vcam
                        if result_rgb.shape[1] != self.vcam.width or result_rgb.shape[0] != self.vcam.height:
                            result_rgb = cv2.resize(result_rgb, (self.vcam.width, self.vcam.height))
                        self.vcam.send(result_rgb)
                    except Exception:
                        pass

                # Store for preview
                with self._result_lock:
                    self._result_frame = result_bgr

                # Log periodically
                if self.frames_received <= 3 or self.frames_received % 200 == 0:
                    print(f"  [Frame #{self.frames_received}] {elapsed:.0f}ms | "
                          f"~{self.display_fps:.0f}fps | {len(jpeg_bytes)/1024:.0f}KB")

    def _preview_loop(self):
        """OpenCV preview window — runs in main thread."""
        window = "Face Swap → Virtual Camera"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self.width, self.height)

        while self.running:
            with self._result_lock:
                frame = self._result_frame

            if frame is not None:
                # Add status overlay
                display = frame.copy()
                h, w = display.shape[:2]
                cv2.rectangle(display, (0, h - 35), (w, h), (0, 0, 0), -1)
                status = (
                    f"FPS: {self.display_fps:.0f}  |  "
                    f"Latency: {self.latency_ms:.0f}ms  |  "
                    f"Frames: {self.frames_received}  |  "
                    f"VCam: {'ON' if self.vcam else 'OFF'}"
                )
                cv2.putText(
                    display, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or Escape
                print("\n🛑 Quit requested")
                self.running = False
                break

        cv2.destroyAllWindows()

    def run(self):
        """Main entry point — start everything."""
        print("=" * 60)
        print("🎭 PlasticVision Pro v2 — Virtual Camera Client")
        print("=" * 60)

        # 1. Check server
        print("\n📡 Checking GPU server...")
        if not self._check_server():
            print("❌ Server not ready. Start the backend first.")
            return

        # 2. Upload source face if provided
        if self.source_image:
            print(f"\n📸 Uploading source face: {self.source_image}")
            sid = self._upload_source_face(self.source_image)
            if sid:
                self.session_id = sid

        # 3. Open camera
        print("\n📹 Opening webcam...")
        if not self._open_camera():
            return

        # 4. Open virtual camera
        print("\n📺 Starting virtual camera...")
        self._open_vcam()
        if not self.vcam:
            print("   (Continuing without virtual camera — preview only)")

        # 5. Start WebSocket + processing
        print("\n🚀 Starting face swap pipeline...")
        print("   Press 'q' in preview window to quit\n")
        self.running = True

        # Run WebSocket loop in a separate thread
        def ws_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._ws_loop())
            except websockets.exceptions.ConnectionClosed:
                print("⚠️  WebSocket disconnected")
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
            finally:
                self.running = False
                loop.close()

        t = threading.Thread(target=ws_thread, daemon=True)
        t.start()

        # Run preview in main thread (required for OpenCV highgui)
        if self.preview:
            self._preview_loop()
        else:
            # No preview — just wait
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Interrupted")

        # Cleanup
        self.running = False
        if self.vcam:
            self.vcam.close()
            print("📺 Virtual camera closed")
        if self.cap:
            self.cap.release()
            print("📹 Camera closed")

        print(f"\n✅ Done — {self.frames_received} frames processed")


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PlasticVision Pro v2 — Virtual Camera Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to local GPU server (via SSH tunnel)
  python vcam_client.py --server http://localhost:8000

  # Upload a source face and start
  python vcam_client.py --source ~/my_face.jpg

  # Use a specific camera and session
  python vcam_client.py --camera 1 --session abc123

  # No preview window (headless, virtual camera only)
  python vcam_client.py --no-preview

Prerequisites:
  1. GPU server running (locally or via SSH tunnel)
  2. Source face uploaded (via web UI or --source flag)
  3. OBS Virtual Camera activated:
     → Open OBS Studio → Start Virtual Camera → close OBS
     → System Settings → Privacy & Security → approve extension
  4. pip install pyvirtualcam websockets opencv-python
""",
    )

    parser.add_argument("--server", default="http://localhost:8000",
                        help="GPU server URL (default: http://localhost:8000)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Capture height (default: 720)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (default: 30)")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG quality 1-100 (default: 92)")
    parser.add_argument("--session", default=None,
                        help="Session ID (reuse existing session with loaded faces)")
    parser.add_argument("--source", default=None,
                        help="Source face image to upload before starting")
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror the webcam horizontally")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable preview window (virtual camera only)")

    args = parser.parse_args()

    if not WS_AVAILABLE:
        print("❌ websockets library required. Run: pip install websockets")
        sys.exit(1)

    client = VCamClient(
        server_url=args.server,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.quality,
        session_id=args.session,
        source_image=args.source,
        preview=not args.no_preview,
        mirror=args.mirror,
    )
    client.run()


if __name__ == "__main__":
    main()
