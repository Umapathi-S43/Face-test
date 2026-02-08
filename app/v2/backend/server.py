#!/usr/bin/env python3
"""
PlasticVision Pro v2 — GPU Backend Server

FastAPI server with:
  - WebSocket /ws/stream for real-time webcam face swap (binary JPEG)
  - REST endpoints for image swap, video swap, face detection
  - Source face upload (any image format)
  - Per-session source face storage (multi-user support)
"""

import os
import sys
import cv2
import json
import time
import uuid
import shutil
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add parent so engine.py is importable
sys.path.insert(0, str(Path(__file__).parent))
from engine import FaceSwapEngine

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

MODELS_DIR = Path(__file__).parent.parent / "models"
UPLOAD_DIR = Path(__file__).parent / "_uploads"
OUTPUT_DIR = Path(__file__).parent / "_outputs"

# Video progress tracking (keyed by session_id)
video_progress: dict = {}  # {sid: {"progress": 0.0, "stage": "...", "done": False}}
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Singleton engine
engine: Optional[FaceSwapEngine] = None

# Active WebSocket sessions
active_sessions: dict = {}


# ═══════════════════════════════════════════════════════
# LIFESPAN — Initialize engine on startup
# ═══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("=" * 60)
    print("🎭 PlasticVision Pro v2 — GPU Backend")
    print("=" * 60)
    engine = FaceSwapEngine(models_dir=str(MODELS_DIR))
    engine.initialize()
    print(f"GPU: {engine.get_gpu_status()}")
    print("=" * 60)
    yield
    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="PlasticVision Pro v2 — GPU Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow configurable origins (default: all for development)
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body size limit (500MB — covers large video uploads)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 500 * 1024 * 1024))


# ═══════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════

class SettingsRequest(BaseModel):
    mouth_mask: Optional[bool] = None
    sharpness: Optional[float] = None
    enhance: Optional[bool] = None
    opacity: Optional[float] = None
    swap_all: Optional[bool] = None
    poisson_blend: Optional[bool] = None
    interpolation: Optional[bool] = None
    interpolation_weight: Optional[float] = None


# ═══════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "app": "PlasticVision Pro v2",
        "gpu": engine.get_gpu_status() if engine else "not initialized",
        "engine_ready": engine._initialized if engine else False,
    }


@app.get("/status")
async def get_status():
    """Full status with GPU info, loaded faces, settings."""
    return {
        "gpu": engine.get_gpu_status() if engine else "not initialized",
        "engine_ready": engine._initialized if engine else False,
        "faces_loaded": engine.has_source_face() if engine else False,
        "settings": engine.settings if engine else {},
        "active_sessions": len(active_sessions),
        "gfpgan_available": engine.face_enhancer is not None if engine else False,
    }


@app.post("/upload-source-faces")
async def upload_source_faces(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Upload 1-10 source face images (any format: JPEG, PNG, WebP, BMP, TIFF).
    Optionally provide session_id for per-user face storage.
    """
    if not engine or not engine._initialized:
        raise HTTPException(500, "Engine not initialized")

    if len(files) == 0:
        raise HTTPException(400, "No files uploaded")
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 source face images allowed")

    # Generate session_id if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Save uploaded files temporarily
    saved_paths: List[str] = []
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        for f in files:
            # Validate it's an image by extension
            ext = Path(f.filename).suffix.lower() if f.filename else ".jpg"
            if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'):
                ext = '.jpg'  # fallback

            save_path = session_dir / f"{uuid.uuid4().hex}{ext}"
            content = await f.read()

            if len(content) > 50 * 1024 * 1024:  # 50MB limit
                raise HTTPException(400, f"File {f.filename} exceeds 50MB limit")

            with open(save_path, "wb") as fp:
                fp.write(content)
            saved_paths.append(str(save_path))

        # Load faces into engine
        success = engine.load_source_faces(saved_paths, session_id=session_id)

        if success:
            return {
                "success": True,
                "session_id": session_id,
                "count": len(saved_paths),
                "message": f"✅ Loaded {len(saved_paths)} source face(s)",
            }
        else:
            raise HTTPException(400, "No faces detected in uploaded images. Try clearer photos.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing uploads: {str(e)}")


@app.post("/settings")
async def update_settings(req: SettingsRequest):
    """Update quality settings."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")

    if req.mouth_mask is not None:
        engine.settings["mouth_mask"] = req.mouth_mask
    if req.sharpness is not None:
        engine.settings["sharpness"] = max(0.0, min(1.0, req.sharpness))
    if req.enhance is not None:
        engine.settings["enhance"] = req.enhance
    if req.opacity is not None:
        engine.settings["opacity"] = max(0.0, min(1.0, req.opacity))
    if req.swap_all is not None:
        engine.settings["swap_all"] = req.swap_all
    if req.poisson_blend is not None:
        engine.settings["poisson_blend"] = req.poisson_blend
    if req.interpolation is not None:
        engine.settings["interpolation"] = req.interpolation
    if req.interpolation_weight is not None:
        engine.settings["interpolation_weight"] = max(0.0, min(1.0, req.interpolation_weight))

    return {"success": True, "settings": engine.settings}


@app.post("/swap-image")
async def swap_image(
    source_files: List[UploadFile] = File(...),
    target_file: UploadFile = File(...),
    enhance: bool = Form(True),
    swap_all: bool = Form(False),
    session_id: Optional[str] = Form(None),
):
    """Swap faces in a single image. Returns PNG image."""
    if not engine or not engine._initialized:
        raise HTTPException(500, "Engine not initialized")

    sid = session_id or str(uuid.uuid4())
    # Use a dedicated temp dir for uploaded files (not the session face dir)
    # This way cleanup won't delete pre-loaded source faces
    img_tmp_id = f"imgtmp_{uuid.uuid4().hex}"
    img_tmp_dir = UPLOAD_DIR / img_tmp_id
    img_tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save source faces
        src_paths = []
        for f in source_files:
            ext = Path(f.filename).suffix.lower() if f.filename else ".jpg"
            p = img_tmp_dir / f"{uuid.uuid4().hex}{ext}"
            with open(p, "wb") as fp:
                fp.write(await f.read())
            src_paths.append(str(p))

        # Save target
        ext = Path(target_file.filename).suffix.lower() if target_file.filename else ".jpg"
        target_path = img_tmp_dir / f"target_{uuid.uuid4().hex}{ext}"
        with open(target_path, "wb") as fp:
            fp.write(await target_file.read())

        # Process
        result = engine.swap_face(
            source_paths=src_paths,
            target_path=str(target_path),
            enhance=enhance,
            swap_all=swap_all,
            session_id=sid,
        )

        if result is None:
            raise HTTPException(400, "No faces detected in target image")

        # Encode as PNG — result is already BGR (OpenCV native)
        _, buf = cv2.imencode('.png', result)

        return Response(content=buf.tobytes(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Swap error: {str(e)}")
    finally:
        # Always clean up the image temp dir (separate from session face data)
        shutil.rmtree(img_tmp_dir, ignore_errors=True)


@app.get("/video-progress/{sid}")
async def get_video_progress(sid: str):
    """Poll video processing progress. Returns JSON {progress, stage, done}."""
    entry = video_progress.get(sid, {"progress": 0.0, "stage": "waiting", "done": False})
    return JSONResponse(entry)


@app.post("/swap-video")
async def swap_video(
    source_files: List[UploadFile] = File(...),
    target_file: UploadFile = File(...),
    enhance: bool = Form(False),
    swap_all: bool = Form(False),
    mouth_mask: bool = Form(True),
    sharpness: float = Form(0.3),
    session_id: Optional[str] = Form(None),
):
    """Swap faces in a video file. Returns MP4 video."""
    if not engine or not engine._initialized:
        raise HTTPException(500, "Engine not initialized")

    sid = session_id or str(uuid.uuid4())
    # Use a dedicated temp dir for video processing files (not the session face dir)
    # This way cleanup won't delete pre-loaded source faces
    video_tmp_id = f"vidtmp_{uuid.uuid4().hex}"
    video_tmp_dir = UPLOAD_DIR / video_tmp_id
    video_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracking — client polls /video-progress/{sid}
    video_progress[sid] = {"progress": 0.0, "stage": "uploading", "done": False}

    try:
        # Save source faces
        src_paths = []
        for f in source_files:
            ext = Path(f.filename).suffix.lower() if f.filename else ".jpg"
            p = video_tmp_dir / f"{uuid.uuid4().hex}{ext}"
            with open(p, "wb") as fp:
                fp.write(await f.read())
            src_paths.append(str(p))

        # Save target video
        ext = Path(target_file.filename).suffix.lower() if target_file.filename else ".mp4"
        target_path = video_tmp_dir / f"video_{uuid.uuid4().hex}{ext}"
        video_content = await target_file.read()
        if len(video_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(413, f"Video exceeds {MAX_UPLOAD_SIZE // (1024*1024)}MB limit")
        with open(target_path, "wb") as fp:
            fp.write(video_content)

        # Output path
        output_path = str(OUTPUT_DIR / f"result_{uuid.uuid4().hex}.mp4")

        # Progress callback updates the shared dict
        def on_progress(pct: float):
            video_progress[sid] = {
                "progress": round(pct, 4),
                "stage": "swapping" if pct < 1.0 else "muxing_audio",
                "done": False,
            }

        video_progress[sid]["stage"] = "processing"

        # Process (runs in thread to not block event loop)
        success = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.swap_face_video(
                source_paths=src_paths,
                target_video_path=str(target_path),
                output_path=output_path,
                enhance=enhance,
                swap_all=swap_all,
                use_mouth_mask=mouth_mask,
                sharpness=sharpness,
                progress_callback=on_progress,
                session_id=sid,
            ),
        )

        if not success:
            video_progress[sid] = {"progress": 0, "stage": "failed", "done": True}
            raise HTTPException(500, "Video processing failed")

        video_progress[sid] = {"progress": 1.0, "stage": "complete", "done": True}

        # Stream the result file, then clean up
        def file_stream():
            try:
                with open(output_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        yield chunk
            finally:
                # Clean up output file after streaming completes
                try:
                    os.remove(output_path)
                except OSError:
                    pass
                # Clean up progress entry
                video_progress.pop(sid, None)

        return StreamingResponse(
            file_stream(),
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=result.mp4"},
        )

    except HTTPException:
        raise
    except Exception as e:
        video_progress[sid] = {"progress": 0, "stage": "error", "done": True}
        raise HTTPException(500, f"Video error: {str(e)}")
    finally:
        # Always clean up the video temp dir (separate from session face data)
        shutil.rmtree(video_tmp_dir, ignore_errors=True)


@app.post("/detect-faces")
async def detect_faces_endpoint(file: UploadFile = File(...)):
    """Detect faces in an image and return annotated image as PNG."""
    if not engine or not engine._initialized:
        raise HTTPException(500, "Engine not initialized")

    try:
        content = await file.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")

        # InsightFace expects BGR — no conversion needed
        faces = engine.detect_faces(img)

        # Draw boxes (use green in BGR: (0, 255, 0))
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, f"Face {i+1}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buf = cv2.imencode('.png', img)

        return Response(content=buf.tobytes(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Detection error: {str(e)}")


# ═══════════════════════════════════════════════════════
# WEBSOCKET — Live Webcam Stream
# ═══════════════════════════════════════════════════════

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Live face swap over WebSocket.

    Protocol:
      1. Client connects
      2. Server sends JSON: {"status": "connected", "session_id": "..."}
      3. Client can send text JSON for commands:
         - {"action": "set_session", "session_id": "..."}
      4. Client sends binary JPEG frames
      5. Server responds with binary JPEG processed frames
    """
    await ws.accept()

    session_id = str(uuid.uuid4())
    active_sessions[session_id] = ws
    frame_count = 0
    total_ms = 0

    try:
        # Send connection confirmation
        await ws.send_json({
            "status": "connected",
            "session_id": session_id,
            "faces_loaded": engine.has_source_face(session_id) if engine else False,
        })

        while True:
            # Receive data (text for commands, binary for frames)
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Handle text commands
            if "text" in message:
                try:
                    cmd = json.loads(message["text"])
                    action = cmd.get("action")

                    if action == "set_session":
                        new_sid = cmd.get("session_id")
                        if new_sid and engine.has_source_face(new_sid):
                            # Remove old session tracking
                            active_sessions.pop(session_id, None)
                            session_id = new_sid
                            active_sessions[session_id] = ws
                            await ws.send_json({
                                "status": "session_set",
                                "session_id": session_id,
                                "faces_loaded": True,
                            })
                        else:
                            await ws.send_json({
                                "status": "error",
                                "message": "Session not found or no faces loaded",
                            })

                    elif action == "ping":
                        await ws.send_json({"status": "pong", "time": time.time()})

                    elif action == "get_status":
                        await ws.send_json({
                            "status": "ok",
                            "faces_loaded": engine.has_source_face(session_id) if engine else False,
                            "settings": engine.settings if engine else {},
                            "fps": (1000.0 / (total_ms / max(frame_count, 1))) if frame_count > 0 else 0,
                        })

                except json.JSONDecodeError:
                    await ws.send_json({"status": "error", "message": "Invalid JSON"})
                continue

            # Handle binary frame data
            if "bytes" in message:
                data = message["bytes"]
                if not data:
                    continue

                frame_count += 1
                t0 = time.perf_counter()

                try:
                    # ── OPTIMIZED PIPELINE ──
                    # Decode JPEG → BGR (cv2 native format)
                    arr = np.frombuffer(data, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if frame_bgr is None:
                        await ws.send_bytes(data)
                        continue

                    # Check if faces are loaded
                    if not engine or not engine.has_source_face(session_id):
                        # No source face — draw overlay and return
                        h, w = frame_bgr.shape[:2]
                        overlay = frame_bgr.copy()
                        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, frame_bgr, 0.3, 0, frame_bgr)
                        cv2.putText(frame_bgr, "Upload source faces first!",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                        _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        await ws.send_bytes(buf.tobytes())
                        continue

                    # Process frame using OPTIMIZED live path
                    # - Uses 320x320 detection (4x faster than 640x640)
                    # - Works in BGR to skip color conversions
                    # - No GFPGAN (too slow for live)
                    # - Runs in thread pool to avoid blocking event loop
                    settings = engine.settings

                    def _process_frame():
                        result_bgr = engine.swap_face_frame_live(
                            frame_bgr,
                            use_mouth_mask=settings["mouth_mask"],
                            sharpness=settings["sharpness"],
                            opacity=settings["opacity"],
                            swap_all=settings["swap_all"],
                            session_id=session_id,
                        )
                        _, buf = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        return buf.tobytes()

                    result_bytes = await asyncio.to_thread(_process_frame)

                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    total_ms += elapsed_ms

                    # Log periodically
                    if frame_count <= 3 or frame_count % 100 == 0:
                        avg = total_ms / frame_count
                        print(f"[WS] Session {session_id[:8]} | Frame #{frame_count} | "
                              f"{elapsed_ms:.1f}ms | Avg {avg:.1f}ms | ~{1000/avg:.0f}fps")

                    await ws.send_bytes(result_bytes)

                except Exception as e:
                    if frame_count <= 5 or frame_count % 60 == 0:
                        print(f"[WS] Frame error: {e}")
                    # Send back the original frame on error
                    await ws.send_bytes(data)

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {session_id[:8]}")
    except Exception as e:
        print(f"[WS] Connection error: {e}")
    finally:
        active_sessions.pop(session_id, None)
        if frame_count > 0:
            avg = total_ms / frame_count
            print(f"[WS] Session {session_id[:8]} ended | {frame_count} frames | Avg {avg:.1f}ms")


# ═══════════════════════════════════════════════════════
# CLI Entry point (for direct `python server.py`)
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="PlasticVision Pro v2 — GPU Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"🚀 Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ws_max_size=50 * 1024 * 1024,  # 50MB WebSocket message limit
    )
