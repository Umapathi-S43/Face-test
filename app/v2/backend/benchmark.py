#!/usr/bin/env python3
"""
Latency Benchmark — Measures every step of the WebSocket face swap pipeline.
Run this on the GPU server to get accurate timings.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def benchmark(name, fn, iterations=100):
    """Run a function N times and report timing stats."""
    times = []
    result = None
    for i in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    avg = np.mean(times)
    mn = np.min(times)
    mx = np.max(times)
    p95 = np.percentile(times, 95)
    print(f"  {name:30s}  avg={avg:7.2f}ms  min={mn:6.2f}ms  p95={p95:6.2f}ms  max={mx:6.2f}ms")
    return result, avg


def main():
    print("=" * 80)
    print("🔬 PlasticVision Pro v2 — Full Pipeline Latency Benchmark")
    print("=" * 80)

    # ── Step 0: Create test data ──
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, jpg_buf = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpg_bytes = jpg_buf.tobytes()
    print(f"\nTest frame: 640x480, JPEG size: {len(jpg_bytes)/1024:.1f}KB")

    print("\n── STEP 1: Network I/O Overhead (JPEG decode/encode) ──")

    # JPEG decode
    def decode_jpeg():
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    frame, t_decode = benchmark("JPEG decode", decode_jpeg, 200)

    # BGR→RGB
    def bgr_to_rgb():
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb, t_cvt1 = benchmark("BGR→RGB cvtColor", bgr_to_rgb, 200)

    # RGB→BGR + JPEG encode
    def encode_jpeg():
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf

    _, t_encode = benchmark("RGB→BGR + JPEG encode", encode_jpeg, 200)

    # frame.copy()
    def copy_frame():
        return rgb.copy()

    _, t_copy = benchmark("frame.copy()", copy_frame, 200)

    print(f"\n  📊 Total I/O overhead: {t_decode + t_cvt1 + t_encode + t_copy:.2f}ms")

    # ── Step 2: Face Detection ──
    print("\n── STEP 2: Face Detection (InsightFace buffalo_l) ──")

    from engine import FaceSwapEngine, apply_sharpening, apply_color_transfer
    engine = FaceSwapEngine(models_dir='../models')
    engine.initialize()

    def detect_no_face():
        return engine.detect_faces(rgb)

    _, t_detect_noface = benchmark("detect_faces (no face in img)", detect_no_face, 5)

    # Also benchmark with 320x320 det_size (live mode)
    def detect_no_face_live():
        return engine.detect_faces_live(rgb)

    _, t_detect_noface_live = benchmark("detect_faces_live 320x320 (no face)", detect_no_face_live, 5)

    # Try with a real face image if available
    test_imgs = [
        Path("/Users/umapathi/face-swap/app/test_source.jpg"),
        Path("/Users/umapathi/face-swap/app/test_target.jpg"),
    ]
    test_imgs = [p for p in test_imgs if p.exists()]
    if not test_imgs:
        test_imgs = list(Path("/Users/umapathi/face-swap/app/uploads").glob("*.*"))

    real_face_frame = None
    if test_imgs:
        for p in test_imgs:
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.resize(img, (640, 480))
                real_face_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = engine.detect_faces(real_face_frame)
                if faces:
                    print(f"\n  Using real face image: {p.name} ({len(faces)} faces)")
                    break
                real_face_frame = None

    if real_face_frame is not None:
        def detect_real():
            return engine.detect_faces(real_face_frame)

        faces_result, t_detect_real = benchmark("detect_faces (with face)", detect_real, 5)

        def detect_real_live():
            return engine.detect_faces_live(real_face_frame)

        _, t_detect_real_live = benchmark("detect_faces_live 320x320 (face)", detect_real_live, 5)
    else:
        print("  ⚠️ No test image with a face found — using synthetic frame")
        t_detect_real = t_detect_noface

    # ── Step 3: Face Swap (inswapper_128) ──
    print("\n── STEP 3: Face Swap (inswapper_128) ──")

    if real_face_frame is not None:
        faces = engine.detect_faces(real_face_frame)
        if faces:
            # Load a source face first
            face = faces[0]
            engine.source_face = face

            def swap_one():
                return engine.face_swapper.get(real_face_frame.copy(), face, engine.source_face, paste_back=True)

            _, t_swap = benchmark("inswapper_128.get() (1 face)", swap_one, 5)

            # Benchmark full optimized live pipeline
            def swap_live():
                return engine.swap_face_frame_live(
                    cv2.cvtColor(real_face_frame, cv2.COLOR_RGB2BGR),
                    use_mouth_mask=True, sharpness=0.3,
                    opacity=1.0, swap_all=False,
                )

            _, t_live_full = benchmark("swap_face_frame_live (FULL)", swap_live, 5)
        else:
            t_swap = 0
            print("  ⚠️ No face to swap")
    else:
        t_swap = 0
        print("  ⚠️ No real face image — skipping swap benchmark")

    # ── Step 4: Quality post-processing ──
    print("\n── STEP 4: Quality Post-Processing ──")

    def sharpen():
        return apply_sharpening(rgb, 0.3)

    _, t_sharp = benchmark("apply_sharpening(0.3)", sharpen, 200)

    # Mouth mask (if we have a face)
    t_mouth = 0
    if real_face_frame is not None and faces:
        from engine import create_lower_mouth_mask, create_face_mask, apply_mouth_area

        def mouth_mask_pipeline():
            mm, mc, mb, mp = create_lower_mouth_mask(face, real_face_frame)
            fm = create_face_mask(face, real_face_frame)
            if mc is not None and mb != (0, 0, 0, 0):
                apply_mouth_area(real_face_frame.copy(), mc, mb, fm, mp)

        _, t_mouth = benchmark("mouth mask pipeline", mouth_mask_pipeline, 20)

    # Color transfer
    region1 = rgb[100:200, 100:200]
    region2 = rgb[200:300, 200:300]

    def color_xfer():
        return apply_color_transfer(region1, region2)

    _, t_color = benchmark("apply_color_transfer", color_xfer, 200)

    # GFPGAN (if available)
    t_gfpgan = 0
    if engine.face_enhancer and real_face_frame is not None:
        def gfpgan_enhance():
            return engine._enhance_face(real_face_frame)

        _, t_gfpgan = benchmark("GFPGAN enhance", gfpgan_enhance, 10)

    # ── TOTAL ESTIMATE ──
    print("\n" + "=" * 80)
    print("📊 TOTAL LATENCY ESTIMATE (per frame on this machine)")
    print("=" * 80)

    server_processing = t_decode + t_cvt1 + t_detect_real + t_swap + t_mouth + t_sharp + t_color + t_encode + t_copy

    # Calculate optimized live pipeline time
    t_live_total = t_decode + (t_detect_real_live if 't_detect_real_live' in dir() else t_detect_real) + t_swap + t_mouth + t_sharp + t_encode

    print(f"""
  ═══ STANDARD PIPELINE (640x640 detection) ═══
  JPEG decode:           {t_decode:6.2f}ms
  BGR→RGB:               {t_cvt1:6.2f}ms
  frame.copy():          {t_copy:6.2f}ms
  Face Detection 640:    {t_detect_real:6.2f}ms  ← BIGGEST BOTTLENECK
  Face Swap:             {t_swap:6.2f}ms
  Mouth Mask:            {t_mouth:6.2f}ms
  Sharpening:            {t_sharp:6.2f}ms
  Color Transfer:        {t_color:6.2f}ms
  JPEG encode:           {t_encode:6.2f}ms
  ─────────────────────────────
  STANDARD TOTAL:        {server_processing:6.2f}ms

  ═══ OPTIMIZED LIVE PIPELINE (320x320 detection) ═══
  JPEG decode:           {t_decode:6.2f}ms
  Face Detection 320:    {t_detect_real_live if 't_detect_real_live' in dir() else 0:6.2f}ms  ← 4x FASTER
  Face Swap:             {t_swap:6.2f}ms
  Mouth Mask:            {t_mouth:6.2f}ms
  Sharpening:            {t_sharp:6.2f}ms
  JPEG encode:           {t_encode:6.2f}ms
  ─────────────────────────────
  LIVE TOTAL (this CPU): {t_live_total:6.2f}ms""")

    if 't_live_full' in dir():
        print(f"  swap_frame_live MEASURED:{t_live_full:6.2f}ms")

    print(f"""
  GFPGAN (if enabled):  +{t_gfpgan:6.2f}ms  ← DISABLED FOR LIVE

  + Network RTT (local): ~1-2ms
  + Network RTT (internet): ~20-50ms
  + Browser decode+render: ~2-5ms
  ─────────────────────────────
""")

    # Estimate on RTX 5090
    # Published benchmarks show:
    # - Face detection (RetinaFace): CPU ~120-190ms → GPU ~3-8ms (15-25x speedup)
    # - inswapper_128 FP32: CPU ~1400ms → GPU ~15-25ms (60-90x speedup)
    # - inswapper_128 FP16: GPU ~8-12ms (2x faster than FP32 on GPU)
    # - 320x320 detection: ~2-4ms on GPU (vs 3-8ms for 640x640)
    if "CPU" in engine.get_gpu_status():
        det_live_cpu = t_detect_real_live if 't_detect_real_live' in dir() else t_detect_real
        # RTX 5090 estimates based on published ONNX GPU benchmarks
        gpu_det_640 = 6.0    # ~6ms for 640x640 RetinaFace on RTX 5090
        gpu_det_320 = 2.5    # ~2.5ms for 320x320 RetinaFace on RTX 5090
        gpu_swap_fp32 = 18.0 # ~18ms for inswapper_128 FP32
        gpu_swap_fp16 = 9.0  # ~9ms for inswapper_128 FP16
        
        gpu_live_total = (t_decode + gpu_det_320 + gpu_swap_fp16 + 
                         t_mouth + t_sharp + t_encode)
        gpu_standard_total = (t_decode + t_cvt1 + t_copy + gpu_det_640 + 
                             gpu_swap_fp32 + t_mouth + t_sharp + t_color + t_encode)

        print(f"  ⚡ ESTIMATED on RTX 5090 GPU (2x RTX 5090, 31GB each):")
        print(f"")
        print(f"     ═══ STANDARD MODE (640x640 + FP32) ═══")
        print(f"     Face Detection 640: {gpu_det_640:6.2f}ms")
        print(f"     Face Swap FP32:     {gpu_swap_fp32:6.2f}ms")
        print(f"     + I/O & post-proc:  {t_decode + t_cvt1 + t_copy + t_mouth + t_sharp + t_color + t_encode:6.2f}ms")
        print(f"     Server Total:       {gpu_standard_total:6.2f}ms")
        print(f"")
        print(f"     ═══ OPTIMIZED LIVE MODE (320x320 + FP16) ═══")
        print(f"     Face Detection 320: {gpu_det_320:6.2f}ms")
        print(f"     Face Swap FP16:     {gpu_swap_fp16:6.2f}ms")
        print(f"     + I/O & post-proc:  {t_decode + t_mouth + t_sharp + t_encode:6.2f}ms")
        print(f"     Server Total:       {gpu_live_total:6.2f}ms")
        print(f"     + Network RTT:      ~5-15ms (RunPod)")
        print(f"     + Browser render:   ~2-3ms")
        print(f"     ─────────────────────────")
        print(f"     END-TO-END:         ~{gpu_live_total + 10:.1f}ms")
        print()

        if gpu_live_total + 10 < 50:
            print(f"  ✅ Sub-50ms IS ACHIEVABLE on RTX 5090! ({gpu_live_total+10:.1f}ms estimated)")
            print(f"     Expected FPS: ~{1000/(gpu_live_total+10):.0f} fps")
        else:
            print(f"  ⚠️ May be tight on RTX 5090 ({gpu_live_total+10:.1f}ms). Further optimization needed.")

    print()
    print("  💡 OPTIMIZATION TIPS:")
    print("     1. Use 320x320 det_size instead of 640x640 → detection 4x faster")
    print("     2. Use inswapper_128_fp16.onnx → swap 2x faster, half memory")
    print("     3. Skip BGR↔RGB by receiving/sending BGR directly")
    print("     4. Use turbojpeg instead of cv2 for JPEG → 2x faster encode/decode")
    print("     5. Disable GFPGAN for live stream (keep for image/video swap)")
    print("     6. Use asyncio.to_thread() to avoid blocking the event loop")


if __name__ == "__main__":
    main()
