/**
 * PlasticVision Pro v2 — Frontend Application
 *
 * WebSocket client for real-time face swap streaming.
 * REST client for image/video swap and face detection.
 */

// ═══════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════

let serverUrl = 'http://localhost:8000';
let wsUrl = 'ws://localhost:8000';
let sessionId = null;
let ws = null;
let isStreaming = false;
let webcamStream = null;
let animFrameId = null;

// FPS / latency tracking
let framesSent = 0;
let framesReceived = 0;
let lastFpsUpdate = 0;
let lastSendTime = 0;
let latencyMs = 0;
let displayFps = 0;
let waitingForResponse = false;
let outputCtx = null;  // Cached canvas context for output rendering

// ═══════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════

document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    });
});

// ═══════════════════════════════════════════════════════
// SERVER CONNECTION
// ═══════════════════════════════════════════════════════

async function connectToServer() {
    const input = document.getElementById('server-url');
    serverUrl = input.value.replace(/\/+$/, '');
    wsUrl = serverUrl.replace(/^http/, 'ws');

    setConnectionStatus('connecting', '🟡 Connecting...');

    try {
        const res = await fetch(`${serverUrl}/status`);
        const data = await res.json();

        setConnectionStatus('connected', '🟢 Connected');
        document.getElementById('gpu-info').textContent = data.gpu || '';

        if (data.faces_loaded) {
            setStatus('load-status', '✅ Server has source faces loaded');
        }
    } catch (e) {
        setConnectionStatus('disconnected', '🔴 Cannot reach server');
        document.getElementById('gpu-info').textContent = '';
        console.error('Connection failed:', e);
    }
}

function setConnectionStatus(cls, text) {
    const el = document.getElementById('connection-status');
    el.className = `status ${cls}`;
    el.textContent = text;
}

function setStatus(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ═══════════════════════════════════════════════════════
// SOURCE FACE UPLOAD
// ═══════════════════════════════════════════════════════

function previewSourceFiles() {
    const files = document.getElementById('source-files').files;
    const grid = document.getElementById('source-preview');
    const btn = document.getElementById('load-faces-btn');

    // Revoke old preview URLs to prevent memory leaks
    grid.querySelectorAll('img').forEach(img => {
        if (img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
    });
    grid.innerHTML = '';
    btn.disabled = files.length === 0;

    for (const file of files) {
        const img = document.createElement('img');
        const url = URL.createObjectURL(file);
        img.src = url;
        img.onload = () => URL.revokeObjectURL(url); // Free after rendered
        img.onerror = () => URL.revokeObjectURL(url); // Free on load failure too
        grid.appendChild(img);
    }

    if (files.length > 0) {
        setStatus('load-status', `📸 ${files.length} photo(s) selected — click Load Faces`);
    }
}

async function uploadSourceFaces() {
    const files = document.getElementById('source-files').files;
    if (files.length === 0) {
        setStatus('load-status', '❌ Select face photos first');
        return;
    }

    setStatus('load-status', '⏳ Uploading and loading faces...');
    document.getElementById('load-faces-btn').disabled = true;

    try {
        const form = new FormData();
        for (const file of files) {
            form.append('files', file);
        }
        if (sessionId) {
            form.append('session_id', sessionId);
        }

        const res = await fetch(`${serverUrl}/upload-source-faces`, {
            method: 'POST',
            body: form,
        });

        const data = await res.json();

        if (res.ok && data.success) {
            sessionId = data.session_id;
            setStatus('load-status', data.message);

            // If WebSocket is connected, link session
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'set_session', session_id: sessionId }));
            }
        } else {
            setStatus('load-status', `❌ ${data.detail || data.message || 'Upload failed'}`);
        }
    } catch (e) {
        setStatus('load-status', `❌ Error: ${e.message}`);
    } finally {
        document.getElementById('load-faces-btn').disabled = false;
    }
}

// ═══════════════════════════════════════════════════════
// SETTINGS
// ═══════════════════════════════════════════════════════

async function updateSettings() {
    const settings = {
        mouth_mask: document.getElementById('set-mouth-mask').checked,
        sharpness: parseFloat(document.getElementById('set-sharpness').value),
        enhance: document.getElementById('set-enhance').checked,
        opacity: parseFloat(document.getElementById('set-opacity').value),
        swap_all: document.getElementById('set-swap-all').checked,
    };

    try {
        const res = await fetch(`${serverUrl}/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
        });
        const data = await res.json();
        if (data.success) {
            const s = data.settings;
            setStatus('settings-status',
                `✅ Lip Sync=${s.mouth_mask}, Sharp=${s.sharpness}, HD=${s.enhance}, Opacity=${s.opacity}`);
        }
    } catch (e) {
        setStatus('settings-status', `⚠️ Server not connected — settings saved locally`);
    }
}

// ═══════════════════════════════════════════════════════
// LIVE WEBCAM STREAM
// ═══════════════════════════════════════════════════════

async function enumerateCameras() {
    try {
        // Need to get permission first
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
        tempStream.getTracks().forEach(t => t.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const select = document.getElementById('camera-select');
        select.innerHTML = '';
        devices.filter(d => d.kind === 'videoinput').forEach((d, i) => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            opt.textContent = d.label || `Camera ${i + 1}`;
            select.appendChild(opt);
        });
    } catch (e) {
        console.warn('Cannot enumerate cameras:', e);
    }
}

async function startStream() {
    if (isStreaming) return;

    setStatus('live-status', '⏳ Starting webcam and connecting...');

    try {
        // Start webcam
        const cameraId = document.getElementById('camera-select').value;
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 },
                ...(cameraId ? { deviceId: { exact: cameraId } } : {}),
            },
        };

        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById('webcam-video');
        video.srcObject = webcamStream;
        await video.play();

        // Set up capture canvas
        const canvas = document.getElementById('capture-canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        // Connect WebSocket
        ws = new WebSocket(`${wsUrl}/ws/stream`);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            console.log('WebSocket connected');
            isStreaming = true;
            document.getElementById('start-stream-btn').disabled = true;
            document.getElementById('stop-stream-btn').disabled = false;
            setStatus('live-status', '🟢 Streaming — waiting for server...');
        };

        ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                // JSON control message
                const msg = JSON.parse(event.data);
                console.log('Server:', msg);

                if (msg.session_id && !sessionId) {
                    sessionId = msg.session_id;
                }

                // If we have a session, link it
                if (msg.status === 'connected' && sessionId) {
                    ws.send(JSON.stringify({ action: 'set_session', session_id: sessionId }));
                }

                if (msg.faces_loaded !== undefined) {
                    setStatus('live-status', msg.faces_loaded
                        ? '🟢 Streaming with face swap active'
                        : '⚠️ Streaming — upload source faces in Setup tab');
                }
                return;
            }

            // Binary frame — display it
            framesReceived++;
            latencyMs = performance.now() - lastSendTime;
            waitingForResponse = false;

            // Use createImageBitmap() — faster than Blob→ObjectURL→Image
            // Avoids GC pressure from URL.createObjectURL/revokeObjectURL per frame
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            createImageBitmap(blob).then(bmp => {
                const outCanvas = document.getElementById('output-canvas');
                // Only resize canvas when dimensions actually change (avoids flicker)
                if (outCanvas.width !== bmp.width || outCanvas.height !== bmp.height) {
                    outCanvas.width = bmp.width;
                    outCanvas.height = bmp.height;
                    outputCtx = null;  // Canvas resize resets context
                }
                // Cache context — avoid repeated getContext() calls per frame
                if (!outputCtx) {
                    outputCtx = outCanvas.getContext('2d', { desynchronized: true });
                }
                outputCtx.drawImage(bmp, 0, 0);
                bmp.close(); // Release bitmap memory immediately
            }).catch(err => {
                console.warn('Frame decode error:', err);
            });

            // Update FPS display
            const now = performance.now();
            if (now - lastFpsUpdate > 500) {
                displayFps = Math.round(framesReceived / ((now - lastFpsUpdate) / 1000));
                framesReceived = 0;
                lastFpsUpdate = now;

                document.getElementById('live-fps').textContent = `FPS: ${displayFps}`;
                document.getElementById('live-latency').textContent = `Latency: ${Math.round(latencyMs)}ms`;
            }
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            if (isStreaming) stopStream();
        };

        ws.onerror = (e) => {
            console.error('WebSocket error:', e);
            waitingForResponse = false; // Reset so capture loop doesn't freeze
            setStatus('live-status', '🔴 WebSocket error — check server');
        };

        // Wait for connection then start sending frames
        await new Promise((resolve, reject) => {
            const check = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) { clearInterval(check); resolve(); }
                if (ws.readyState === WebSocket.CLOSED) { clearInterval(check); reject(new Error('Connection failed')); }
            }, 50);
            setTimeout(() => { clearInterval(check); reject(new Error('Connection timeout')); }, 5000);
        });

        // Start frame capture loop
        lastFpsUpdate = performance.now();
        captureLoop();

    } catch (e) {
        setStatus('live-status', `❌ Error: ${e.message}`);
        stopStream();
    }
}

// Cache DOM elements for capture loop (avoid repeated getElementById per frame)
let _captureVideo = null;
let _captureCanvas = null;
let _captureCtx = null;

function captureLoop() {
    if (!isStreaming) return;

    // Don't send a new frame if we're still waiting for the previous response
    // This prevents frame buildup and keeps latency low
    if (!waitingForResponse && ws && ws.readyState === WebSocket.OPEN) {
        if (!_captureVideo) _captureVideo = document.getElementById('webcam-video');
        if (!_captureCanvas) _captureCanvas = document.getElementById('capture-canvas');
        if (!_captureCtx) _captureCtx = _captureCanvas.getContext('2d');

        // Ensure canvas matches video dimensions
        if (_captureCanvas.width !== _captureVideo.videoWidth || _captureCanvas.height !== _captureVideo.videoHeight) {
            _captureCanvas.width = _captureVideo.videoWidth || 640;
            _captureCanvas.height = _captureVideo.videoHeight || 480;
        }

        _captureCtx.drawImage(_captureVideo, 0, 0, _captureCanvas.width, _captureCanvas.height);

        // Encode as JPEG and send as binary
        _captureCanvas.toBlob((blob) => {
            if (blob && ws && ws.readyState === WebSocket.OPEN) {
                blob.arrayBuffer().then(buf => {
                    ws.send(buf);
                    framesSent++;
                    lastSendTime = performance.now();
                    waitingForResponse = true;

                    // Update frame size display
                    const sizeKB = (buf.byteLength / 1024).toFixed(1);
                    document.getElementById('live-size').textContent = `Frame: ${sizeKB}KB`;
                }).catch(err => {
                    console.warn('Frame encode error:', err);
                    waitingForResponse = false; // Reset so stream doesn't freeze
                });
            }
        }, 'image/jpeg', 0.92);
    }

    // Schedule next frame — use requestAnimationFrame for smooth timing
    animFrameId = requestAnimationFrame(captureLoop);
}

function stopStream() {
    isStreaming = false;
    waitingForResponse = false;  // Reset so next session starts cleanly

    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }

    const video = document.getElementById('webcam-video');
    video.srcObject = null;

    // Reset cached DOM refs so they're re-fetched on next session
    _captureVideo = null;
    _captureCanvas = null;
    _captureCtx = null;
    outputCtx = null;

    document.getElementById('start-stream-btn').disabled = false;
    document.getElementById('stop-stream-btn').disabled = true;
    document.getElementById('live-fps').textContent = 'FPS: —';
    document.getElementById('live-latency').textContent = 'Latency: —';
    document.getElementById('live-size').textContent = 'Frame: —';
    setStatus('live-status', '⏹️ Stream stopped');
}

async function switchCamera() {
    if (isStreaming) {
        stopStream();
        await new Promise(r => setTimeout(r, 500));
        startStream();
    }
}

// ═══════════════════════════════════════════════════════
// IMAGE SWAP
// ═══════════════════════════════════════════════════════

async function swapImage() {
    const sourceFiles = document.getElementById('img-source-files').files;
    const targetFile = document.getElementById('img-target-file').files[0];

    if (sourceFiles.length === 0) return setStatus('img-status', '❌ Upload source face images');
    if (!targetFile) return setStatus('img-status', '❌ Upload target image');

    setStatus('img-status', '⏳ Processing...');

    try {
        const form = new FormData();
        for (const f of sourceFiles) form.append('source_files', f);
        form.append('target_file', targetFile);
        form.append('enhance', document.getElementById('img-enhance').checked);
        form.append('swap_all', document.getElementById('img-swap-all').checked);
        if (sessionId && sessionId !== 'null') form.append('session_id', sessionId);

        const res = await fetch(`${serverUrl}/swap-image`, {
            method: 'POST',
            body: form,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Swap failed');
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        const img = document.getElementById('img-result');
        // Revoke previous result URL to prevent memory leak
        if (img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
        img.src = url;
        img.style.display = 'block';

        const dl = document.getElementById('img-download');
        if (dl.href.startsWith('blob:')) URL.revokeObjectURL(dl.href);
        dl.href = url;
        dl.style.display = 'inline-block';

        setStatus('img-status', '✅ Face swap completed!');
    } catch (e) {
        setStatus('img-status', `❌ ${e.message}`);
    }
}

// ═══════════════════════════════════════════════════════
// VIDEO SWAP
// ═══════════════════════════════════════════════════════

let videoProgressInterval = null;

function startVideoProgress(sid) {
    const bar = document.getElementById('vid-progress');
    const fill = bar.querySelector('.progress-fill');
    bar.style.display = 'block';
    fill.style.width = '0%';

    videoProgressInterval = setInterval(async () => {
        try {
            const res = await fetch(`${serverUrl}/video-progress/${sid}`);
            if (!res.ok) return;
            const data = await res.json();
            const pct = Math.round(data.progress * 100);
            fill.style.width = `${pct}%`;

            const stageLabels = {
                uploading: '📤 Uploading...',
                processing: '⚙️ Initializing...',
                swapping: `🔄 Swapping faces... ${pct}%`,
                muxing_audio: '🔊 Preserving audio...',
                complete: '✅ Complete!',
                failed: '❌ Failed',
                error: '❌ Error',
            };
            setStatus('vid-status', stageLabels[data.stage] || `⏳ ${data.stage} ${pct}%`);

            if (data.done) stopVideoProgress();
        } catch (_) { /* ignore poll errors */ }
    }, 500);
}

function stopVideoProgress() {
    if (videoProgressInterval) {
        clearInterval(videoProgressInterval);
        videoProgressInterval = null;
    }
}

async function swapVideo() {
    const sourceFiles = document.getElementById('vid-source-files').files;
    const targetFile = document.getElementById('vid-target-file').files[0];

    if (sourceFiles.length === 0) return setStatus('vid-status', '❌ Upload source face images');
    if (!targetFile) return setStatus('vid-status', '❌ Upload target video');

    setStatus('vid-status', '⏳ Uploading video...');

    // Generate a dedicated progress session ID so client and server track the same sid
    const videoSid = sessionId || `vid_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    try {
        const form = new FormData();
        for (const f of sourceFiles) form.append('source_files', f);
        form.append('target_file', targetFile);
        form.append('enhance', document.getElementById('vid-enhance').checked);
        form.append('swap_all', document.getElementById('vid-swap-all').checked);
        form.append('mouth_mask', document.getElementById('vid-mouth').checked);
        form.append('sharpness', document.getElementById('vid-sharp').value);
        form.append('session_id', videoSid); // Always send so server uses same ID

        // Start polling progress with the SAME sid the server will use
        startVideoProgress(videoSid);

        const res = await fetch(`${serverUrl}/swap-video`, {
            method: 'POST',
            body: form,
        });

        stopVideoProgress();

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Video processing failed');
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        const vid = document.getElementById('vid-result');
        // Revoke previous result URL to prevent memory leak
        if (vid.src && vid.src.startsWith('blob:')) URL.revokeObjectURL(vid.src);
        vid.src = url;
        vid.style.display = 'block';

        const dl = document.getElementById('vid-download');
        if (dl.href.startsWith('blob:')) URL.revokeObjectURL(dl.href);
        dl.href = url;
        dl.style.display = 'inline-block';

        // Fill progress bar to 100%
        const fill = document.getElementById('vid-progress').querySelector('.progress-fill');
        fill.style.width = '100%';

        setStatus('vid-status', '✅ Video processing completed! (audio preserved)');
    } catch (e) {
        stopVideoProgress();
        setStatus('vid-status', `❌ ${e.message}`);
    }
}

// ═══════════════════════════════════════════════════════
// FACE DETECTION
// ═══════════════════════════════════════════════════════

async function detectFaces() {
    const file = document.getElementById('detect-file').files[0];
    if (!file) return setStatus('detect-status', '❌ Upload an image first');

    setStatus('detect-status', '⏳ Detecting faces...');

    try {
        const form = new FormData();
        form.append('file', file);

        const res = await fetch(`${serverUrl}/detect-faces`, {
            method: 'POST',
            body: form,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Detection failed');
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        const img = document.getElementById('detect-result');
        // Revoke previous result URL to prevent memory leak
        if (img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
        img.src = url;
        img.style.display = 'block';

        setStatus('detect-status', '✅ Detection complete');
    } catch (e) {
        setStatus('detect-status', `❌ ${e.message}`);
    }
}

// ═══════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════

// Auto-connect on page load
window.addEventListener('load', async () => {
    // Try to get server URL from query params
    const params = new URLSearchParams(window.location.search);
    const server = params.get('server');
    if (server) {
        document.getElementById('server-url').value = server;
    }

    await connectToServer();
    await enumerateCameras();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (isStreaming) stopStream();
});
