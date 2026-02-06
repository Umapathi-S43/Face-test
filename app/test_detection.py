#!/usr/bin/env python3
"""Test face detection with the uploaded images"""

import cv2
import numpy as np
import glob
import os

def test_detection():
    # Find PNG files in gradio temp folder
    pngs = glob.glob('/private/var/folders/gt/dxcj23nd70551__wnyw78dgm0000gn/T/gradio/*/*.png')
    print(f'Found {len(pngs)} png files in gradio temp')
    
    # Also check for local test images
    local_images = glob.glob('./*.jpg') + glob.glob('./*.png')
    print(f'Found {len(local_images)} local images')
    
    test_images = pngs + local_images
    
    if not test_images:
        print("No test images found!")
        return
    
    # Test with OpenCV Haar cascade first
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for test_path in test_images[:3]:  # Test first 3 images
        print(f'\n--- Testing: {os.path.basename(test_path)} ---')
        
        # Read image
        img = cv2.imread(test_path)
        if img is None:
            img = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print('  Failed to load image')
            continue
            
        print(f'  Original shape: {img.shape}, dtype: {img.dtype}')
        
        # Handle RGBA
        if len(img.shape) == 3 and img.shape[2] == 4:
            print('  Converting RGBA to BGR...')
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Save original size
        orig_h, orig_w = img.shape[:2]
        
        # Resize for detection
        max_size = 640
        if max(orig_h, orig_w) > max_size:
            scale = max_size / max(orig_h, orig_w)
            img_resized = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
            print(f'  Resized to: {img_resized.shape}')
        else:
            img_resized = img
            
        print(f'  Pixel range: {img_resized.min()} - {img_resized.max()}')
        
        # Test OpenCV Haar
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        print(f'  OpenCV Haar: {len(faces_haar)} faces')
        
        # Test InsightFace
        try:
            from insightface.app import FaceAnalysis
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            app = FaceAnalysis(
                name='buffalo_l', 
                root='./models', 
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)
            
            faces_if = app.get(img_rgb)
            print(f'  InsightFace: {len(faces_if)} faces')
            
            if faces_if:
                for i, face in enumerate(faces_if):
                    bbox = face.bbox.astype(int)
                    print(f'    Face {i}: bbox={bbox}')
        except Exception as e:
            print(f'  InsightFace error: {e}')

if __name__ == '__main__':
    test_detection()
