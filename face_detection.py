# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils import face_align # Import alignment tool

def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Lightweight sharpening: Unsharp Masking
    More natural than direct kernel sharpening, does not overly destroy image structure.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, 0)
    sharpened = np.minimum(sharpened, 255)
    return sharpened.astype(np.uint8)

def detect_faces_with_insightface(frame_paths, output_dir, min_face_size=30, face_size=112, margin_ratio=0.0):
    """
    Detect AND Align faces using InsightFace (SCRFD).
    CRITICAL: Uses norm_crop to align face landmarks, which is required for ArcFace accuracy.
    """
    print("Initializing InsightFace Detector (SCRFD)...")
    
    # Load detection model
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    face_paths_out = []
    face_count = 0
    
    print("Detecting and aligning faces...")
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        try:
            # 1. Detect faces
            faces = app.get(frame)
        except Exception:
            continue
        
        for i, face in enumerate(faces):
            # Check face size (bbox height)
            bbox = face.bbox
            if (bbox[3] - bbox[1]) < min_face_size:
                continue
            
            # 2. Key step: Face Alignment
            # ArcFace relies heavily on facial feature alignment; simple cropping yields poor results.
            # norm_crop aligns and crops the face to a standard 112x112 based on 5 key points (kps).
            try:
                align_img = face_align.norm_crop(frame, landmark=face.kps, image_size=112)
            except Exception:
                continue
                
            if align_img is None or align_img.size == 0:
                continue
            
            align_img = apply_unsharp_mask(align_img, amount=1.0)

            # 3. Save aligned images
            frame_name = os.path.basename(frame_path)
            face_name = f"{os.path.splitext(frame_name)[0]}_face_{i}.jpg"
            save_path = os.path.join(output_dir, face_name)
            
            cv2.imwrite(save_path, align_img)
            face_paths_out.append(save_path)
            face_count += 1
            
    print(f"A total of {face_count} aligned faces were detected and saved")
    return face_paths_out