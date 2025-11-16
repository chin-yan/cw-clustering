# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math
import facenet.src.align.detect_face as detect_face

def detect_faces_adjusted(sess, frame_paths, output_dir, min_face_size=60, face_size=160,
                         margin=44, detect_multiple_faces=True):
    """
    Use unified foreground face detection logic
    """
    print("Creating MTCNN network for foreground-focused face detection...")
    pnet, rnet, onet = create_mtcnn(sess, None)
    
    # Unified detection parameters
    min_face_area_ratio = 0.008  # Face area must occupy at least 0.8% of image
    max_faces_per_frame = 5      # Keep at most 5 faces per frame
    
    face_paths = []
    face_count = 0
    
    print("Detecting foreground faces from frames...")
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        original_frame = frame.copy()
        
        # Use unified detection function
        filtered_bboxes = detect_foreground_faces_in_frame(
            frame, pnet, rnet, onet, 
            min_face_size=min_face_size,
            min_face_area_ratio=min_face_area_ratio,
            max_faces_per_frame=max_faces_per_frame
        )
        
        # Process filtered faces
        for i, bbox in enumerate(filtered_bboxes):
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            adaptive_margin = int(margin * bbox_size / 160)
            
            x1 = max(0, bbox[0] - adaptive_margin)
            y1 = max(0, bbox[1] - adaptive_margin)
            x2 = min(original_frame.shape[1], bbox[2] + adaptive_margin)
            y2 = min(original_frame.shape[0], bbox[3] + adaptive_margin)
            
            face = original_frame[y1:y2, x1:x2, :]
            
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                continue
            
            face = mild_preprocessing(face)
            face_resized = cv2.resize(face, (face_size, face_size))
            
            frame_name = os.path.basename(frame_path)
            face_name = f"{os.path.splitext(frame_name)[0]}_mainface_{i}.jpg"
            face_path = os.path.join(output_dir, face_name)
            
            cv2.imwrite(face_path, face_resized)
            face_paths.append(face_path)
            face_count += 1
    
    print(f"A total of {face_count} main character faces were detected")
    return face_paths

def create_mtcnn(sess, model_path):
    """
    Create MTCNN detection network
    
    Args:
        sess: TensorFlow session
        model_path: Path to MTCNN model
        
    Returns:
        MTCNN detection components
    """
    if not model_path:
        model_path = None
    
    pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)
    return pnet, rnet, onet

def mild_contrast_enhancement(image, clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    Apply a mild contrast enhancement to help detection without degrading quality
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting (reduced from previous)
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into different channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel with mild settings
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the original A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to RGB color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return enhanced

def mild_preprocessing(face_img):
    """
    Apply mild preprocessing to improve face image quality without degradation
    
    Args:
        face_img: Input face image
        
    Returns:
        Mildly preprocessed face image
    """
    # Check if image is valid
    if face_img is None or face_img.size == 0:
        return face_img
    
    # -----------------------------
    # Step 1: Adaptive Contrast Enhancement (CLAHE)
    # -----------------------------
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 80:        # Too dark → enhance contrast
        clip_limit = 1.2
    elif brightness > 180:     # Too bright → reduce contrast
        clip_limit = 0.6
    else:                      # Normal → medium contrast
        clip_limit = 0.8

    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # -----------------------------
    # Step 2: Adaptive Denoising (NL-means)
    # -----------------------------
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < 50:   # Blurry → reduce denoising to avoid more blur
        h = 3
    elif variance > 150: # Very clear → can apply stronger denoising
        h = 7
    else:               # Medium → medium strength
        h = 5

    img_denoised = cv2.fastNlMeansDenoisingColored(
        enhanced, None,
        h, h, 7, 21
    )
    
    # No further processing to preserve original qualities
    return img_denoised

def detect_foreground_faces_in_frame(frame, pnet, rnet, onet, min_face_size=60, 
                                    min_face_area_ratio=0.008, max_faces_per_frame=5):
    """
    Unified foreground face detection logic, shared by clustering and annotation stages
    
    Args:
        frame: Input image (BGR format)
        pnet, rnet, onet: MTCNN detector components
        min_face_size: Minimum face size
        min_face_area_ratio: Minimum ratio of face area to image area
        max_faces_per_frame: Maximum number of faces to keep per frame
        
    Returns:
        filtered_bboxes: List of filtered face bounding boxes
    """
    import facenet.src.align.detect_face as detect_face
    
    # Convert to RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Calculate image area
    frame_area = frame.shape[0] * frame.shape[1]
    
    # Apply mild contrast enhancement (consistent with clustering stage)
    frame_rgb = mild_contrast_enhancement(frame_rgb)
    
    # Use same detection parameters as clustering stage
    threshold = [0.7, 0.8, 0.8]  # Stricter thresholds
    
    # Primary detection
    bounding_boxes, _ = detect_face.detect_face(
        frame_rgb, min_face_size, pnet, rnet, onet, threshold, 0.709
    )
    
    # If no detection, try again with lower threshold (consistent with clustering stage)
    if len(bounding_boxes) == 0:
        side_threshold = [0.5, 0.6, 0.7] 
        bounding_boxes, _ = detect_face.detect_face(
            frame_rgb, min_face_size * 0.8, pnet, rnet, onet, side_threshold, 0.6
        )
    
    # Filter and sort faces (completely consistent with clustering stage)
    valid_faces = []
    for i, bbox in enumerate(bounding_boxes):
        bbox = bbox.astype(np.int)
        
        # Calculate face area
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_area = face_width * face_height
        face_area_ratio = face_area / frame_area
        
        # Filter faces that are too small
        if face_area_ratio >= min_face_area_ratio:
            valid_faces.append({
                'bbox': bbox,
                'area': face_area,
                'area_ratio': face_area_ratio,
                'index': i
            })
    
    # Sort by area and keep the largest ones
    valid_faces.sort(key=lambda x: x['area'], reverse=True)
    valid_faces = valid_faces[:max_faces_per_frame]
    
    # Return filtered bounding boxes
    filtered_bboxes = [face_info['bbox'] for face_info in valid_faces]
    
    # Debug information
    if len(bounding_boxes) > len(filtered_bboxes):
        print(f" Face filtering: {len(bounding_boxes)} → {len(filtered_bboxes)} (keeping main foreground characters)")
    
    return filtered_bboxes