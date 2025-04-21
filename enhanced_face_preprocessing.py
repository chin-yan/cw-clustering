# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math
import facenet.src.align.detect_face as detect_face

def detect_faces_adjusted(sess, frame_paths, output_dir, min_face_size=20, face_size=160,
                         margin=44, detect_multiple_faces=True):
    """
    Adjusted face detection and preprocessing with more conservative enhancement
    to avoid quality degradation while still improving side face detection
    
    Args:
        sess: TensorFlow session
        frame_paths: frame path list
        output_dir: The directory where the detected faces are saved
        min_face_size: minimum face size
        face_size: The size of the output face image
        margin: Margin for the crop around the bounding box (in pixels)
        detect_multiple_faces: If true, detect multiple faces in an image
        
    Returns:
        Path list of detected face images
    """
    print("Creating MTCNN network for adjusted face detection...")
    pnet, rnet, onet = create_mtcnn(sess, None)
    
    # Parameters for better side face detection - less aggressive than before
    threshold = [0.6, 0.7, 0.8]  # Default is [0.6, 0.7, 0.7] - minor adjustment
    
    face_paths = []
    face_count = 0
    
    print("Detecting faces from frames with adjusted preprocessing...")
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Keep a copy of the original frame for face extraction
        original_frame = frame.copy()
        
        # Convert to RGB for detection (keep original colors for extraction)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply mild contrast enhancement to help with detection
        frame_rgb = mild_contrast_enhancement(frame_rgb)
        
        # Main face detection
        bounding_boxes, _ = detect_face.detect_face(
            frame_rgb, min_face_size, pnet, rnet, onet, threshold, 0.709
        )
        
        for i, bbox in enumerate(bounding_boxes):
            bbox = bbox.astype(np.int)
            
            # Calculate margin for better face crop
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            adaptive_margin = int(margin * bbox_size / 160)
            
            # Increase the bounding box size to include more facial features
            x1 = max(0, bbox[0] - adaptive_margin)
            y1 = max(0, bbox[1] - adaptive_margin)
            x2 = min(original_frame.shape[1], bbox[2] + adaptive_margin)
            y2 = min(original_frame.shape[0], bbox[3] + adaptive_margin)
            
            # Extract face from the original frame to preserve quality
            face = original_frame[y1:y2, x1:x2, :]
            
            # Checking the validity of face images
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                continue
            
            # Apply mild preprocessing - more conservative than before
            face = mild_preprocessing(face)
                
            # Resize to specified size
            face_resized = cv2.resize(face, (face_size, face_size))
            
            # Generate output path
            frame_name = os.path.basename(frame_path)
            face_name = f"{os.path.splitext(frame_name)[0]}_face_{i}.jpg"
            face_path = os.path.join(output_dir, face_name)
            
            # Save face
            cv2.imwrite(face_path, face_resized)
            face_paths.append(face_path)
            face_count += 1
        
        # If no faces detected with standard settings, try a second pass with different parameters
        # but only if no faces were found in the first pass
        if len(bounding_boxes) == 0:
            # Secondary detection with parameters more suitable for side faces
            side_threshold = [0.5, 0.6, 0.7]  # Lower thresholds for side faces
            side_bounding_boxes, _ = detect_face.detect_face(
                frame_rgb, min_face_size * 0.8, pnet, rnet, onet, side_threshold, 0.6
            )
            
            for i, bbox in enumerate(side_bounding_boxes):
                bbox = bbox.astype(np.int)
                
                # Calculate margin
                bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                adaptive_margin = int(margin * bbox_size / 160)
                
                # Increase the bounding box size
                x1 = max(0, bbox[0] - adaptive_margin)
                y1 = max(0, bbox[1] - adaptive_margin)
                x2 = min(original_frame.shape[1], bbox[2] + adaptive_margin)
                y2 = min(original_frame.shape[0], bbox[3] + adaptive_margin)
                
                # Extract face from original frame
                face = original_frame[y1:y2, x1:x2, :]
                
                # Check validity
                if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                    continue
                
                # Apply mild preprocessing
                face = mild_preprocessing(face)
                    
                # Resize
                face_resized = cv2.resize(face, (face_size, face_size))
                
                # Generate output path
                frame_name = os.path.basename(frame_path)
                face_name = f"{os.path.splitext(frame_name)[0]}_sideface_{i}.jpg"
                face_path = os.path.join(output_dir, face_name)
                
                # Save face
                cv2.imwrite(face_path, face_resized)
                face_paths.append(face_path)
                face_count += 1
    
    print(f"A total of {face_count} faces were detected with adjusted preprocessing")
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
    
    # Apply mild denoising (removed gamma correction)
    img_denoised = cv2.fastNlMeansDenoisingColored(
        face_img, 
        None, 
        h=5,       # Filter strength (reduced for less smoothing)
        hColor=5,  # Same value for color components
        templateWindowSize=7,  # Size of template patch
        searchWindowSize=21    # Size of search window
    )
    
    # No further processing to preserve original qualities
    return img_denoised