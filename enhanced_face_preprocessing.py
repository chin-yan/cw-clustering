# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math
import json
import re
import facenet.src.align.detect_face as detect_face

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def compute_face_quality(face_path):
    """
    Compute the quality score for a face image with more permissive criteria
    Higher scores indicate better quality (more frontal, better lighting)
    
    Args:
        face_path: Path to the face image
        
    Returns:
        Quality score (0-1)
    """
    # Load the image
    img = cv2.imread(face_path)
    if img is None:
        return 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute quality metrics
    # 1. Variance of Laplacian (measure of image sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian) / 100.0  # Normalize
    sharpness = min(1.0, sharpness)  # Cap at 1.0
    
    # 2. Histogram spread (measure of contrast)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()  # Normalize histogram
    non_zero_bins = np.count_nonzero(hist_norm > 0.0005)  # More permissive threshold
    contrast = non_zero_bins / 256.0
    
    # 3. Face size relative to image size (larger faces are usually better)
    height, width = gray.shape
    face_area = height * width
    face_size_score = min(1.0, face_area / (160.0 * 160.0))
    
    # Combine metrics (adjusted weights - less emphasis on sharpness)
    quality_score = 0.35 * sharpness + 0.35 * contrast + 0.3 * face_size_score
    
    # Make overall scoring more permissive
    quality_score = min(1.0, quality_score * 1.2)  # Boost scores by 20%
    
    return quality_score

def detect_faces_adjusted(sess, frame_paths, faces_output_dir, frames_output_dir, 
                         min_face_size=20, face_size=160, margin=44, 
                         detect_multiple_faces=True, 
                         coordinate_system=None, output_metadata_path=None):
    """
    Adjusted face detection and preprocessing with coordinate system tracking
    
    Args:
        sess: TensorFlow session
        frame_paths: frame path list
        faces_output_dir: The directory where the detected faces are saved
        frames_output_dir: The directory where frames are saved
        min_face_size: minimum face size
        face_size: The size of the output face image
        margin: Margin for the crop around the bounding box (in pixels)
        detect_multiple_faces: If true, detect multiple faces in an image
        coordinate_system: Dictionary with 'width' and 'height' for standard coordinates
        output_metadata_path: Path to save the detection coordinate system metadata
        
    Returns:
        Path list of detected face images and detection metadata
    """
    print("Creating MTCNN network for adjusted face detection...")
    pnet, rnet, onet = create_mtcnn(sess, None)
    
    # Create output directories if they don't exist
    if not os.path.exists(faces_output_dir):
        os.makedirs(faces_output_dir)
    if not os.path.exists(frames_output_dir):
        os.makedirs(frames_output_dir)
    
    # Parameters for better side face detection
    threshold = [0.6, 0.7, 0.8]  # Default is [0.6, 0.7, 0.7] - minor adjustment
    
    face_paths = []
    face_count = 0
    
    # Store detection metadata
    detection_metadata = {
        "coordinate_system": {},
        "face_detections": []
    }
    
    # Get dimensions from first frame if not provided
    if coordinate_system is None or coordinate_system["width"] == 0:
        if frame_paths:
            first_frame = cv2.imread(frame_paths[0])
            if first_frame is not None:
                coordinate_system = {
                    "width": first_frame.shape[1],
                    "height": first_frame.shape[0]
                }
    
    # Store coordinate system info
    detection_metadata["coordinate_system"] = coordinate_system
    print(f"Using coordinate system: {coordinate_system['width']}x{coordinate_system['height']}")
    
    print("Detecting faces from frames with adjusted preprocessing...")
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Get the base name of the frame
        frame_name = os.path.basename(frame_path)
        
        # Extract frame ID from filename
        # Expected format: frame_XXXXXX.jpg
        frame_id = -1
        match = re.match(r'frame_(\d+)\.jpg', frame_name)
        if match:
            frame_id = int(match.group(1))
        
        # Copy the frame to frames_output_dir
        frame_out_path = os.path.join(frames_output_dir, frame_name)
        cv2.imwrite(frame_out_path, frame)

        # Keep a copy of the original frame for face extraction
        original_frame = frame.copy()
        
        # Get current frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate scale factors if coordinate systems differ
        scale_x = 1.0
        scale_y = 1.0
        if coordinate_system and coordinate_system["width"] > 0:
            scale_x = coordinate_system["width"] / frame_width
            scale_y = coordinate_system["height"] / frame_height
        
        # Convert to RGB for detection (keep original colors for extraction)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply mild contrast enhancement to help with detection
        frame_rgb = mild_contrast_enhancement(frame_rgb)
        
        # Main face detection
        bounding_boxes, _ = detect_face.detect_face(
            frame_rgb, min_face_size, pnet, rnet, onet, threshold, 0.709
        )
        
        frame_detections = []
        
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
            
            # Scale the coordinates to match the coordinate system
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y)
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y)
            
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
            face_name = f"{os.path.splitext(frame_name)[0]}_face_{i}.jpg"
            face_path = os.path.join(faces_output_dir, face_name)
            
            # Save face
            cv2.imwrite(face_path, face_resized)
            face_paths.append(face_path)
            face_count += 1
            
            # Store detection info
            face_detection = {
                "frame_id": frame_id,
                "face_idx": i,
                "face_path": face_path,
                "bbox": [scaled_x1, scaled_y1, scaled_x2, scaled_y2],  # Store in standardized coordinates
                "original_bbox": [x1, y1, x2, y2],  # Store original coordinates
                "quality": compute_face_quality(face_path)
            }
            
            frame_detections.append(face_detection)
        
        # Add frame detections to metadata
        if frame_detections:
            detection_metadata["face_detections"].extend(frame_detections)
    
    print(f"A total of {face_count} faces were detected with adjusted preprocessing")
    
    # Save detection metadata if path provided
    if output_metadata_path:
        with open(output_metadata_path, 'w') as f:
            json.dump(detection_metadata, f, indent=2, cls=NumpyJSONEncoder)
        print(f"Detection metadata saved to {output_metadata_path}")
    
    return face_paths, detection_metadata


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