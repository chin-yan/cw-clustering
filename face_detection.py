# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import facenet.src.align.detect_face as detect_face

def detect_faces_in_frames(sess, frame_paths, output_dir, min_face_size=20, face_size=160):
    """
    Detecting faces from frames using MTCNN

    Args:
    sess: TensorFlow session
    frame_paths: frame path list
    output_dir: The directory where the detected faces are saved
    min_face_size: minimum face size
    face_size: The size of the output face image

    Returns:
    Path list of detected face images
    """
    print("Creating MTCNN network...")
    pnet, rnet, onet = create_mtcnn(sess, None)
    
    face_paths = []
    face_count = 0
    
    print("Detecting faces from frame...")
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = detect_face.detect_face(
            frame_rgb, min_face_size, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.7
        )
        for i, bbox in enumerate(bounding_boxes):
            bbox = bbox.astype(np.int)
            # Increase the bounding box size to include more facial features
            x1 = max(0, bbox[0] - 10)
            y1 = max(0, bbox[1] - 10)
            x2 = min(frame.shape[1], bbox[2] + 10)
            y2 = min(frame.shape[0], bbox[3] + 10)
            
            # Extracting faces
            face = frame[y1:y2, x1:x2, :]
            
            # Checking the validity of face images
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                continue
                
            # Resize
            face_resized = cv2.resize(face, (face_size, face_size))
            
            # Generate output path
            frame_name = os.path.basename(frame_path)
            face_name = f"{os.path.splitext(frame_name)[0]}_face_{i}.jpg"
            face_path = os.path.join(output_dir, face_name)
            
            # Save face
            cv2.imwrite(face_path, face_resized)
            face_paths.append(face_path)
            face_count += 1
    
    print(f"A total of {face_count} faces were detected")
    return face_paths

def create_mtcnn(sess, model_path):
    if not model_path:
        model_path = None
    
    pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)
    return pnet, rnet, onet