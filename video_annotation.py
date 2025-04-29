# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
import time
import math
import facenet

import face_detection
import feature_extraction

def load_centers_data(centers_data_path):
    """
    Load previously saved cluster center data
    
    Args:
        centers_data_path: Path to the cluster center data
        
    Returns:
        Dictionary containing cluster information
    """
    print("Loading cluster center data...")
    with open(centers_data_path, 'rb') as f:
        centers_data = pickle.load(f)
    
    # Check data integrity
    if 'cluster_centers' not in centers_data:
        raise ValueError("Missing cluster center information in the data")
    
    centers, center_paths = centers_data['cluster_centers']
    print(f"Successfully loaded {len(centers)} cluster centers")
    return centers, center_paths, centers_data

def match_face_with_centers(face_encoding, centers, threshold=0.8):
    """
    Match a face encoding with the cluster centers
    
    Args:
        face_encoding: Face encoding vector
        centers: List of cluster center encodings
        threshold: Similarity threshold
        
    Returns:
        Index of the matched center and similarity score, or (-1, 0) if no match
    """
    if len(centers) == 0:
        return -1, 0
    
    # Calculate cosine similarity (dot product) with all centers
    similarities = np.dot(centers, face_encoding)
    
    # Find the most similar center
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]
    
    # Return match if similarity exceeds threshold
    if best_similarity > threshold:
        return best_index, best_similarity
    else:
        return -1, 0

def create_mtcnn_detector(sess):
    """
    Create MTCNN face detector
    
    Args:
        sess: TensorFlow session
        
    Returns:
        MTCNN detector components
    """
    print("Creating MTCNN detector...")
    import facenet.src.align.detect_face as detect_face
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet

def detect_and_match_faces(frame, pnet, rnet, onet, sess, images_placeholder, 
                         embeddings, phase_train_placeholder, centers, min_face_size=20):
    """
    Detect faces in a frame and match them with cluster centers
    
    Args:
        frame: Input video frame
        pnet, rnet, onet: MTCNN detector components
        sess: TensorFlow session
        images_placeholder: Input placeholder for FaceNet
        embeddings: Output embeddings tensor
        phase_train_placeholder: Phase train placeholder
        centers: Cluster center encodings
        min_face_size: Minimum face size for detection
        
    Returns:
        List of detected faces with bounding boxes and matched center indices
    """
    import facenet.src.align.detect_face as detect_face
    
    # Convert frame to RGB (MTCNN uses RGB)
    if frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Detect faces
    bounding_boxes, _ = detect_face.detect_face(
        frame_rgb, min_face_size, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.7
    )
    
    faces = []
    
    # Process each detected face
    for bbox in bounding_boxes:
        bbox = bbox.astype(np.int)
        
        # Extract face area with some margin
        x1 = max(0, bbox[0] - 10)
        y1 = max(0, bbox[1] - 10)
        x2 = min(frame.shape[1], bbox[2] + 10)
        y2 = min(frame.shape[0], bbox[3] + 10)
        
        face = frame_rgb[y1:y2, x1:x2, :]
        
        # Skip invalid faces
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue
            
        # Resize to FaceNet input size
        face_resized = cv2.resize(face, (160, 160))
        
        # Preprocess for FaceNet
        face_prewhitened = facenet.src.facenet.prewhiten(face_resized)
        face_input = face_prewhitened.reshape(-1, 160, 160, 3)
        
        # Get face encoding
        feed_dict = {images_placeholder: face_input, phase_train_placeholder: False}
        face_encoding = sess.run(embeddings, feed_dict=feed_dict)[0]
        
        # Match with centers
        match_idx, similarity = match_face_with_centers(face_encoding, centers)
        
        # Store face info
        faces.append({
            'bbox': (x1, y1, x2, y2),
            'match_idx': match_idx,
            'similarity': similarity
        })
    
    return faces

def annotate_video(input_video, output_video, centers_data_path, model_dir):
    """
    Annotate video with face identities based on cluster centers
    
    Args:
        input_video: Input video path
        output_video: Output video path
        centers_data_path: Path to cluster center data
        model_dir: FaceNet model directory
    """
    # Load centers data
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Create face detector
            pnet, rnet, onet = create_mtcnn_detector(sess)
            
            # Load FaceNet model
            print("Loading FaceNet model...")
            model_dir = os.path.expanduser(model_dir)
            feature_extraction.load_model(sess, model_dir)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            # Process video frames
            print(f"Processing video with {total_frames} frames...")
            frame_count = 0
            processing_times = []
            
            # For progress tracking
            pbar = tqdm(total=total_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Process every 2 frames to speed up (can be adjusted)
                if frame_count % 2 == 0:
                    # Detect and match faces
                    faces = detect_and_match_faces(
                        frame, pnet, rnet, onet, sess, 
                        images_placeholder, embeddings, phase_train_placeholder, 
                        centers
                    )
                    
                    # Annotate frame
                    for face in faces:
                        x1, y1, x2, y2 = face['bbox']
                        match_idx = face['match_idx']
                        similarity = face['similarity']
                        
                        # Draw bounding box
                        if match_idx >= 0:
                            # Matched face - green box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Display identity information
                            label = f"ID: {match_idx}, Sim: {similarity:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # Unmatched face - red box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            # Display unknown label
                            cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Write frame to output
                out.write(frame)
                
                end_time = time.time()
                processing_times.append(end_time - start_time)
                
                # Update progress
                frame_count += 1
                pbar.update(1)
                
                # Print processing stats every 50 frames
                if frame_count % 50 == 0:
                    avg_time = np.mean(processing_times[-50:])
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"\nFrame {frame_count}/{total_frames}, Avg. processing time: {avg_time:.3f}s, FPS: {fps:.2f}")
            
            pbar.close()
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video annotation completed. Output saved to {output_video}")

if __name__ == "__main__":
    # Configuration
    input_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\annotated_video.avi"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\centers\centers_data.pkl"
    model_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759"
    
    # Run video annotation
    annotate_video(
        input_video=input_video,
        output_video=output_video,
        centers_data_path=centers_data_path,
        model_dir=model_dir
    )