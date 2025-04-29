# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
import time
import math
import dlib
from scipy.spatial import distance

import face_detection
import feature_extraction
import face_retrieval

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

def init_facial_landmark_detector():
    """
    Initialize the facial landmark detector (dlib)
    
    Returns:
        Dlib facial landmark predictor
    """
    print("Initializing facial landmark detector...")
    # Download model file if not exists
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print(f"Facial landmark model not found at {model_path}")
        print("Please download the model from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place in the current directory")
        raise FileNotFoundError(f"Missing required model file: {model_path}")
    
    return dlib.shape_predictor(model_path)

def detect_faces_with_landmarks(frame, pnet, rnet, onet, landmark_predictor, min_face_size=20):
    """
    Detect faces and their facial landmarks in a frame
    
    Args:
        frame: Input video frame
        pnet, rnet, onet: MTCNN detector components
        landmark_predictor: Dlib facial landmark predictor
        min_face_size: Minimum face size for detection
        
    Returns:
        List of detected faces with bounding boxes and facial landmarks
    """
    import facenet.src.align.detect_face as detect_face
    
    # Convert frame to RGB (MTCNN uses RGB)
    if frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Convert to grayscale for dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
        
        # Convert bbox to dlib rectangle format
        rect = dlib.rectangle(x1, y1, x2, y2)
        
        # Get facial landmarks
        shape = landmark_predictor(gray, rect)
        landmarks = []
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks.append((x, y))
        
        # Get mouth landmarks (indices 48-68)
        mouth_landmarks = landmarks[48:68]
        
        # Store face info
        faces.append({
            'bbox': (x1, y1, x2, y2),
            'landmarks': landmarks,
            'mouth_landmarks': mouth_landmarks,
            'rect': rect
        })
    
    return faces

def compute_face_encodings_for_frame(frame, faces, sess, images_placeholder, 
                                    embeddings, phase_train_placeholder):
    """
    Compute facial feature encodings for faces in a frame
    
    Args:
        frame: Input video frame
        faces: List of detected faces with bounding boxes
        sess: TensorFlow session
        images_placeholder: Input placeholder for FaceNet
        embeddings: Output embeddings tensor
        phase_train_placeholder: Phase train placeholder
        
    Returns:
        Updated faces list with encoding information
    """
    import facenet.src.facenet as facenet
    
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        
        # Extract face area
        face_img = frame[y1:y2, x1:x2, :]
        
        # Skip invalid faces
        if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            face['encoding'] = None
            continue
            
        # Resize to FaceNet input size
        face_resized = cv2.resize(face_img, (160, 160))
        
        # Preprocess for FaceNet
        face_prewhitened = facenet.prewhiten(face_resized)
        face_input = face_prewhitened.reshape(-1, 160, 160, 3)
        
        # Get face encoding
        feed_dict = {images_placeholder: face_input, phase_train_placeholder: False}
        face_encoding = sess.run(embeddings, feed_dict=feed_dict)[0]
        
        # Update face info
        face['encoding'] = face_encoding
    
    return faces

def detect_speaking_face(prev_faces, curr_faces, threshold=1.5):
    """
    Detect which face is currently speaking by analyzing mouth movement
    with focus on significant mouth changes like teeth exposure
    
    Args:
        prev_faces: Faces detected in previous frame
        curr_faces: Faces detected in current frame
        threshold: Mouth movement threshold (increased to detect only significant movements)
        
    Returns:
        Index of speaking face, or -1 if none
    """
    if not prev_faces or not curr_faces:
        return -1
    
    max_movement = 0
    speaking_idx = -1
    
    # Match faces between frames
    for curr_idx, curr_face in enumerate(curr_faces):
        curr_encoding = curr_face.get('encoding')
        if curr_encoding is None:
            continue
        
        # Find the same face in previous frame
        best_match_idx = -1
        best_match_sim = 0
        
        for prev_idx, prev_face in enumerate(prev_faces):
            prev_encoding = prev_face.get('encoding')
            if prev_encoding is None:
                continue
            
            # Calculate similarity
            similarity = np.dot(curr_encoding, prev_encoding)
            
            if similarity > best_match_sim:
                best_match_sim = similarity
                best_match_idx = prev_idx
        
        # If we found a match, calculate mouth movement with emphasis on significant changes
        if best_match_idx >= 0 and best_match_sim > 0.8:
            prev_mouth = prev_faces[best_match_idx]['mouth_landmarks']
            curr_mouth = curr_face['mouth_landmarks']
            
            # Calculate overall mouth movement (average landmark displacement)
            movement = 0
            for i in range(len(prev_mouth)):
                movement += distance.euclidean(prev_mouth[i], curr_mouth[i])
            movement /= len(prev_mouth)
            
            # Calculate vertical mouth opening (distance between upper and lower lip)
            # Upper lip middle point (index 51) and lower lip middle point (index 57)
            # These are mapped to indices 3 and 9 in our mouth_landmarks array
            prev_mouth_opening = distance.euclidean(prev_mouth[3], prev_mouth[9])
            curr_mouth_opening = distance.euclidean(curr_mouth[3], curr_mouth[9])
            mouth_opening_change = abs(curr_mouth_opening - prev_mouth_opening)
            
            # Boost movement score if mouth opening changes significantly (likely speaking with teeth showing)
            if mouth_opening_change > 2.0:
                movement *= 1.5
            
            # Update max movement
            if movement > max_movement:
                max_movement = movement
                speaking_idx = curr_idx
    
    # Return speaking face index if movement exceeds threshold (now higher to avoid detecting small movements)
    if max_movement > threshold:
        return speaking_idx
    else:
        return -1

def annotate_video_with_speaking_face(input_video, output_video, centers_data_path, 
                                     model_dir, detection_interval=2):
    """
    Annotate video with only the speaking face identified and labeled
    
    Args:
        input_video: Input video path
        output_video: Output video path
        centers_data_path: Path to cluster center data
        model_dir: FaceNet model directory
        detection_interval: Interval for face detection (process every N frames)
    """
    # Load centers data
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    # Build Annoy index for faster retrieval (using same approach as in face_retrieval.py)
    print("Building Annoy index for fast retrieval...")
    from annoy import AnnoyIndex
    
    embedding_size = centers[0].shape[0]
    annoy_index = AnnoyIndex(embedding_size, 'euclidean')
    
    # Add cluster centers to the index
    for i, center in enumerate(centers):
        annoy_index.add_item(i, center)
    
    # Build the index with 10 trees
    annoy_index.build(10)
    print("Annoy index built successfully")
    
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
    
    # Tracking variables for stable detection
    speaking_face_counter = {}  # Count consecutive frames a face is speaking
    stable_speaking_face_idx = -1
    consecutive_frames_threshold = 3  # Number of consecutive frames to confirm speaking
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Create face detector
            pnet, rnet, onet = create_mtcnn_detector(sess)
            
            # Initialize facial landmark detector
            landmark_predictor = init_facial_landmark_detector()
            
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
            prev_faces = []
            speaking_face_idx = -1
            processing_times = []
            
            # For progress tracking
            pbar = tqdm(total=total_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Process face detection at specified intervals
                if frame_count % detection_interval == 0:
                    # Detect faces and landmarks
                    faces = detect_faces_with_landmarks(
                        frame, pnet, rnet, onet, landmark_predictor
                    )
                    
                    # Compute face encodings
                    faces = compute_face_encodings_for_frame(
                        frame, faces, sess, images_placeholder, 
                        embeddings, phase_train_placeholder
                    )
                    
                    # Match faces with centers using Annoy for faster retrieval
                    for face in faces:
                        if face['encoding'] is not None:
                            # Use Annoy index to find nearest neighbor
                            nearest_indices, distances = annoy_index.get_nns_by_vector(
                                face['encoding'], 1, include_distances=True
                            )
                            
                            # Convert Annoy distance (Euclidean) to similarity
                            if len(nearest_indices) > 0:
                                match_idx = nearest_indices[0]
                                # Lower distance means higher similarity - convert to similarity score
                                similarity = 1.0 / (1.0 + distances[0])
                                
                                # Only accept if similarity is high enough
                                if similarity > 0.7:
                                    face['match_idx'] = match_idx
                                    face['similarity'] = similarity
                                else:
                                    face['match_idx'] = -1
                                    face['similarity'] = 0
                            else:
                                face['match_idx'] = -1
                                face['similarity'] = 0
                        else:
                            face['match_idx'] = -1
                            face['similarity'] = 0
                    
                    # Detect speaking face with improved algorithm
                    if prev_faces:
                        speaking_face_idx = detect_speaking_face(prev_faces, faces)
                        
                        # Update speaking face counter for stability
                        for i in range(len(faces)):
                            if i == speaking_face_idx:
                                speaking_face_counter[i] = speaking_face_counter.get(i, 0) + 1
                            else:
                                speaking_face_counter[i] = 0
                        
                        # Check if any face has been speaking for consecutive frames
                        stable_speaking_face_idx = -1
                        for idx, count in speaking_face_counter.items():
                            if count >= consecutive_frames_threshold and idx < len(faces):
                                stable_speaking_face_idx = idx
                                break
                    
                    # Update previous faces
                    prev_faces = faces
                
                # Annotate frame - only annotate stable speaking face
                if prev_faces and stable_speaking_face_idx >= 0 and stable_speaking_face_idx < len(prev_faces):
                    speaking_face = prev_faces[stable_speaking_face_idx]
                    x1, y1, x2, y2 = speaking_face['bbox']
                    match_idx = speaking_face.get('match_idx', -1)
                    similarity = speaking_face.get('similarity', 0)
                    
                    # Display identity information if matched
                    if match_idx >= 0:
                         # Draw bounding box for speaking face
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        label = f"ID: {match_idx}, Sim: {similarity:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                         # Draw bounding box for speaking face
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
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
    
    print(f"Video annotation with speaking face completed. Output saved to {output_video}")

if __name__ == "__main__":
    # Configuration
    input_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\speaking_face_video.avi"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\centers\centers_data.pkl"
    model_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759"
    
    # Run video annotation with speaking face detection
    annotate_video_with_speaking_face(
        input_video=input_video,
        output_video=output_video,
        centers_data_path=centers_data_path,
        model_dir=model_dir,
        detection_interval=2  # Process face detection every 2 frames
    )