# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
import time
from collections import deque
import math
from facenet.src import facenet

import face_detection
import feature_extraction
import enhanced_face_preprocessing
import math

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

def match_face_with_centers(face_encoding, centers, threshold=0.65):
    """
    Match a face encoding with the cluster centers using cosine similarity
    
    Args:
        face_encoding: Face encoding vector
        centers: List of cluster center encodings
        threshold: Similarity threshold
        
    Returns:
        Index of the matched center, similarity score, and all similarity scores
    """
    if len(centers) == 0:
        return -1, 0, []
    
    # Calculate cosine similarity (dot product of normalized vectors) with all centers
    similarities = np.dot(centers, face_encoding)
    
    # Find the most similar center
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]
    
    # Return match if similarity exceeds threshold
    if best_similarity > threshold:
        return best_index, best_similarity, similarities
    else:
        return -1, 0, similarities

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
                         embeddings, phase_train_placeholder, centers, 
                         frame_histories, min_face_size=20, temporal_weight=0.3):
    """
    Detect faces in a frame and match them with cluster centers using temporal consistency
    
    Args:
        frame: Input video frame
        pnet, rnet, onet: MTCNN detector components
        sess: TensorFlow session
        images_placeholder: Input placeholder for FaceNet
        embeddings: Output embeddings tensor
        phase_train_placeholder: Phase train placeholder
        centers: Cluster center encodings
        frame_histories: Face tracking histories
        min_face_size: Minimum face size for detection
        temporal_weight: Weight for temporal consistency
        
    Returns:
        List of detected faces with bounding boxes and matched center indices
    """
    import facenet.src.align.detect_face as detect_face
    
    # Convert frame to RGB (MTCNN uses RGB)
    if frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Detect faces with two passes (first for frontal, then for side faces)
    # First pass with standard parameters
    bounding_boxes, _ = detect_face.detect_face(
        frame_rgb, min_face_size, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.7
    )
    
    # Second pass with lower thresholds for side faces if no faces found in first pass
    if len(bounding_boxes) == 0:
        bounding_boxes, _ = detect_face.detect_face(
            frame_rgb, min_face_size, pnet, rnet, onet, [0.5, 0.6, 0.6], 0.6
        )
    
    faces = []
    
    # Preprocess faces and compute embeddings in batch for efficiency
    face_crops = []
    face_bboxes = []
    
    # Process each detected face
    for bbox in bounding_boxes:
        bbox = bbox.astype(np.int)
        
        # Calculate margin adaptively based on face size
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        margin = int(bbox_size * 0.2)  # 20% margin
        
        # Extract face area with margin
        x1 = max(0, bbox[0] - margin)
        y1 = max(0, bbox[1] - margin)
        x2 = min(frame.shape[1], bbox[2] + margin)
        y2 = min(frame.shape[0], bbox[3] + margin)
        
        face = frame_rgb[y1:y2, x1:x2, :]
        
        # Skip invalid faces
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue
            
        # Resize to FaceNet input size
        face_resized = cv2.resize(face, (160, 160))
        
        # Preprocess for FaceNet
        face_prewhitened = facenet.prewhiten(face_resized)
        
        # Add to batch
        face_crops.append(face_prewhitened)
        face_bboxes.append((x1, y1, x2, y2))
    
    # If no valid faces, return empty list
    if not face_crops:
        return []
    
    # Convert to batch
    face_batch = np.stack(face_crops)
    
    # Get face encodings in batch
    feed_dict = {images_placeholder: face_batch, phase_train_placeholder: False}
    face_encodings = sess.run(embeddings, feed_dict=feed_dict)
    
    # Match each face with centers
    for i, (bbox, encoding) in enumerate(zip(face_bboxes, face_encodings)):
        # Generate a face ID based on position (simple tracking)
        x1, y1, x2, y2 = bbox
        face_id = f"{(x1 + x2) // 2}_{(y1 + y2) // 2}"  # Center position as ID
        
        # Get current match
        match_idx, similarity, all_similarities = match_face_with_centers(encoding, centers)
        
        # Apply temporal consistency if this face has history
        if face_id in frame_histories:
            history = frame_histories[face_id]
            
            # Apply temporal consistency only if current match is somewhat similar
            if similarity > 0.4:  # Lower threshold for applying temporal boost
                # Get history match
                if len(history) > 0:
                    # Check if there's a consistent match in history
                    hist_counts = {}
                    hist_sims = {}
                    
                    for hist_match, hist_sim in history:
                        if hist_match >= 0:
                            if hist_match not in hist_counts:
                                hist_counts[hist_match] = 0
                                hist_sims[hist_match] = 0
                            
                            hist_counts[hist_match] += 1
                            hist_sims[hist_match] += hist_sim
                    
                    # Find most frequent match in history
                    most_freq_match = -1
                    most_freq_count = 0
                    
                    for hist_match, count in hist_counts.items():
                        if count > most_freq_count:
                            most_freq_count = count
                            most_freq_match = hist_match
                    
                    # If there's a consistent match and it's in top matches
                    if most_freq_match >= 0 and most_freq_count >= 2:
                        hist_avg_sim = hist_sims[most_freq_match] / hist_counts[most_freq_match]
                        
                        # If current match is different than history match
                        if match_idx != most_freq_match:
                            # Check if history match is close in similarity to current best match
                            current_sim = similarity
                            hist_match_current_sim = all_similarities[most_freq_match]
                            
                            # If history match is close enough to current match
                            if hist_match_current_sim > current_sim * 0.8:
                                # Apply temporal consistency - weighted combination
                                adjusted_sim = (1 - temporal_weight) * hist_match_current_sim + temporal_weight * hist_avg_sim
                                
                                # If adjusted similarity is better than current match
                                if adjusted_sim > current_sim:
                                    match_idx = most_freq_match
                                    similarity = adjusted_sim
        
        # Update history for this face
        if face_id not in frame_histories:
            frame_histories[face_id] = deque(maxlen=10)  # Keep history of last 10 frames
        
        frame_histories[face_id].append((match_idx, similarity))
        
        # Store face info
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Calculate face quality (used for visualization)
        face_quality = 0.5
        try:
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            face_quality = min(1.0, np.var(laplacian) / 500)
        except:
            pass
        
        faces.append({
            'bbox': (x1, y1, x2, y2),
            'match_idx': match_idx,
            'similarity': similarity,
            'face_id': face_id,
            'size': face_width * face_height,
            'quality': face_quality
        })
    
    return faces

def annotate_video_with_enhanced_detection(input_video, output_video, centers_data_path, model_dir,
                                         detection_interval=1, similarity_threshold=0.55, 
                                         temporal_weight=0.3):
    """
    Annotate video with face identities using enhanced detection and temporal consistency
    
    Args:
        input_video: Input video path
        output_video: Output video path
        centers_data_path: Path to cluster center data
        model_dir: FaceNet model directory
        detection_interval: Process every N frames for detection
        similarity_threshold: Minimum similarity threshold for matching
        temporal_weight: Weight for temporal consistency
    """
    # Store the detection results of each frame
    frame_detection_results = {}

    # Add after processing each frame
    frame_detection_results[frame_count] = faces
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
    
    # Generate colors for each cluster
    import colorsys
    n_centers = len(centers)
    colors = []
    for i in range(n_centers):
        # Generate distinct colors using HSV
        h = i / n_centers
        s = 0.8
        v = 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        # Convert to BGR (OpenCV format) with range 0-255
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
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
            
            # Dictionary to store face tracking histories
            frame_histories = {}
            
            # List to store face tracking data for visualization
            tracking_data = []
            
            # Process video frames
            print(f"Processing video with {total_frames} frames...")
            frame_count = 0
            processing_times = []
            
            # Cache detected faces for non-detection frames
            cached_faces = []
            
            # For progress tracking
            pbar = tqdm(total=total_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Process every N frames for detection
                if frame_count % detection_interval == 0:
                    # Detect and match faces
                    faces = detect_and_match_faces(
                        frame, pnet, rnet, onet, sess, 
                        images_placeholder, embeddings, phase_train_placeholder, 
                        centers, frame_histories, 
                        min_face_size=20, temporal_weight=temporal_weight
                    )
                    
                    # Update cache
                    cached_faces = faces
                else:
                    # Use cached faces with adjusted bounding boxes (simple tracking)
                    # In real applications, a proper tracking algorithm would be used
                    faces = cached_faces
                
                # Annotate frame
                for face in faces:
                    x1, y1, x2, y2 = face['bbox']
                    match_idx = face['match_idx']
                    similarity = face['similarity']
                    
                    # Draw bounding box
                    if match_idx >= 0:
                        # Matched face - use cluster-specific color
                        color = colors[match_idx]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Display identity information
                        label = f"ID: {match_idx}, Sim: {similarity:.2f}"
                        
                        # Background for text
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, 
                                     (x1, y1 - text_size[1] - 10), 
                                     (x1 + text_size[0], y1),
                                     color, -1)  # Filled rectangle
                        
                        # Text
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Store tracking data
                        if frame_count % 5 == 0:  # Record every 5 frames to reduce data size
                            tracking_data.append({
                                'frame': frame_count,
                                'face_id': face['face_id'],
                                'match_idx': match_idx,
                                'similarity': similarity,
                                'position': ((x1 + x2) // 2, (y1 + y2) // 2)
                            })
                    else:
                        # Unmatched face - red box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Display unknown label
                        cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame to output
                out.write(frame)
                
                end_time = time.time()
                processing_times.append(end_time - start_time)
                
                # Update progress
                frame_count += 1
                pbar.update(1)
                
                # Print processing stats every 100 frames
                if frame_count % 100 == 0:
                    avg_time = np.mean(processing_times[-100:])
                    fps_rate = 1.0 / (avg_time + 1e-6)
                    print(f"\nFrame {frame_count}/{total_frames}, Avg. processing time: {avg_time:.3f}s, FPS: {fps_rate:.2f}")
            
            pbar.close()
    
    # Release resources
    cap.release()
    out.release()
    
    # Save tracking data for analysis
    tracking_data_path = os.path.join(os.path.dirname(output_video), 'tracking_data.pkl')
    with open(tracking_data_path, 'wb') as f:
        pickle.dump(tracking_data, f)
    
    print(f"Video annotation completed. Output saved to {output_video}")
    print(f"Tracking data saved to {tracking_data_path}")
    
    # Save the test results for later processing
    detection_results_path = os.path.join(os.path.dirname(output_video), 'enhanced_detection_results.pkl')
    with open(detection_results_path, 'wb') as f:
        pickle.dump(frame_detection_results, f)

    print(f"The detection results have been saved to {detection_results_path}")

def annotate_speaking_face_with_enhanced_detection(input_video, output_video, centers_data_path, model_dir,
                                                detection_interval=2, silence_threshold=500, audio_window=10):
    """
    Annotate video highlighting only the speaking face using audio analysis
    
    Args:
        input_video: Input video path
        output_video: Output video path
        centers_data_path: Path to cluster center data
        model_dir: FaceNet model directory
        detection_interval: Process every N frames for detection
        silence_threshold: Audio level threshold to detect speech
        audio_window: Window size for audio analysis in frames
    """
    try:
        import librosa
        import soundfile as sf
        has_audio_libraries = True
    except ImportError:
        print("Warning: librosa or soundfile not found. Will run without audio analysis.")
        has_audio_libraries = False
    
    # Load centers data
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    # Create temporary directory for audio extraction
    import tempfile
    import subprocess
    from pathlib import Path
    
    temp_dir = tempfile.mkdtemp()
    temp_audio = os.path.join(temp_dir, "audio.wav")
    
    # Extract audio if libraries available
    audio_data = None
    audio_sr = None
    
    if has_audio_libraries:
        try:
            # Extract audio using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg", "-i", input_video, "-vn", "-acodec", "pcm_s16le", 
                "-ar", "16000", "-ac", "1", temp_audio, "-y"
            ]
            
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Load audio file
            audio_data, audio_sr = librosa.load(temp_audio, sr=None)
            print(f"Audio loaded: {len(audio_data)/audio_sr:.2f} seconds at {audio_sr}Hz")
        except Exception as e:
            print(f"Error extracting audio: {e}")
            has_audio_libraries = False
    
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
    
    # Calculate audio frames per video frame
    audio_frames_per_video_frame = None
    if has_audio_libraries and audio_data is not None:
        audio_frames_per_video_frame = audio_sr / fps
    
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
            
            # Dictionary to store face tracking histories
            frame_histories = {}
            
            # Process video frames
            print(f"Processing video with {total_frames} frames...")
            frame_count = 0
            
            # Cache detected faces for non-detection frames
            cached_faces = []
            
            # Audio energy history for each face ID
            audio_energy_history = {}
            
            # For progress tracking
            pbar = tqdm(total=total_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every N frames for detection
                if frame_count % detection_interval == 0:
                    # Detect and match faces
                    faces = detect_and_match_faces(
                        frame, pnet, rnet, onet, sess, 
                        images_placeholder, embeddings, phase_train_placeholder, 
                        centers, frame_histories
                    )
                    
                    # Update cache
                    cached_faces = faces
                else:
                    # Use cached faces
                    faces = cached_faces
                
                # Determine if there's speech in this frame using audio analysis
                is_speaking = False
                audio_energy = 0
                
                if has_audio_libraries and audio_data is not None and audio_frames_per_video_frame is not None:
                    # Calculate corresponding audio segment
                    start_idx = int(frame_count * audio_frames_per_video_frame)
                    end_idx = int(start_idx + audio_frames_per_video_frame * audio_window)
                    
                    if start_idx < len(audio_data) and end_idx <= len(audio_data):
                        # Get audio segment and compute energy
                        audio_segment = audio_data[start_idx:end_idx]
                        audio_energy = np.mean(np.abs(audio_segment)) * 10000
                        
                        # Determine if speech is occurring
                        is_speaking = audio_energy > silence_threshold
                
                # Identify the speaking face
                speaking_face_id = None
                
                if is_speaking and faces:
                    # Update audio energy for each face
                    for face in faces:
                        face_id = face['face_id']
                        if face_id not in audio_energy_history:
                            audio_energy_history[face_id] = deque(maxlen=10)
                        
                        # Add current energy with decay factor for older entries
                        if len(audio_energy_history[face_id]) > 0:
                            prev_energy = audio_energy_history[face_id][-1]
                            # Smooth energy transition
                            smoothed_energy = 0.7 * audio_energy + 0.3 * prev_energy
                            audio_energy_history[face_id].append(smoothed_energy)
                        else:
                            audio_energy_history[face_id].append(audio_energy)
                    
                    # Find face with highest average energy
                    max_energy = 0
                    for face in faces:
                        face_id = face['face_id']
                        if face_id in audio_energy_history and len(audio_energy_history[face_id]) > 0:
                            avg_energy = sum(audio_energy_history[face_id]) / len(audio_energy_history[face_id])
                            if avg_energy > max_energy:
                                max_energy = avg_energy
                                speaking_face_id = face_id
                
                # Annotate frame
                for face in faces:
                    x1, y1, x2, y2 = face['bbox']
                    match_idx = face['match_idx']
                    similarity = face['similarity']
                    face_id = face['face_id']
                    
                    # Determine if this is the speaking face
                    is_this_speaking = (face_id == speaking_face_id)
                    
                    if match_idx >= 0:
                        # Choose color based on speaking status
                        if is_this_speaking:
                            # Speaking face - green box
                            color = (0, 255, 0)
                            line_thickness = 3
                        else:
                            # Non-speaking face - yellow box
                            color = (0, 165, 255)
                            line_thickness = 1
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
                        
                        # Display identity and speaking status
                        if is_this_speaking:
                            label = f"ID: {match_idx} (Speaking)"
                        else:
                            label = f"ID: {match_idx}"
                        
                        # Background for text
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, 
                                     (x1, y1 - text_size[1] - 10), 
                                     (x1 + text_size[0], y1),
                                     color, -1)  # Filled rectangle
                        
                        # Text
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        # Unmatched face - red box with thinner line
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                # Add frame counter and audio energy
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if has_audio_libraries:
                    cv2.putText(frame, f"Audio Energy: {audio_energy:.1f}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame to output
                out.write(frame)
                
                # Update progress
                frame_count += 1
                pbar.update(1)
                
                # Print processing stats occasionally
                if frame_count % 100 == 0:
                    print(f"\nProcessed {frame_count}/{total_frames} frames")
            
            pbar.close()
    
    # Clean up
    cap.release()
    out.release()
    
    # Remove temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Video annotation with speaking face detection completed. Output saved to {output_video}")

if __name__ == "__main__":
    # Configure parameters
    input_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\enhanced_annotated_video.avi"
    speaking_output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\enhanced_speaking_face_video.avi"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\centers\centers_data.pkl"
    model_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759"
    
    # Run enhanced video annotation
    annotate_video_with_enhanced_detection(
        input_video=input_video,
        output_video=output_video,
        centers_data_path=centers_data_path,
        model_dir=model_dir,
        detection_interval=2,  # Process every 2 frames
        similarity_threshold=0.55,  # Lower threshold for matching
        temporal_weight=0.3  # Weight for temporal consistency
    )
    
    # Run speaking face detection (requires librosa and soundfile)
    try:
        annotate_speaking_face_with_enhanced_detection(
            input_video=input_video,
            output_video=speaking_output_video,
            centers_data_path=centers_data_path,
            model_dir=model_dir,
            detection_interval=2,
            silence_threshold=500,
            audio_window=10
        )
    except Exception as e:
        print(f"Error in speaking face detection: {e}")
        print("Make sure librosa and soundfile are installed for audio processing")
        print("pip install librosa soundfile")