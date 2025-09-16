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
                         frame_histories, min_face_size=60, temporal_weight=0.25):
    """
    Use the same foreground face detection logic as clustering stage
    """
    from enhanced_face_preprocessing import detect_foreground_faces_in_frame
    
    # Use unified detection parameters (consistent with clustering stage)
    min_face_area_ratio = 0.008  # Face area must occupy at least 0.8% of image
    max_faces_per_frame = 5      # Keep at most 5 faces per frame
    
    # Use unified foreground face detection function
    filtered_bboxes = detect_foreground_faces_in_frame(
        frame, pnet, rnet, onet,
        min_face_size=min_face_size,
        min_face_area_ratio=min_face_area_ratio,
        max_faces_per_frame=max_faces_per_frame
    )
    
    # If no foreground faces detected, return empty list
    if not filtered_bboxes:
        return []
    
    faces = []
    face_crops = []
    face_bboxes = []
    
    # Process filtered faces
    for bbox in filtered_bboxes:
        # Calculate adaptive margin (consistent with clustering stage)
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        margin = int(bbox_size * 0.2)  # 20% margin
        
        x1 = max(0, bbox[0] - margin)
        y1 = max(0, bbox[1] - margin)
        x2 = min(frame.shape[1], bbox[2] + margin)
        y2 = min(frame.shape[0], bbox[3] + margin)
        
        # Convert to RGB (required by FaceNet)
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        face = frame_rgb[y1:y2, x1:x2, :]
        
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue
            
        # Resize to FaceNet input size
        face_resized = cv2.resize(face, (160, 160))
        
        # FaceNet preprocessing
        face_prewhitened = facenet.prewhiten(face_resized)
        
        face_crops.append(face_prewhitened)
        face_bboxes.append((x1, y1, x2, y2))
    
    # If no valid faces, return empty list
    if not face_crops:
        return []
    
    # Batch compute face encodings
    face_batch = np.stack(face_crops)
    feed_dict = {images_placeholder: face_batch, phase_train_placeholder: False}
    face_encodings = sess.run(embeddings, feed_dict=feed_dict)
    
    # Match each face with cluster centers
    for i, (bbox, encoding) in enumerate(zip(face_bboxes, face_encodings)):
        x1, y1, x2, y2 = bbox
        face_id = f"{(x1 + x2) // 2}_{(y1 + y2) // 2}"  # Position-based ID
        
        # Get current match
        match_idx, similarity, all_similarities = match_face_with_centers(encoding, centers)
        
        # Temporal consistency processing (keep original logic)
        if face_id in frame_histories:
            history = frame_histories[face_id]
            
            if similarity > 0.4:  # Threshold for applying temporal boost
                if len(history) > 0:
                    hist_counts = {}
                    hist_sims = {}
                    
                    for hist_match, hist_sim in history:
                        if hist_match >= 0:
                            if hist_match not in hist_counts:
                                hist_counts[hist_match] = 0
                                hist_sims[hist_match] = 0
                            
                            hist_counts[hist_match] += 1
                            hist_sims[hist_match] += hist_sim
                    
                    most_freq_match = -1
                    most_freq_count = 0
                    
                    for hist_match, count in hist_counts.items():
                        if count > most_freq_count:
                            most_freq_count = count
                            most_freq_match = hist_match
                    
                    if most_freq_match >= 0 and most_freq_count >= 2:
                        hist_avg_sim = hist_sims[most_freq_match] / hist_counts[most_freq_match]
                        
                        if match_idx != most_freq_match:
                            current_sim = similarity
                            hist_match_current_sim = all_similarities[most_freq_match]
                            
                            if hist_match_current_sim > current_sim * 0.8:
                                adjusted_sim = (1 - temporal_weight) * hist_match_current_sim + temporal_weight * hist_avg_sim
                                
                                if adjusted_sim > current_sim:
                                    match_idx = most_freq_match
                                    similarity = adjusted_sim
        
        # Update history
        if face_id not in frame_histories:
            frame_histories[face_id] = deque(maxlen=10)
        
        frame_histories[face_id].append((match_idx, similarity))
        
        # Calculate face quality
        face_width = x2 - x1
        face_height = y2 - y1
        
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
                
                if frame_count % detection_interval == 0:
                    faces = detect_and_match_faces(
                        frame, pnet, rnet, onet, sess, 
                        images_placeholder, embeddings, phase_train_placeholder, 
                        centers, frame_histories, 
                        min_face_size=60,  # Consistent with clustering stage
                        temporal_weight=temporal_weight
                    )
                    
                    cached_faces = faces
                    frame_detection_results[frame_count] = faces
                    
                    # Debug output
                    if frame_count % 100 == 0 and faces:
                        print(f"Frame {frame_count}: Detected {len(faces)} foreground faces")
                        
                else:
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

    print(f"Detection results saved to {detection_results_path}")

def annotate_speaking_face_with_enhanced_detection(input_video, output_video, centers_data_path, model_dir,
                                                detection_interval=2, silence_threshold=500, audio_window=10):
    """
    Annotate video highlighting only the speaking face using audio analysis
    MODIFIED: Speaking person gets green thick border, others get light gray or hidden
    
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
        raise ValueError(f"Could not open video file: {input_video}")
    
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
            
            # Tracking variables for stable detection
            speaking_face_counter = {}  # Count consecutive frames a face is speaking
            stable_speaking_face_idx = -1
            consecutive_frames_threshold = 3  # Number of consecutive frames to confirm speaking
            
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
                
                # Update speaking face tracking
                if speaking_face_id:
                    for i, face in enumerate(faces):
                        if face['face_id'] == speaking_face_id:
                            speaking_face_counter[i] = speaking_face_counter.get(i, 0) + 1
                        else:
                            speaking_face_counter[i] = 0
                    
                    # Check if any face has been speaking for consecutive frames
                    stable_speaking_face_idx = -1
                    for idx, count in speaking_face_counter.items():
                        if count >= consecutive_frames_threshold and idx < len(faces):
                            stable_speaking_face_idx = idx
                            break
                else:
                    # No speaking detected, reset counters
                    speaking_face_counter = {}
                    stable_speaking_face_idx = -1
                
                # ============================================================================
                # MODIFIED ANNOTATION SECTION - Enhanced Speaking Face Visualization
                # ============================================================================
                
                if faces:
                    for i, face in enumerate(faces):
                        x1, y1, x2, y2 = face['bbox']
                        match_idx = face['match_idx']
                        similarity = face['similarity']
                        
                        if i == stable_speaking_face_idx:
                            # SPEAKING PERSON - Green thick border with large text
                            if match_idx >= 0:
                                # Thick green border for speaking person
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green, thick=4
                                
                                # Large text for speaking person
                                label = f"ID: {match_idx} (SPEAKING)"
                                font_scale = 0.8  # Larger font
                                thickness = 3     # Thicker text
                                
                                # Calculate text size for background
                                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                                
                                # Draw green background for text
                                cv2.rectangle(frame, 
                                             (x1, y1 - text_size[1] - 15), 
                                             (x1 + text_size[0] + 10, y1 - 5),
                                             (0, 255, 0), -1)  # Green background
                                
                                # Draw white text on green background
                                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                                
                                # Additional speaking indicator (emoji-like)
                                cv2.putText(frame, ">>", (x1 - 25, y1 + 20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            else:
                                # Unknown speaking person - still use green but with "Unknown" label
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                                
                                label = "Unknown (SPEAKING)"
                                font_scale = 0.8
                                thickness = 3
                                
                                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                                
                                # Red background for unknown speaker
                                cv2.rectangle(frame, 
                                             (x1, y1 - text_size[1] - 15), 
                                             (x1 + text_size[0] + 10, y1 - 5),
                                             (0, 0, 255), -1)  # Red background
                                
                                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                        
                        else:
                            # NON-SPEAKING FACES - Light gray, thin border (or hidden)
                            if match_idx >= 0:
                                # Very light gray border for non-speaking identified faces
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)  # Light gray, thin=1
                                
                                # Small, light text
                                label = f"ID: {match_idx}"
                                font_scale = 0.4  # Smaller font
                                thickness = 1     # Thinner text
                                
                                cv2.putText(frame, label, (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), thickness)
                            else:
                                # Unknown non-speaking faces - very light gray
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                                
                                label = "Unknown"
                                cv2.putText(frame, label, (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                            
                            # Option: Completely hide non-speaking faces (uncomment below to use)
                            # pass  # Do nothing - don't draw non-speaking faces
                
                # Add frame counter and status information
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add speaking detection status
                if stable_speaking_face_idx >= 0:
                    cv2.putText(frame, "SPEAKING DETECTED", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO SPEAKER", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
                # Show audio energy if available
                if has_audio_libraries:
                    cv2.putText(frame, f"Audio Energy: {audio_energy:.1f}", (10, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # ============================================================================
                # END OF MODIFIED ANNOTATION SECTION
                # ============================================================================
                
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

def annotate_video(input_video, output_video, centers_data_path, model_dir):
    """
    Simple video annotation function (original version)
    
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
            
            # Dictionary to store face tracking histories
            frame_histories = {}
            
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
                        centers, frame_histories
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
                    fps_rate = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"\nFrame {frame_count}/{total_frames}, Avg. processing time: {avg_time:.3f}s, FPS: {fps_rate:.2f}")
            
            pbar.close()
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video annotation completed. Output saved to {output_video}")

if __name__ == "__main__":
    # Configuration parameters for testing
    input_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\enhanced_annotated_video.avi"
    speaking_output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\enhanced_speaking_face_video.avi"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\centers\centers_data.pkl"
    model_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759"
    
    # Run enhanced video annotation
    print("Creating main annotated video...")
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
    print("\nCreating speaking face video...")
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
        print("Speaking face video created successfully!")
        
    except Exception as e:
        print(f"Error in speaking face detection: {e}")
        print("Make sure librosa and soundfile are installed for audio processing:")
        print("pip install librosa soundfile")