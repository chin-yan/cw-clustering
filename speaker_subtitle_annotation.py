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
import pysrt
import datetime
from scipy.spatial import distance
import colorsys
import warnings
import subprocess
import tempfile
from collections import defaultdict, Counter

# Suppress numpy warnings from MTCNN
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

import face_detection
import feature_extraction


def generate_distinct_colors(n):
    """
    Generate N visually distinct colors using HSV color space
    
    Args:
        n: Number of colors needed
        
    Returns:
        List of BGR color tuples
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8
        value = 0.9
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to BGR for OpenCV (0-255 range)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    return colors


class ColorManager:
    """
    Manage colors for different character IDs
    """
    
    def __init__(self, n_characters):
        """
        Initialize color manager
        
        Args:
            n_characters: Number of characters/clusters
        """
        self.colors = generate_distinct_colors(n_characters)
        self.unknown_color = (128, 128, 128)  # Gray for unknown
        self.default_color = (255, 255, 255)  # White for narration/not visible
    
    def get_color(self, character_id):
        """
        Get color for a specific character ID
        
        Args:
            character_id: Character/cluster ID (-1 for unknown)
            
        Returns:
            BGR color tuple
        """
        if character_id == -1:
            return self.unknown_color
        elif character_id >= 0 and character_id < len(self.colors):
            return self.colors[character_id]
        else:
            return self.unknown_color


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
    Detect faces and their facial landmarks in a frame with error handling
    
    Args:
        frame: Input video frame
        pnet, rnet, onet: MTCNN detector components
        landmark_predictor: Dlib facial landmark predictor
        min_face_size: Minimum face size for detection
        
    Returns:
        List of detected faces with bounding boxes and facial landmarks
    """
    import facenet.src.align.detect_face as detect_face
    
    try:
        # Convert frame to RGB (MTCNN uses RGB)
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with error handling
        try:
            bounding_boxes, _ = detect_face.detect_face(
                frame_rgb, min_face_size, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.7
            )
        except (ValueError, RuntimeWarning, ZeroDivisionError):
            return []
        
        # Check if bounding_boxes is valid
        if bounding_boxes is None or len(bounding_boxes) == 0:
            return []
        
        faces = []
        
        # Process each detected face
        for bbox in bounding_boxes:
            try:
                bbox = bbox.astype(np.int)
                
                # Validate bbox coordinates
                if len(bbox) < 4:
                    continue
                
                # Extract face area with some margin
                x1 = max(0, bbox[0] - 10)
                y1 = max(0, bbox[1] - 10)
                x2 = min(frame.shape[1], bbox[2] + 10)
                y2 = min(frame.shape[0], bbox[3] + 10)
                
                # Skip invalid bounding boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue
                
                # Convert bbox to dlib rectangle format
                rect = dlib.rectangle(x1, y1, x2, y2)
                
                # Get facial landmarks
                try:
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
                except Exception:
                    continue
                    
            except Exception:
                continue
        
        return faces
        
    except Exception:
        return []


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


def detect_speaking_face(prev_faces, curr_faces, threshold=0.5):
    """
    Detect which face is currently speaking by analyzing mouth movement
    
    Args:
        prev_faces: Faces detected in previous frame
        curr_faces: Faces detected in current frame
        threshold: Mouth movement threshold
        
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
        
        # If we found a match, calculate mouth movement
        if best_match_idx >= 0 and best_match_sim > 0.8:
            prev_mouth = prev_faces[best_match_idx]['mouth_landmarks']
            curr_mouth = curr_face['mouth_landmarks']
            
            # Calculate mouth movement as average landmark displacement
            movement = 0
            for i in range(len(prev_mouth)):
                movement += distance.euclidean(prev_mouth[i], curr_mouth[i])
            movement /= len(prev_mouth)
            
            # Update max movement
            if movement > max_movement:
                max_movement = movement
                speaking_idx = curr_idx
    
    # Return speaking face index if movement exceeds threshold
    if max_movement > threshold:
        return speaking_idx
    else:
        return -1


def parse_subtitle_file(subtitle_path):
    """
    Parse subtitle file (SRT format)
    
    Args:
        subtitle_path: Path to the subtitle file
        
    Returns:
        List of subtitle entries
    """
    if not os.path.exists(subtitle_path):
        print(f"Warning: Subtitle file not found: {subtitle_path}")
        return []
    
    try:
        subtitles = pysrt.open(subtitle_path)
        return subtitles
    except Exception as e:
        print(f"Error parsing subtitle file: {e}")
        return []


def wrap_text(text, max_width=30):
    """
    Wrap text to fit within a certain width
    
    Args:
        text: Input text
        max_width: Maximum line width
        
    Returns:
        Wrapped text with newlines
    """
    if not text:
        return ""
    
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        if current_width + len(word) + len(current_line) > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = len(word)
        else:
            current_line.append(word)
            current_width += len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def draw_subtitle_near_face(frame, text, bbox, color, position='above'):
    """
    Draw subtitle text near a face with background
    
    Args:
        frame: Video frame
        text: Subtitle text
        bbox: Face bounding box (x1, y1, x2, y2)
        color: Text and border color
        position: 'above' or 'below' the face
        
    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = bbox
    
    # Wrap text
    wrapped_text = wrap_text(text, max_width=25)
    lines = wrapped_text.split('\n')
    
    # Calculate text position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 25
    
    # Calculate total text box size
    max_text_width = 0
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
    
    total_height = line_height * len(lines)
    
    # Determine position
    if position == 'above':
        text_y = max(30, y1 - 10)
        text_x = x1
    else:
        text_y = min(frame.shape[0] - total_height - 10, y2 + 30)
        text_x = x1
    
    # Ensure text stays within frame
    text_x = max(5, min(text_x, frame.shape[1] - max_text_width - 10))
    
    # Draw background for all lines
    padding = 8
    bg_y1 = text_y - line_height - padding
    bg_y2 = text_y + (len(lines) - 1) * line_height + padding
    bg_x1 = text_x - padding
    bg_x2 = text_x + max_text_width + padding
    
    # Ensure background stays within frame
    bg_y1 = max(0, bg_y1)
    bg_y2 = min(frame.shape[0], bg_y2)
    bg_x1 = max(0, bg_x1)
    bg_x2 = min(frame.shape[1], bg_x2)
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw colored border
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
    
    # Draw text lines
    for i, line in enumerate(lines):
        line_y = text_y + i * line_height
        cv2.putText(frame, line, (text_x, line_y),
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_subtitle_at_bottom(frame, text, color=(255, 255, 255)):
    """
    Draw subtitle at bottom center of frame (traditional subtitle position)
    
    Args:
        frame: Video frame
        text: Subtitle text
        color: Text color (default white)
        
    Returns:
        Modified frame
    """
    if not text:
        return frame
    
    # Wrap text
    wrapped_text = wrap_text(text, max_width=50)
    lines = wrapped_text.split('\n')
    
    # Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_height = 35
    
    # Calculate position (centered at bottom)
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate total text box size
    max_text_width = 0
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
    
    total_height = line_height * len(lines)
    
    # Position at bottom center
    start_y = frame_height - total_height - 40
    
    # Draw background for all lines
    padding = 12
    bg_y1 = start_y - line_height + 5
    bg_y2 = start_y + (len(lines) - 1) * line_height + padding + 5
    bg_x1 = (frame_width - max_text_width) // 2 - padding
    bg_x2 = (frame_width + max_text_width) // 2 + padding
    
    # Draw semi-transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw text lines (centered)
    for i, line in enumerate(lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = (frame_width - text_width) // 2
        line_y = start_y + i * line_height
        
        # Draw text with slight shadow for better readability
        cv2.putText(frame, line, (text_x + 2, line_y + 2),
                   font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (text_x, line_y),
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def determine_speaker_for_subtitle(subtitle, detection_results, fps):
    """
    Determine the speaker for an entire subtitle segment using voting
    
    Args:
        subtitle: Subtitle entry from pysrt
        detection_results: Dictionary mapping frame_idx to detected faces
        fps: Video frame rate
        
    Returns:
        (speaker_character_id, speaker_bbox) or (None, None) if no speaker found
    """
    # Calculate frame range for this subtitle
    start_ms = subtitle.start.ordinal
    end_ms = subtitle.end.ordinal
    
    start_frame = int((start_ms / 1000.0) * fps)
    end_frame = int((end_ms / 1000.0) * fps)
    
    # Collect all speaking face detections in this time range
    speaker_votes = Counter()
    speaker_bboxes = {}
    
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx in detection_results:
            result = detection_results[frame_idx]
            speaking_idx = result.get('speaking_idx', -1)
            faces = result.get('faces', [])
            
            if speaking_idx >= 0 and speaking_idx < len(faces):
                speaking_face = faces[speaking_idx]
                char_id = speaking_face.get('match_idx', -1)
                
                if char_id >= 0:
                    speaker_votes[char_id] += 1
                    speaker_bboxes[char_id] = speaking_face['bbox']
    
    # Use most common speaker
    if speaker_votes:
        most_common_speaker = speaker_votes.most_common(1)[0][0]
        return most_common_speaker, speaker_bboxes.get(most_common_speaker)
    
    return None, None


def merge_audio_to_video(video_without_audio, original_video, output_video):
    """
    Merge audio from original video to annotated video using FFmpeg
    
    Args:
        video_without_audio: Path to video without audio
        original_video: Path to original video with audio
        output_video: Path to output video with audio
        
    Returns:
        True if successful, False otherwise
    """
    print("\nMerging audio from original video...")
    
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_without_audio,
            '-i', original_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"Audio merged successfully: {output_video}")
            return True
        else:
            print(f"FFmpeg failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FFmpeg operation timed out")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg to preserve audio.")
        return False
    except Exception as e:
        print(f"Error merging audio: {e}")
        return False


def annotate_video_with_speaker_subtitles(input_video, output_video, centers_data_path, 
                                         subtitle_path, model_dir, detection_interval=2,
                                         similarity_threshold=0.65, preserve_audio=True):
    """
    Annotate video with color-coded speaker subtitles with continuous display
    
    Args:
        input_video: Input video path
        output_video: Output video path
        centers_data_path: Path to cluster center data
        subtitle_path: Path to subtitle file (SRT format)
        model_dir: FaceNet model directory
        detection_interval: Interval for face detection (process every N frames)
        similarity_threshold: Threshold for face matching
        preserve_audio: Whether to preserve original audio
    """
    # Load centers data
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    # Initialize color manager
    color_manager = ColorManager(len(centers))
    print(f"Initialized color manager with {len(centers)} distinct colors")
    
    # Parse subtitle file
    subtitles = parse_subtitle_file(subtitle_path)
    print(f"Loaded {len(subtitles)} subtitles")
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Create temporary video without audio
    if preserve_audio:
        temp_video = output_video.replace('.avi', '_temp_no_audio.avi')
        final_output = output_video
    else:
        temp_video = output_video
        final_output = None
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
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
            
            # PHASE 1: Detect faces and speakers for all frames
            print("\n" + "="*70)
            print("PHASE 1: Detecting faces and identifying speakers")
            print("="*70)
            
            detection_results = {}
            prev_faces = []
            
            pbar = tqdm(total=total_frames, desc="Detecting")
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                    
                    # Match faces with centers
                    for face in faces:
                        if face['encoding'] is not None:
                            match_idx, similarity = match_face_with_centers(
                                face['encoding'], centers, threshold=similarity_threshold
                            )
                            face['match_idx'] = match_idx
                            face['similarity'] = similarity
                        else:
                            face['match_idx'] = -1
                            face['similarity'] = 0
                    
                    # Detect speaking face
                    speaking_idx = -1
                    if prev_faces:
                        speaking_idx = detect_speaking_face(prev_faces, faces)
                    
                    # Store detection result
                    detection_results[frame_count] = {
                        'faces': faces,
                        'speaking_idx': speaking_idx
                    }
                    
                    prev_faces = faces
                
                frame_count += 1
                pbar.update(1)
            
            pbar.close()
            cap.release()
            
            print(f"Detected faces in {len(detection_results)} frames")
            
            # PHASE 2: Determine speaker for each subtitle segment
            print("\n" + "="*70)
            print("PHASE 2: Determining speakers for subtitle segments")
            print("="*70)
            
            subtitle_speakers = {}
            
            for subtitle in tqdm(subtitles, desc="Processing subtitles"):
                speaker_id, speaker_bbox = determine_speaker_for_subtitle(
                    subtitle, detection_results, fps
                )
                
                subtitle_speakers[subtitle.index] = {
                    'speaker_id': speaker_id,
                    'speaker_bbox': speaker_bbox,
                    'text': subtitle.text_without_tags,
                    'start_ms': subtitle.start.ordinal,
                    'end_ms': subtitle.end.ordinal
                }
            
            print(f"Processed {len(subtitle_speakers)} subtitle segments")
            
            # PHASE 3: Render video with continuous subtitles
            print("\n" + "="*70)
            print("PHASE 3: Rendering final video with subtitles")
            print("="*70)
            
            # Reopen video for rendering
            cap = cv2.VideoCapture(input_video)
            frame_count = 0
            
            pbar = tqdm(total=total_frames, desc="Rendering")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp_ms = int((frame_count / fps) * 1000)
                
                # Find current subtitle
                current_subtitle = None
                current_subtitle_info = None
                
                for sub_idx, sub_info in subtitle_speakers.items():
                    if sub_info['start_ms'] <= timestamp_ms <= sub_info['end_ms']:
                        current_subtitle = sub_info['text']
                        current_subtitle_info = sub_info
                        break
                
                # Get detected faces for this frame (if available)
                faces_in_frame = []
                if frame_count in detection_results:
                    faces_in_frame = detection_results[frame_count]['faces']
                
                # Annotate frame
                if current_subtitle and current_subtitle_info:
                    speaker_id = current_subtitle_info['speaker_id']
                    speaker_bbox = current_subtitle_info['speaker_bbox']
                    
                    # If speaker is identified
                    if speaker_id is not None and speaker_id >= 0:
                        color = color_manager.get_color(speaker_id)
                        
                        # Find speaker's current bbox in frame (if detected)
                        current_speaker_bbox = None
                        for face in faces_in_frame:
                            if face.get('match_idx') == speaker_id:
                                current_speaker_bbox = face['bbox']
                                break
                        
                        # Use current bbox if available, otherwise use stored bbox
                        display_bbox = current_speaker_bbox if current_speaker_bbox else speaker_bbox
                        
                        if display_bbox:
                            # Draw speaker's bounding box
                            x1, y1, x2, y2 = display_bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            
                            # Draw ID label
                            id_label = f"Speaker ID: {speaker_id}"
                            label_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                        0.6, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0] + 10, y1), color, -1)
                            cv2.putText(frame, id_label, (x1 + 5, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Draw subtitle near speaker
                            frame = draw_subtitle_near_face(frame, current_subtitle, 
                                                          display_bbox, color, position='above')
                        else:
                            # Speaker bbox not available, show at bottom
                            frame = draw_subtitle_at_bottom(frame, current_subtitle, color)
                        
                        # Draw other faces with their colors
                        for face in faces_in_frame:
                            face_id = face.get('match_idx', -1)
                            if face_id >= 0 and face_id != speaker_id:
                                face_color = color_manager.get_color(face_id)
                                fx1, fy1, fx2, fy2 = face['bbox']
                                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 1)
                    else:
                        # No identified speaker - show subtitle at bottom
                        frame = draw_subtitle_at_bottom(frame, current_subtitle, 
                                                       color=(255, 255, 255))
                        
                        # Still draw detected faces
                        for face in faces_in_frame:
                            face_id = face.get('match_idx', -1)
                            if face_id >= 0:
                                face_color = color_manager.get_color(face_id)
                                fx1, fy1, fx2, fy2 = face['bbox']
                                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 1)
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame
                out.write(frame)
                
                frame_count += 1
                pbar.update(1)
            
            pbar.close()
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\nVideo rendering complete: {temp_video}")
    
    # Merge audio if requested
    if preserve_audio and final_output:
        print("\n" + "="*70)
        print("PHASE 4: Merging audio from original video")
        print("="*70)
        
        success = merge_audio_to_video(temp_video, input_video, final_output)
        
        if success:
            # Clean up temporary file
            try:
                os.remove(temp_video)
                print(f"Cleaned up temporary file: {temp_video}")
            except:
                pass
            
            print(f"\n{'='*70}")
            print("FINAL OUTPUT WITH AUDIO:")
            print(f"  {final_output}")
            print("="*70)
        else:
            print(f"\nWarning: Audio merging failed. Video without audio saved at:")
            print(f"  {temp_video}")
    else:
        print(f"\n{'='*70}")
        print("FINAL OUTPUT (NO AUDIO):")
        print(f"  {temp_video}")
        print("="*70)


if __name__ == "__main__":
    # Configuration
    input_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\S02\ep1.mp4"
    output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_s2ep1\color_coded_subtitles.mp4"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_s2ep1\centers\centers_data.pkl"
    subtitle_path = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\S02\subtitles\s2ep1.srt"
    model_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759"
    
    # Run video annotation with color-coded speaker subtitles
    annotate_video_with_speaker_subtitles(
        input_video=input_video,
        output_video=output_video,
        centers_data_path=centers_data_path,
        subtitle_path=subtitle_path,
        model_dir=model_dir,
        detection_interval=2,
        similarity_threshold=0.65,
        preserve_audio=True  # Set to False if you don't have FFmpeg or don't need audio
    )