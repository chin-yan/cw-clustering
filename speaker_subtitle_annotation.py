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
import json
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


class BBoxSmoother:
    """
    Smooth bounding box positions using Exponential Moving Average (EMA)
    to reduce jitter and flickering in subtitle positions
    """
    
    def __init__(self, initial_bbox, alpha=0.3):
        """
        Initialize bbox smoother with initial position
        
        Args:
            initial_bbox: Initial bounding box (x1, y1, x2, y2)
            alpha: Smoothing factor (0-1). Lower values = more smoothing
                   0.1 = very smooth, slow to adapt
                   0.3 = balanced (recommended)
                   0.5 = less smooth, faster adaptation
                   1.0 = no smoothing (raw bbox)
        """
        if initial_bbox is None:
            self.smoothed_bbox = None
        else:
            self.smoothed_bbox = [float(x) for x in initial_bbox]
        self.alpha = alpha
        self.initialized = (initial_bbox is not None)
    
    def update(self, new_bbox):
        """
        Update and return smoothed bounding box
        
        Args:
            new_bbox: New detected bbox (x1, y1, x2, y2) or None
            
        Returns:
            Smoothed bbox as tuple of integers (x1, y1, x2, y2)
            Returns None if smoother is not initialized
        """
        if new_bbox is None:
            # No new detection, return current smoothed position
            if self.smoothed_bbox is None:
                return None
            return tuple(int(round(x)) for x in self.smoothed_bbox)
        
        if not self.initialized:
            # First bbox, initialize directly
            self.smoothed_bbox = [float(x) for x in new_bbox]
            self.initialized = True
            return tuple(int(round(x)) for x in self.smoothed_bbox)
        
        # Apply Exponential Moving Average: smoothed = alpha * new + (1 - alpha) * old
        for i in range(4):
            self.smoothed_bbox[i] = (
                self.alpha * new_bbox[i] + 
                (1 - self.alpha) * self.smoothed_bbox[i]
            )
        
        return tuple(int(round(x)) for x in self.smoothed_bbox)
    
    def get_current(self):
        """
        Get current smoothed bbox without updating
        
        Returns:
            Current smoothed bbox or None
        """
        if self.smoothed_bbox is None:
            return None
        return tuple(int(round(x)) for x in self.smoothed_bbox)

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


def validate_face_quality(face, frame):
    """
    Validate if the detected face is actually a face with good quality
    
    Args:
        face: Face dictionary with bbox and landmarks
        frame: Video frame
        
    Returns:
        True if face passes quality checks, False otherwise
    """
    x1, y1, x2, y2 = face['bbox']
    
    # Check 1: Minimum size requirement (avoid tiny detections)
    face_width = x2 - x1
    face_height = y2 - y1
    min_size = 40
    
    if face_width < min_size or face_height < min_size:
        return False
    
    # Check 2: Aspect ratio check (faces should be roughly square-ish)
    aspect_ratio = face_width / face_height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
    
    # Check 3: Landmarks validation
    if 'landmarks' not in face or not face['landmarks']:
        return False
    
    landmarks = face['landmarks']
    
    # Check if landmarks are within the bounding box
    landmarks_inside = 0
    for lx, ly in landmarks:
        if x1 <= lx <= x2 and y1 <= ly <= y2:
            landmarks_inside += 1
    
    # At least 80% of landmarks should be inside the bbox
    if landmarks_inside < len(landmarks) * 0.8:
        return False
    
    # Check 4: Face region should have reasonable variance (not blank/uniform)
    try:
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return False
        
        # Convert to grayscale for variance calculation
        if len(face_region.shape) == 3:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_region
        
        variance = np.var(face_gray)
        
        # If variance is too low, it's likely a uniform region (not a face)
        if variance < 100:
            return False
    except:
        return False
    
    # Check 5: Verify mouth landmarks form a reasonable mouth shape
    if 'mouth_landmarks' in face and face['mouth_landmarks']:
        mouth_landmarks = face['mouth_landmarks']
        
        # Calculate mouth width and height
        mouth_points = np.array(mouth_landmarks)
        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
        mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
        
        # Mouth should have reasonable size relative to face
        if mouth_width < face_width * 0.2 or mouth_width > face_width * 0.8:
            return False
        
        if mouth_height < 5 or mouth_height > face_height * 0.5:
            return False
    
    return True


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
                    face_data = {
                        'bbox': (x1, y1, x2, y2),
                        'landmarks': landmarks,
                        'mouth_landmarks': mouth_landmarks,
                        'rect': rect
                    }
                    
                    # Validate face quality before adding
                    if validate_face_quality(face_data, frame):
                        faces.append(face_data)
                    
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


def detect_speaking_face(prev_faces, curr_faces, threshold=0.8):
    """
    Detect which face is currently speaking by analyzing mouth movement
    
    Args:
        prev_faces: Faces detected in previous frame
        curr_faces: Faces detected in current frame
        threshold: Mouth movement threshold (increased for stricter detection)
        
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
            
            # Additional check: Calculate mouth opening (vertical movement)
            if len(prev_mouth) >= 12 and len(curr_mouth) >= 12:
                # Top lip center (index 3 in outer lip landmarks)
                prev_top = prev_mouth[3][1]
                curr_top = curr_mouth[3][1]
                # Bottom lip center (index 9 in outer lip landmarks)
                prev_bottom = prev_mouth[9][1]
                curr_bottom = curr_mouth[9][1]
                
                # Calculate mouth opening change
                prev_opening = abs(prev_bottom - prev_top)
                curr_opening = abs(curr_bottom - curr_top)
                opening_change = abs(curr_opening - prev_opening)
                
                # Weight movement by opening change
                weighted_movement = movement * (1 + opening_change / 10.0)
            else:
                weighted_movement = movement
            
            # Update max movement
            if weighted_movement > max_movement:
                max_movement = weighted_movement
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


def draw_subtitle_beside_label(frame, text, bbox, color, label_width=120):
    """
    Draw subtitle text beside the character label with stable positioning
    
    Args:
        frame: Video frame
        text: Subtitle text
        bbox: Face bounding box (x1, y1, x2, y2)
        color: Text and border color
        label_width: Width of the character label (to position subtitle beside it)
        
    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame.shape[:2]
    
    # Wrap text with reasonable width
    wrapped_text = wrap_text(text, max_width=30)
    lines = wrapped_text.split('\n')
    
    # Text styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_height = 28
    
    # Calculate text box size
    max_text_width = 0
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
    
    total_height = line_height * len(lines)
    padding = 10
    
    # Position subtitle to the RIGHT of the character label
    spacing = 20
    subtitle_x = x1 + label_width + spacing
    subtitle_y = y1 - line_height
    
    # Ensure subtitle stays within frame bounds
    if subtitle_x + max_text_width + padding * 2 > frame_width:
        # If goes off right edge, position to the LEFT of the bbox instead
        subtitle_x = max(padding, x1 - max_text_width - spacing - padding * 2)
    
    # Ensure subtitle doesn't go off top
    subtitle_y = max(line_height + padding, subtitle_y)
    
    # Background box coordinates
    bg_x1 = subtitle_x - padding
    bg_y1 = subtitle_y - line_height - padding
    bg_x2 = subtitle_x + max_text_width + padding
    bg_y2 = subtitle_y + (len(lines) - 1) * line_height + padding
    
    # Clamp to frame boundaries
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(frame_width, bg_x2)
    bg_y2 = min(frame_height, bg_y2)
    
    # Draw semi-transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw colored border
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
    
    # Draw text lines
    for i, line in enumerate(lines):
        line_y = subtitle_y + i * line_height
        # Draw text with slight shadow for better readability
        cv2.putText(frame, line, (subtitle_x + 1, line_y + 1),
                   font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (subtitle_x, line_y),
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
        (speaker_character_id, speaker_bbox, all_speaker_ids, all_faces_info) tuple
    """
    # Calculate frame range for this subtitle
    start_ms = subtitle.start.ordinal
    end_ms = subtitle.end.ordinal
    
    start_frame = int((start_ms / 1000.0) * fps)
    end_frame = int((end_ms / 1000.0) * fps)
    
    # Collect all speaking face detections in this time range
    speaker_votes = Counter()
    speaker_bboxes = {}
    all_speaker_ids = []
    all_faces_data = []
    
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx in detection_results:
            result = detection_results[frame_idx]
            speaking_idx = result.get('speaking_idx', -1)
            faces = result.get('faces', [])
            
            # Collect all face IDs in this frame
            frame_face_ids = []
            for face in faces:
                face_id = face.get('match_idx', -1)
                if face_id >= 0:
                    frame_face_ids.append({
                        'face_id': face_id,
                        'bbox': face['bbox'],
                        'similarity': face.get('similarity', 0)
                    })
            
            if frame_face_ids:
                all_faces_data.append({
                    'frame_idx': frame_idx,
                    'faces': frame_face_ids
                })
            
            if speaking_idx >= 0 and speaking_idx < len(faces):
                speaking_face = faces[speaking_idx]
                char_id = speaking_face.get('match_idx', -1)
                
                if char_id >= 0:
                    speaker_votes[char_id] += 1
                    speaker_bboxes[char_id] = speaking_face['bbox']
                    all_speaker_ids.append(char_id)
    
    # Use most common speaker
    if speaker_votes:
        most_common_speaker = speaker_votes.most_common(1)[0][0]
        # Get unique speaker IDs
        unique_speaker_ids = list(set(all_speaker_ids))
        return most_common_speaker, speaker_bboxes.get(most_common_speaker), unique_speaker_ids, all_faces_data
    
    return None, None, [], all_faces_data


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
    print(f"  Input video (no audio): {video_without_audio}")
    print(f"  Original video (with audio): {original_video}")
    print(f"  Output video: {output_video}")
    
    try:
        # Always use a temporary file for FFmpeg output to avoid conflicts
        output_dir = os.path.dirname(output_video)
        temp_ffmpeg_output = os.path.join(output_dir, 'temp_ffmpeg_output.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_without_audio,
            '-i', original_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            temp_ffmpeg_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Move temporary FFmpeg output to final destination
            try:
                if os.path.exists(output_video):
                    os.remove(output_video)
                os.rename(temp_ffmpeg_output, output_video)
                print(f"Audio merged successfully!")
                return True
            except Exception as e:
                print(f"Error moving file: {e}")
                print(f"Temporary file saved at: {temp_ffmpeg_output}")
                return False
        else:
            print(f"FFmpeg failed: {result.stderr}")
            # Clean up temp file if it exists
            if os.path.exists(temp_ffmpeg_output):
                try:
                    os.remove(temp_ffmpeg_output)
                except:
                    pass
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


def convert_to_json_serializable(obj):
    """
    Convert NumPy and other non-serializable types to JSON-serializable types
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def save_annotation_json(json_data, output_path):
    """
    Save annotation data to JSON file with proper formatting
    
    Args:
        json_data: Dictionary containing annotation data
        output_path: Path to save JSON file
    """
    try:
        # Convert all NumPy types to native Python types
        serializable_data = convert_to_json_serializable(json_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] JSON annotation saved: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save JSON annotation: {e}")
        import traceback
        traceback.print_exc()
        return False


def annotate_video_with_speaker_subtitles(input_video, output_video, centers_data_path, 
                                         subtitle_path, model_dir, detection_interval=2,
                                         similarity_threshold=0.65, speaking_threshold=0.8,
                                         preserve_audio=True, smoothing_alpha=0.3,
                                         generate_json=True, subtitle_offset=0.0):
    """
    Annotate video with color-coded speaker subtitles with continuous display
    
    Args:
        input_video: Input video path
        output_video: Output video path
        centers_data_path: Path to cluster center data
        subtitle_path: Path to subtitle file (SRT format)
        model_dir: FaceNet model directory
        detection_interval: Interval for face detection (process every N frames)
        similarity_threshold: Threshold for face matching (default: 0.65)
        speaking_threshold: Threshold for speaking detection (default: 0.8, higher = stricter)
        preserve_audio: Whether to preserve original audio
        smoothing_alpha: Smoothing factor for bbox positions (0-1, default: 0.3)
                         Lower = more smoothing, higher = faster adaptation
        generate_json: Whether to generate JSON annotation file (default: True)
        subtitle_offset: Time offset in seconds to apply to all subtitles
                        Positive = delay subtitles, Negative = advance subtitles
    """
    # Load centers data
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    # Initialize color manager
    color_manager = ColorManager(len(centers))
    print(f"Initialized color manager with {len(centers)} distinct colors")
    print(f"Face matching threshold: {similarity_threshold}")
    print(f"Speaking detection threshold: {speaking_threshold}")
    print(f"BBox smoothing alpha: {smoothing_alpha}")
    print(f"Subtitle offset: {subtitle_offset:.3f}s")
    print(f"Generate JSON annotation: {generate_json}")
    
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
        # Generate temporary filename that won't conflict
        output_dir = os.path.dirname(output_video)
        output_name = os.path.basename(output_video)
        name_without_ext, ext = os.path.splitext(output_name)
        temp_video = os.path.join(output_dir, f'{name_without_ext}_temp_no_audio{ext}')
        final_output = output_video
        print(f"  Temporary video (no audio): {temp_video}")
        print(f"  Final output (with audio): {final_output}")
    else:
        temp_video = output_video
        final_output = None
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    # JSON annotation data structure
    json_annotations = {}
    
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
                        speaking_idx = detect_speaking_face(prev_faces, faces, threshold=speaking_threshold)
                    
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
            
            # Calculate detection statistics
            total_frames_with_faces = len([r for r in detection_results.values() if r['faces']])
            total_faces_detected = sum(len(r['faces']) for r in detection_results.values())
            frames_with_speaking = len([r for r in detection_results.values() if r['speaking_idx'] >= 0])
            
            print(f"\nDetection Statistics:")
            print(f"  Frames processed: {len(detection_results)}")
            print(f"  Frames with valid faces: {total_frames_with_faces}")
            print(f"  Total faces detected: {total_faces_detected}")
            print(f"  Frames with speaking detection: {frames_with_speaking}")
            
            # PHASE 2: Determine speaker for each subtitle segment
            print("\n" + "="*70)
            print("PHASE 2: Determining speakers for subtitle segments")
            print("="*70)
            
            subtitle_speakers = {}
            
            for subtitle in tqdm(subtitles, desc="Processing subtitles"):
                speaker_id, speaker_bbox, speaker_ids, all_faces = determine_speaker_for_subtitle(
                    subtitle, detection_results, fps
                )
                
                # Calculate timestamps WITH OFFSET APPLIED
                start_timestamp = subtitle.start.ordinal / 1000.0 + subtitle_offset
                end_timestamp = subtitle.end.ordinal / 1000.0 + subtitle_offset
                
                # Initialize bbox_smoother as None (will be created in PHASE 3)
                bbox_smoother = None
                
                # Store with offset applied
                subtitle_speakers[subtitle.index] = {
                    'speaker_id': speaker_id,
                    'speaker_bbox': speaker_bbox,
                    'speaker_ids': speaker_ids if speaker_ids else [],
                    'all_faces': all_faces,
                    'bbox_smoother': bbox_smoother,
                    'smoothing_alpha': smoothing_alpha,
                    'text': subtitle.text_without_tags,
                    'start_ms': subtitle.start.ordinal + int(subtitle_offset * 1000),
                    'end_ms': subtitle.end.ordinal + int(subtitle_offset * 1000),
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp
                }
                
                # Prepare JSON annotation entries
                if generate_json:
                    # Start entry
                    start_key = f"{subtitle.index}_start"
                    
                    # Convert speaker_id to native Python int or None
                    json_speaker_id = None
                    if speaker_id is not None and speaker_id >= 0:
                        json_speaker_id = int(speaker_id)
                    
                    # Convert speaker_ids list to native Python ints
                    json_speaker_ids = [int(sid) for sid in speaker_ids] if speaker_ids else []
                    
                    json_annotations[start_key] = {
                        'subtitle_id': int(subtitle.index),
                        'position': 'start',
                        'timestamp': round(float(start_timestamp), 3),
                        'text': subtitle.text_without_tags,
                        'speaker_id': json_speaker_id,
                        'speaker_ids': json_speaker_ids,
                        'all_faces': [],
                        'status': 'unknown',
                        'note': ''
                    }
                    
                    # Determine status
                    if speaker_id is not None and speaker_id >= 0:
                        json_annotations[start_key]['status'] = 'speaker_identified'
                    elif all_faces:
                        json_annotations[start_key]['status'] = 'faces_detected_no_speaker'
                    else:
                        json_annotations[start_key]['status'] = 'speaker_not_visible'
                    
                    # Populate all_faces with simplified data
                    if all_faces:
                        first_frame_faces = all_faces[0]['faces']
                        json_annotations[start_key]['all_faces'] = [
                            {
                                'face_id': int(face['face_id']),
                                'bbox': [int(x) for x in face['bbox']],
                                'similarity': float(round(face['similarity'], 3))
                            }
                            for face in first_frame_faces
                        ]
            
            # Calculate speaker statistics
            identified_speakers = len([s for s in subtitle_speakers.values() if s['speaker_id'] is not None and s['speaker_id'] >= 0])
            narrations = len(subtitle_speakers) - identified_speakers
            
            print(f"\nSubtitle Statistics:")
            print(f"  Total subtitles: {len(subtitle_speakers)}")
            print(f"  With identified speaker: {identified_speakers}")
            print(f"  Narration/No speaker: {narrations}")
            
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
                    # Get the speaker ID determined in PHASE 2 (from voting across subtitle time range)
                    speaker_id = current_subtitle_info['speaker_id']
                    
                    # Determine color based on speaker assignment
                    if speaker_id is not None and speaker_id >= 0:
                        # Subtitle has an assigned speaker - use speaker's color
                        color = color_manager.get_color(speaker_id)
                    else:
                        # Subtitle has no assigned speaker (narration, unknown, etc.) - use white
                        color = color_manager.default_color
                    
                    # Check if the assigned speaker is visible in current frame
                    speaker_visible = False
                    speaker_bbox = None
                    
                    if speaker_id is not None and speaker_id >= 0:
                        for face in faces_in_frame:
                            if face.get('match_idx') == speaker_id:
                                speaker_visible = True
                                speaker_bbox = face['bbox']
                                break
                    
                    # Initialize or update BBoxSmoother
                    bbox_smoother = current_subtitle_info.get('bbox_smoother')
                    display_bbox = None
                    
                    if speaker_visible and speaker_bbox is not None:
                        # Speaker is visible in current frame
                        if bbox_smoother is None:
                            # First time detecting this speaker in this subtitle segment
                            smoothing_alpha = current_subtitle_info.get('smoothing_alpha', 0.3)
                            bbox_smoother = BBoxSmoother(speaker_bbox, alpha=smoothing_alpha)
                            current_subtitle_info['bbox_smoother'] = bbox_smoother
                            display_bbox = speaker_bbox
                        else:
                            # Update smoother with new detection
                            display_bbox = bbox_smoother.update(speaker_bbox)
                    else:
                        # Speaker is NOT visible in current frame
                        # Do NOT update smoother, do NOT display bbox
                        display_bbox = None
                    
                    # Render subtitle and bounding boxes
                    if display_bbox and speaker_id >= 0:
                        # Case 1: Speaker is visible - display subtitle beside face
                        x1, y1, x2, y2 = display_bbox
                        
                        # Draw thick bounding box around speaker's face
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        
                        # Draw character ID label
                        id_label = f"Character {speaker_id}"
                        label_font = cv2.FONT_HERSHEY_DUPLEX
                        label_scale = 0.7
                        label_thickness = 2
                        label_size = cv2.getTextSize(id_label, label_font, 
                                                    label_scale, label_thickness)[0]
                        
                        # Draw label background
                        label_padding = 8
                        label_width = label_size[0] + label_padding * 2
                        cv2.rectangle(frame, 
                                    (x1, y1 - label_size[1] - label_padding * 2), 
                                    (x1 + label_width, y1), 
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, id_label, 
                                  (x1 + label_padding, y1 - label_padding),
                                  label_font, label_scale, (255, 255, 255), 
                                  label_thickness, cv2.LINE_AA)
                        
                        # Draw subtitle beside the face
                        frame = draw_subtitle_beside_label(frame, current_subtitle, 
                                                         display_bbox, color, label_width)
                    else:
                        # Case 2: Speaker is NOT visible (or no speaker assigned)
                        # Display subtitle at bottom with appropriate color
                        # - If speaker assigned: use speaker's color
                        # - If no speaker: use white
                        frame = draw_subtitle_at_bottom(frame, current_subtitle, color)
                    
                    # Draw bounding boxes for all other detected faces (non-speakers)
                    for face in faces_in_frame:
                        face_id = face.get('match_idx', -1)
                        if face_id >= 0 and face_id != speaker_id:
                            # Draw thin bounding box for non-speaking characters
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
    
    # Save JSON annotation if requested
    if generate_json and json_annotations:
        json_output_path = output_video.replace('.mp4', '_annotation.json')
        save_annotation_json(json_annotations, json_output_path)
    
    # Merge audio if requested
    if preserve_audio and final_output:
        print("\n" + "="*70)
        print("PHASE 4: Merging audio from original video")
        print("="*70)
        
        success = merge_audio_to_video(temp_video, input_video, final_output)
        
        if success:
            # Clean up temporary video file (without audio)
            try:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                    print(f"Cleaned up temporary file: {os.path.basename(temp_video)}")
            except Exception as e:
                print(f"Note: Could not remove temporary file: {e}")
            
            print(f"\n{'='*70}")
            print("FINAL OUTPUT WITH AUDIO:")
            print(f"  Video: {final_output}")
            if generate_json:
                print(f"  JSON: {json_output_path}")
            print("="*70)
        else:
            print(f"\nWarning: Audio merging failed. Video without audio saved at:")
            print(f"  {temp_video}")
    else:
        print(f"\n{'='*70}")
        print("FINAL OUTPUT (NO AUDIO):")
        print(f"  Video: {temp_video}")
        if generate_json:
            print(f"  JSON: {json_output_path}")
        print("="*70)


if __name__ == "__main__":
    # Configuration
    input_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\S02\ep10.mp4"
    output_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_s2ep10\color_coded_subtitles.mp4"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_s2ep10\centers\centers_data.pkl"
    subtitle_path = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\S02\subtitles\s2ep10.srt"
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
        speaking_threshold=0.7,
        preserve_audio=True,
        subtitle_offset=0.0,
        generate_json=True
    )