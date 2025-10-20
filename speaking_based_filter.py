# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math
import facenet.src.align.detect_face as detect_face

def detect_faces_with_speaking_filter(sess, frame_paths, output_dir, video_path=None,
                                     min_face_size=60, face_size=160, margin=44):
    """
    Detect faces with speaking-based filtering
    
    Args:
        sess: TensorFlow session
        frame_paths: List of frame paths
        output_dir: Output directory
        video_path: Original video path (for audio analysis)
        min_face_size: Minimum face size
        face_size: Face image size
        margin: Margin around face
        
    Returns:
        List of detected face paths
    """
    print("Creating MTCNN network for speaking-aware face detection...")
    pnet, rnet, onet = create_mtcnn(sess, None)
    
    # Initialize audio analyzer if video is provided
    audio_analyzer = None
    if video_path:
        try:
            audio_analyzer = AudioActivityAnalyzer(video_path)
            print("Audio analyzer initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize audio analyzer: {e}")
            audio_analyzer = None
    
    # Initialize face movement tracker
    movement_tracker = FaceMovementTracker()
    
    face_paths = []
    face_count = 0
    
    print("Detecting faces with speaking-based filtering...")
    for frame_idx, frame_path in enumerate(tqdm(frame_paths)):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        original_frame = frame.copy()
        
        # Calculate timestamp for this frame
        # Assume frame_paths are sequential with consistent interval
        timestamp = frame_idx * 30 / 30.0  # Adjust based on your frame interval
        
        # Get audio activity at this timestamp
        audio_active = False
        audio_energy = 0
        if audio_analyzer:
            audio_active, audio_energy = audio_analyzer.is_speech_active(timestamp)
        
        # Detect all faces with lenient parameters
        all_bboxes = detect_all_faces_lenient(frame, pnet, rnet, onet, min_face_size)
        
        if len(all_bboxes) == 0:
            continue
        
        # Calculate speaking likelihood for each face
        face_scores = []
        for bbox in all_bboxes:
            # Calculate face movement (mouth movement indicator)
            movement_score = movement_tracker.calculate_movement_score(
                bbox, frame, frame_idx
            )
            
            # Calculate position score (center faces more likely to speak)
            position_score = calculate_position_score(bbox, frame.shape)
            
            # Calculate size score (larger faces more likely to be main speakers)
            size_score = calculate_size_score(bbox, frame.shape)
            
            # Combine scores to estimate speaking likelihood
            speaking_likelihood = combine_speaking_scores(
                movement_score, position_score, size_score, 
                audio_active, audio_energy
            )
            
            face_scores.append({
                'bbox': bbox,
                'speaking_likelihood': speaking_likelihood,
                'movement_score': movement_score,
                'position_score': position_score,
                'size_score': size_score
            })
        
        # Filter faces based on speaking likelihood
        filtered_faces = filter_by_speaking_likelihood(
            face_scores, audio_active, max_faces=8
        )
        
        # Process and save filtered faces
        for i, face_info in enumerate(filtered_faces):
            bbox = face_info['bbox']
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
            face_name = f"{os.path.splitext(frame_name)[0]}_speaker_{i}.jpg"
            face_path = os.path.join(output_dir, face_name)
            
            cv2.imwrite(face_path, face_resized)
            face_paths.append(face_path)
            face_count += 1
    
    print(f"A total of {face_count} speaker faces were detected")
    return face_paths


class AudioActivityAnalyzer:
    """Analyze audio to detect speech activity"""
    
    def __init__(self, video_path):
        """
        Initialize audio analyzer
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.audio_data = None
        self.sample_rate = None
        self.speech_segments = []
        
        self._extract_audio()
        self._detect_speech_segments()
    
    def _extract_audio(self):
        """Extract audio from video"""
        try:
            import librosa
            import subprocess
            import tempfile
            
            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_audio_path = tmp.name
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', self.video_path, '-vn', 
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', temp_audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Load audio
            self.audio_data, self.sample_rate = librosa.load(temp_audio_path, sr=16000)
            
            # Clean up
            os.unlink(temp_audio_path)
            
            print(f"Audio extracted: {len(self.audio_data)/self.sample_rate:.2f}s at {self.sample_rate}Hz")
            
        except Exception as e:
            print(f"Warning: Could not extract audio: {e}")
            self.audio_data = None
    
    def _detect_speech_segments(self):
        """Detect segments with speech activity"""
        if self.audio_data is None:
            return
        
        # Simple energy-based speech detection
        frame_length = int(0.02 * self.sample_rate)  # 20ms frames
        hop_length = int(0.01 * self.sample_rate)    # 10ms hop
        
        # Calculate energy for each frame
        energies = []
        for i in range(0, len(self.audio_data) - frame_length, hop_length):
            frame = self.audio_data[i:i+frame_length]
            energy = np.sum(frame ** 2)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Adaptive threshold (mean + 0.5 * std)
        threshold = np.mean(energies) + 0.5 * np.std(energies)
        
        # Find speech segments
        is_speech = energies > threshold
        
        # Convert to time segments
        self.speech_segments = []
        in_speech = False
        start_time = 0
        
        for i, active in enumerate(is_speech):
            time = i * hop_length / self.sample_rate
            
            if active and not in_speech:
                # Start of speech
                start_time = time
                in_speech = True
            elif not active and in_speech:
                # End of speech
                self.speech_segments.append((start_time, time))
                in_speech = False
        
        print(f"Detected {len(self.speech_segments)} speech segments")
    
    def is_speech_active(self, timestamp, window=0.2):
        """
        Check if speech is active at given timestamp
        
        Args:
            timestamp: Time in seconds
            window: Time window to check (seconds)
            
        Returns:
            (is_active, energy): Tuple of bool and float
        """
        if self.audio_data is None:
            return False, 0
        
        # Check if timestamp falls in any speech segment
        for start, end in self.speech_segments:
            if start - window <= timestamp <= end + window:
                # Calculate energy in this window
                start_sample = int((timestamp - window/2) * self.sample_rate)
                end_sample = int((timestamp + window/2) * self.sample_rate)
                
                start_sample = max(0, start_sample)
                end_sample = min(len(self.audio_data), end_sample)
                
                if end_sample > start_sample:
                    segment = self.audio_data[start_sample:end_sample]
                    energy = np.mean(np.abs(segment))
                    return True, energy
        
        return False, 0


class FaceMovementTracker:
    """Track face movement between frames to detect speaking"""
    
    def __init__(self, history_length=5):
        """
        Initialize movement tracker
        
        Args:
            history_length: Number of previous frames to track
        """
        self.history_length = history_length
        self.face_history = {}  # face_id -> list of (frame_idx, bbox, features)
    
    def calculate_movement_score(self, bbox, frame, frame_idx):
        """
        Calculate movement score for a face (especially mouth region)
        
        Args:
            bbox: Face bounding box [x1, y1, x2, y2]
            frame: Current frame
            frame_idx: Frame index
            
        Returns:
            Movement score (0-1)
        """
        x1, y1, x2, y2 = bbox[:4]
        
        # Calculate face ID based on position
        face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        face_id = self._find_matching_face_id(face_center, frame_idx)
        
        if face_id is None:
            # New face, create ID
            face_id = f"{int(face_center[0])}_{int(face_center[1])}"
            self.face_history[face_id] = []
        
        # Extract mouth region (lower 1/3 of face)
        face_height = y2 - y1
        mouth_y1 = int(y1 + face_height * 0.6)
        mouth_y2 = y2
        
        mouth_region = frame[mouth_y1:mouth_y2, x1:x2]
        
        if mouth_region.size == 0:
            return 0.0
        
        # Calculate features for mouth region
        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouth_features = cv2.mean(gray_mouth)[0]
        
        # Store current frame data
        self.face_history[face_id].append({
            'frame_idx': frame_idx,
            'bbox': bbox,
            'mouth_features': mouth_features
        })
        
        # Keep only recent history
        if len(self.face_history[face_id]) > self.history_length:
            self.face_history[face_id].pop(0)
        
        # Calculate movement if we have history
        if len(self.face_history[face_id]) < 2:
            return 0.0
        
        # Calculate mouth feature variance (indicates movement)
        recent_features = [h['mouth_features'] for h in self.face_history[face_id]]
        feature_variance = np.var(recent_features)
        
        # Normalize variance to 0-1 score
        movement_score = min(1.0, feature_variance / 100.0)
        
        return movement_score
    
    def _find_matching_face_id(self, face_center, frame_idx, threshold=50):
        """Find matching face ID from previous frames"""
        for face_id, history in self.face_history.items():
            if not history:
                continue
            
            # Check most recent entry
            last_entry = history[-1]
            if frame_idx - last_entry['frame_idx'] > 10:
                # Too old, not a match
                continue
            
            last_bbox = last_entry['bbox']
            last_center = ((last_bbox[0] + last_bbox[2]) / 2, 
                          (last_bbox[1] + last_bbox[3]) / 2)
            
            # Calculate distance
            distance = np.sqrt((face_center[0] - last_center[0])**2 + 
                             (face_center[1] - last_center[1])**2)
            
            if distance < threshold:
                return face_id
        
        return None


def detect_all_faces_lenient(frame, pnet, rnet, onet, min_face_size):
    """
    Detect all faces with lenient parameters
    """
    import facenet.src.align.detect_face as detect_face
    
    # Convert to RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Apply mild contrast enhancement
    frame_rgb = mild_contrast_enhancement(frame_rgb)
    
    # Lenient thresholds to detect all faces
    threshold = [0.6, 0.7, 0.7]
    
    # Detect faces
    bounding_boxes, _ = detect_face.detect_face(
        frame_rgb, min_face_size, pnet, rnet, onet, threshold, 0.709
    )
    
    # If no detection, try with even lower threshold
    if len(bounding_boxes) == 0:
        threshold = [0.5, 0.6, 0.6]
        bounding_boxes, _ = detect_face.detect_face(
            frame_rgb, min_face_size * 0.8, pnet, rnet, onet, threshold, 0.6
        )
    
    return [bbox.astype(np.int32) for bbox in bounding_boxes]


def calculate_position_score(bbox, frame_shape):
    """
    Calculate position score (center faces more likely to speak)
    
    Args:
        bbox: Face bounding box
        frame_shape: Frame shape (height, width, channels)
        
    Returns:
        Position score (0-1)
    """
    frame_height, frame_width = frame_shape[:2]
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    # Calculate face center
    face_center_x = (bbox[0] + bbox[2]) / 2
    face_center_y = (bbox[1] + bbox[3]) / 2
    
    # Calculate normalized distance from center
    dist_x = abs(face_center_x - frame_center_x) / frame_width
    dist_y = abs(face_center_y - frame_center_y) / frame_height
    
    # Combined distance
    distance = np.sqrt(dist_x**2 + dist_y**2)
    
    # Convert to score (closer to center = higher score)
    position_score = max(0, 1.0 - distance * 1.5)
    
    return position_score


def calculate_size_score(bbox, frame_shape):
    """
    Calculate size score (larger faces more likely to be main speakers)
    
    Args:
        bbox: Face bounding box
        frame_shape: Frame shape
        
    Returns:
        Size score (0-1)
    """
    frame_area = frame_shape[0] * frame_shape[1]
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    face_area = face_width * face_height
    
    face_area_ratio = face_area / frame_area
    
    # Normalize to 0-1 (0.05 area ratio = 1.0 score)
    size_score = min(1.0, face_area_ratio / 0.05)
    
    return size_score


def combine_speaking_scores(movement_score, position_score, size_score, 
                            audio_active, audio_energy):
    """
    Combine multiple scores to estimate speaking likelihood
    
    Args:
        movement_score: Face movement score (0-1)
        position_score: Position score (0-1)
        size_score: Size score (0-1)
        audio_active: Whether audio is active (bool)
        audio_energy: Audio energy level (0-1)
        
    Returns:
        Combined speaking likelihood (0-1)
    """
    # Base visual score (without audio)
    visual_score = (
        movement_score * 0.50 +  # Movement is most important visual cue
        position_score * 0.25 +   # Center faces more likely
        size_score * 0.25         # Larger faces more likely
    )
    
    # If audio is available, use it to boost scores
    if audio_active:
        # Audio active: boost scores for likely speakers
        # High movement + audio = very likely speaking
        audio_boost = 0.3 * audio_energy
        speaking_likelihood = min(1.0, visual_score + audio_boost)
    else:
        # No audio or audio inactive: rely more on visual cues
        speaking_likelihood = visual_score
    
    return speaking_likelihood


def filter_by_speaking_likelihood(face_scores, audio_active, max_faces=8):
    """
    Filter faces based on speaking likelihood
    
    Args:
        face_scores: List of face score dictionaries
        audio_active: Whether audio is active
        max_faces: Maximum faces to keep
        
    Returns:
        Filtered list of faces
    """
    if not face_scores:
        return []
    
    # Sort by speaking likelihood
    face_scores.sort(key=lambda x: x['speaking_likelihood'], reverse=True)
    
    if audio_active:
        # When audio is active, keep faces with high speaking likelihood
        # Threshold: 0.3 (relatively lenient to avoid missing speakers)
        threshold = 0.3
        filtered = [f for f in face_scores if f['speaking_likelihood'] > threshold]
        
        # If too few faces, keep top faces anyway
        if len(filtered) < 2:
            filtered = face_scores[:min(3, len(face_scores))]
        
        # Limit to max_faces
        filtered = filtered[:max_faces]
    else:
        # When audio is inactive, be more lenient
        # Keep top faces but with lower threshold
        threshold = 0.2
        filtered = [f for f in face_scores if f['speaking_likelihood'] > threshold]
        
        # Keep at least top 3 faces
        if len(filtered) < 3:
            filtered = face_scores[:min(3, len(face_scores))]
        
        # Limit to max_faces
        filtered = filtered[:max_faces]
    
    return filtered


def create_mtcnn(sess, model_path):
    """Create MTCNN detection network"""
    if not model_path:
        model_path = None
    
    pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)
    return pnet, rnet, onet


def mild_contrast_enhancement(image, clip_limit=1.5, tile_grid_size=(8, 8)):
    """Apply mild contrast enhancement"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced


def mild_preprocessing(face_img):
    """Apply mild preprocessing to face image"""
    if face_img is None or face_img.size == 0:
        return face_img
    
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 80:
        clip_limit = 1.2
    elif brightness > 180:
        clip_limit = 0.6
    else:
        clip_limit = 0.8

    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < 50:
        h = 3
    elif variance > 150:
        h = 7
    else:
        h = 5

    img_denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, h, h, 7, 21)
    
    return img_denoised