# -*- coding: utf-8 -*-
"""
Speaking-based face detection and filtering for video annotation
Use this in enhanced_video_annotation.py
"""

import cv2
import numpy as np
from collections import deque
import os

def detect_and_match_faces_with_speaking_filter(
    frame, pnet, rnet, onet, sess, images_placeholder, 
    embeddings, phase_train_placeholder, centers, 
    frame_histories, audio_analyzer=None, frame_idx=0,
    min_face_size=60, temporal_weight=0.25):
    """
    Enhanced face detection with speaking-based filtering
    
    This function:
    1. Detects all faces in frame (lenient)
    2. Estimates which faces are speaking
    3. Filters out non-speaking background faces
    4. Matches remaining faces with character centers
    
    Args:
        frame: Input video frame
        pnet, rnet, onet: MTCNN detector components
        sess: TensorFlow session
        images_placeholder: FaceNet input placeholder
        embeddings: FaceNet embeddings tensor
        phase_train_placeholder: FaceNet phase train placeholder
        centers: Character cluster centers
        frame_histories: Dictionary for temporal tracking
        audio_analyzer: AudioActivityAnalyzer instance (optional)
        frame_idx: Current frame index
        min_face_size: Minimum face size for detection
        temporal_weight: Weight for temporal consistency
        
    Returns:
        List of face dictionaries with detection and matching info
    """
    from enhanced_face_preprocessing import detect_foreground_faces_in_frame
    import facenet.src.facenet as facenet
    
    # Step 1: Detect all faces with lenient parameters
    all_bboxes = detect_all_faces_lenient_annotation(
        frame, pnet, rnet, onet, min_face_size
    )
    
    if not all_bboxes:
        return []
    
    # Step 2: Check if audio is active at this frame
    audio_active = False
    audio_energy = 0.0
    if audio_analyzer:
        timestamp = frame_idx / 30.0  # Assume 30 fps, adjust as needed
        audio_active, audio_energy = audio_analyzer.is_speech_active(timestamp)
    
    # Step 3: Calculate speaking likelihood for each face
    face_candidates = []
    for bbox in all_bboxes:
        # Calculate movement score from previous frames
        movement_score = calculate_face_movement_from_history(
            bbox, frame_histories, frame_idx
        )
        
        # Calculate position and size scores
        position_score = calculate_position_score_annotation(bbox, frame.shape)
        size_score = calculate_size_score_annotation(bbox, frame.shape)
        
        # Combine to get speaking likelihood
        speaking_likelihood = combine_speaking_scores_annotation(
            movement_score, position_score, size_score, 
            audio_active, audio_energy
        )
        
        face_candidates.append({
            'bbox': bbox,
            'speaking_likelihood': speaking_likelihood,
            'movement_score': movement_score
        })
    
    # Step 4: Filter faces by speaking likelihood
    filtered_candidates = filter_faces_by_speaking_annotation(
        face_candidates, audio_active, max_faces=8
    )
    
    if not filtered_candidates:
        return []
    
    # Step 5: Process filtered faces - extract features and match
    faces = []
    face_crops = []
    face_bboxes = []
    
    # Convert to RGB
    if frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    for candidate in filtered_candidates:
        bbox = candidate['bbox']
        
        # Calculate adaptive margin
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        margin = int(bbox_size * 0.2)
        
        x1 = max(0, bbox[0] - margin)
        y1 = max(0, bbox[1] - margin)
        x2 = min(frame.shape[1], bbox[2] + margin)
        y2 = min(frame.shape[0], bbox[3] + margin)
        
        face = frame_rgb[y1:y2, x1:x2, :]
        
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue
        
        # Resize to FaceNet input size
        face_resized = cv2.resize(face, (160, 160))
        
        # FaceNet preprocessing
        face_prewhitened = facenet.prewhiten(face_resized)
        
        face_crops.append(face_prewhitened)
        face_bboxes.append((x1, y1, x2, y2))
    
    if not face_crops:
        return []
    
    # Batch compute face encodings
    face_batch = np.stack(face_crops)
    feed_dict = {images_placeholder: face_batch, phase_train_placeholder: False}
    face_encodings = sess.run(embeddings, feed_dict=feed_dict)
    
    # Match each face with cluster centers
    for i, (bbox, encoding) in enumerate(zip(face_bboxes, face_encodings)):
        x1, y1, x2, y2 = bbox
        face_id = f"{(x1 + x2) // 2}_{(y1 + y2) // 2}"
        
        # Get current match
        match_idx, similarity, all_similarities = match_face_with_centers_annotation(
            encoding, centers
        )
        
        # Temporal consistency processing
        if face_id in frame_histories:
            history = frame_histories[face_id]
            
            if similarity > 0.4:
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


def detect_all_faces_lenient_annotation(frame, pnet, rnet, onet, min_face_size):
    """
    Detect all faces with lenient parameters for annotation
    """
    import facenet.src.align.detect_face as detect_face
    
    # Convert to RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Lenient thresholds to detect all possible faces
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


def calculate_face_movement_from_history(bbox, frame_histories, frame_idx):
    """
    Calculate face movement score from frame history
    
    Args:
        bbox: Current face bounding box
        frame_histories: Dictionary of face histories
        frame_idx: Current frame index
        
    Returns:
        Movement score (0-1)
    """
    x1, y1, x2, y2 = bbox[:4]
    face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    face_id = f"{int(face_center[0])}_{int(face_center[1])}"
    
    # Try to find matching face in history
    matched_id = None
    min_distance = float('inf')
    
    for hist_id, history in frame_histories.items():
        if len(history) == 0:
            continue
        
        # Parse historical face ID to get center
        try:
            hist_x, hist_y = map(int, hist_id.split('_'))
            distance = np.sqrt((face_center[0] - hist_x)**2 + (face_center[1] - hist_y)**2)
            
            if distance < min_distance and distance < 50:
                min_distance = distance
                matched_id = hist_id
        except:
            continue
    
    if matched_id is None or len(frame_histories[matched_id]) < 2:
        return 0.0
    
    # Calculate variance in similarity scores as proxy for movement
    # Speaking causes changes in face appearance/position
    recent_sims = [sim for _, sim in list(frame_histories[matched_id])[-5:]]
    
    if len(recent_sims) < 2:
        return 0.0
    
    sim_variance = np.var(recent_sims)
    
    # Higher variance suggests movement/speaking
    movement_score = min(1.0, sim_variance * 10)
    
    return movement_score


def calculate_position_score_annotation(bbox, frame_shape):
    """
    Calculate position score for annotation
    Center and upper faces more likely to be speaking
    """
    frame_height, frame_width = frame_shape[:2]
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    # Calculate face center
    face_center_x = (bbox[0] + bbox[2]) / 2
    face_center_y = (bbox[1] + bbox[3]) / 2
    
    # Horizontal distance from center
    dist_x = abs(face_center_x - frame_center_x) / frame_width
    
    # Vertical position (upper 2/3 of frame preferred)
    vertical_pos = face_center_y / frame_height
    
    # Horizontal score
    h_score = max(0, 1.0 - dist_x * 2)
    
    # Vertical score (prefer upper 2/3)
    if vertical_pos < 0.67:
        v_score = 1.0
    else:
        v_score = max(0.3, 1.0 - (vertical_pos - 0.67) * 2)
    
    # Combined position score
    position_score = (h_score * 0.6 + v_score * 0.4)
    
    return position_score


def calculate_size_score_annotation(bbox, frame_shape):
    """
    Calculate size score for annotation
    Larger faces more likely to be main speakers
    """
    frame_area = frame_shape[0] * frame_shape[1]
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    face_area = face_width * face_height
    
    face_area_ratio = face_area / frame_area
    
    # Normalize to 0-1 score
    # 0.05 area ratio (5%) = 1.0 score
    size_score = min(1.0, face_area_ratio / 0.05)
    
    return size_score


def combine_speaking_scores_annotation(movement_score, position_score, size_score, 
                                       audio_active, audio_energy):
    """
    Combine multiple scores to estimate speaking likelihood for annotation
    
    Scoring weights:
    - Movement: 45% (most important - speaking causes face movement)
    - Position: 25% (center/upper faces more likely to speak)
    - Size: 20% (larger faces more likely to be main speakers)
    - Audio boost: +10% when active
    """
    # Base visual score
    visual_score = (
        movement_score * 0.45 +
        position_score * 0.25 +
        size_score * 0.20
    )
    
    # Audio boost when available
    if audio_active:
        # When audio is active, boost scores
        audio_boost = 0.10 * min(1.0, audio_energy * 2)
        speaking_likelihood = min(1.0, visual_score + audio_boost)
    else:
        # No audio: rely on visual cues only
        speaking_likelihood = visual_score
    
    return speaking_likelihood


def filter_faces_by_speaking_annotation(face_candidates, audio_active, max_faces=8):
    """
    Filter faces by speaking likelihood for annotation
    
    Strategy:
    - When audio active: Keep high-likelihood faces (threshold 0.25)
    - When audio inactive: More lenient (threshold 0.15)
    - Always keep at least top 2 faces
    - Never exceed max_faces limit
    """
    if not face_candidates:
        return []
    
    # Sort by speaking likelihood
    face_candidates.sort(key=lambda x: x['speaking_likelihood'], reverse=True)
    
    if audio_active:
        # Audio is active - filter more aggressively
        threshold = 0.25
        filtered = [f for f in face_candidates if f['speaking_likelihood'] > threshold]
        
        # Keep at least top 2 faces
        if len(filtered) < 2:
            filtered = face_candidates[:min(2, len(face_candidates))]
        
        # Don't exceed max
        filtered = filtered[:max_faces]
        
    else:
        # No audio - be more lenient to avoid missing speakers
        threshold = 0.15
        filtered = [f for f in face_candidates if f['speaking_likelihood'] > threshold]
        
        # Keep at least top 3 faces
        if len(filtered) < 3:
            filtered = face_candidates[:min(3, len(face_candidates))]
        
        # Don't exceed max
        filtered = filtered[:max_faces]
    
    return filtered


def match_face_with_centers_annotation(face_encoding, centers, threshold=0.55):
    """
    Match a face encoding with cluster centers
    
    Args:
        face_encoding: Face encoding vector
        centers: List of cluster center encodings
        threshold: Similarity threshold
        
    Returns:
        Tuple of (match_idx, similarity, all_similarities)
    """
    if len(centers) == 0:
        return -1, 0, []
    
    # Calculate cosine similarity with all centers
    similarities = np.dot(centers, face_encoding)
    
    # Find the most similar center
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]
    
    # Return match if similarity exceeds threshold
    if best_similarity > threshold:
        return best_index, best_similarity, similarities
    else:
        return -1, 0, similarities


class AudioActivityAnalyzer:
    """
    Analyze audio to detect speech activity
    Use this for real-time annotation with audio analysis
    """
    
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
        """Extract audio from video using ffmpeg"""
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
            
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                print(f"Warning: ffmpeg failed to extract audio")
                self.audio_data = None
                return
            
            # Load audio with librosa
            self.audio_data, self.sample_rate = librosa.load(temp_audio_path, sr=16000)
            
            # Clean up
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            print(f"Audio extracted: {len(self.audio_data)/self.sample_rate:.2f}s at {self.sample_rate}Hz")
            
        except ImportError:
            print("Warning: librosa not installed. Audio analysis disabled.")
            print("Install with: pip install librosa")
            self.audio_data = None
        except Exception as e:
            print(f"Warning: Could not extract audio: {e}")
            self.audio_data = None
    
    def _detect_speech_segments(self):
        """Detect segments with speech activity using energy-based VAD"""
        if self.audio_data is None:
            return
        
        # Frame-based energy calculation
        frame_length = int(0.02 * self.sample_rate)  # 20ms frames
        hop_length = int(0.01 * self.sample_rate)    # 10ms hop
        
        # Calculate RMS energy for each frame
        energies = []
        for i in range(0, len(self.audio_data) - frame_length, hop_length):
            frame = self.audio_data[i:i+frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Adaptive threshold based on signal statistics
        # Use median + k*MAD for robustness
        median_energy = np.median(energies)
        mad = np.median(np.abs(energies - median_energy))
        threshold = median_energy + 3 * mad
        
        # Mark frames as speech/non-speech
        is_speech = energies > threshold
        
        # Convert to time segments with smoothing
        self.speech_segments = []
        in_speech = False
        start_time = 0
        min_speech_duration = 0.3  # Minimum 300ms speech segment
        
        for i, active in enumerate(is_speech):
            time = i * hop_length / self.sample_rate
            
            if active and not in_speech:
                # Start of potential speech segment
                start_time = time
                in_speech = True
            elif not active and in_speech:
                # End of speech segment
                duration = time - start_time
                if duration >= min_speech_duration:
                    self.speech_segments.append((start_time, time))
                in_speech = False
        
        print(f"Detected {len(self.speech_segments)} speech segments")
    
    def is_speech_active(self, timestamp, window=0.3):
        """
        Check if speech is active at given timestamp
        
        Args:
            timestamp: Time in seconds
            window: Time window to check (seconds)
            
        Returns:
            Tuple of (is_active: bool, energy: float)
        """
        if self.audio_data is None:
            return False, 0.0
        
        # Check if timestamp falls in any speech segment (with tolerance)
        for start, end in self.speech_segments:
            if start - window <= timestamp <= end + window:
                # Calculate energy in this window
                start_sample = int(max(0, (timestamp - window/2) * self.sample_rate))
                end_sample = int(min(len(self.audio_data), (timestamp + window/2) * self.sample_rate))
                
                if end_sample > start_sample:
                    segment = self.audio_data[start_sample:end_sample]
                    energy = np.sqrt(np.mean(segment ** 2))
                    
                    # Normalize energy to 0-1 range
                    normalized_energy = min(1.0, energy * 10)
                    
                    return True, normalized_energy
        
        return False, 0.0