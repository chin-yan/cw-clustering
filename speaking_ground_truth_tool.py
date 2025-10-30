#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Speaking Moment Ground Truth Annotation Tool with Auto Subtitle Alignment
Features: 
1. Automatic audio-subtitle alignment before annotation
2. Chorus Mode - Multiple speakers at the same time
3. Support for cases where the speaker is not visible in the current frame
"""

import os
import cv2
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import subprocess
import tempfile

try:
    import pysrt
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysrt"])
    import pysrt

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not installed. Install with: pip install librosa")
    LIBROSA_AVAILABLE = False

# Import your modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import feature_extraction
from enhanced_face_preprocessing import detect_foreground_faces_in_frame
import facenet.src.align.detect_face as detect_face



class AudioSubtitleAligner:
    """Automatic audio-subtitle alignment tool with VAD support"""
    
    def __init__(self, video_path, srt_path, vad_mode='webrtc', aggressiveness=3):
        """
        Initialize aligner
        
        Args:
            video_path: Path to video file
            srt_path: Path to SRT subtitle file
            vad_mode: 'energy', 'webrtc', or 'silero'
            aggressiveness: VAD aggressiveness level (0-3 for webrtc)
        """
        self.video_path = video_path
        self.srt_path = srt_path
        self.vad_mode = vad_mode
        self.aggressiveness = aggressiveness
        
        self.audio_data = None
        self.sample_rate = None
        self.energy_profile = None
        self.speech_segments = []
        
        # Initialize VAD if needed
        self.vad = None
        self.vad_model = None
        
        if vad_mode == 'webrtc':
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(aggressiveness)
                print(f"Using WebRTC VAD (aggressiveness: {aggressiveness})")
            except ImportError:
                print("Warning: webrtcvad not available, falling back to energy-based detection")
                self.vad_mode = 'energy'
        
        elif vad_mode == 'silero':
            try:
                import torch
                self.vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.get_speech_timestamps = utils[0]
                print("Using Silero VAD (deep learning model)")
            except Exception as e:
                print(f"Warning: Silero VAD not available ({e}), falling back to energy-based detection")
                self.vad_mode = 'energy'
        
        elif vad_mode == 'energy':
            print("Using energy-based detection")
        
    def extract_audio(self):
        """Extract audio from video"""
        if not LIBROSA_AVAILABLE:
            print("Error: librosa is required for audio alignment")
            return False
            
        print("\n[1/4] Extracting audio from video...")
        
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_audio_path = tmp.name
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', self.video_path, '-vn', 
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', temp_audio_path, '-loglevel', 'error'
            ]
            
            result = subprocess.run(cmd, capture_output=True, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                print("Error: Failed to extract audio with ffmpeg")
                return False
            
            # Load audio with librosa
            self.audio_data, self.sample_rate = librosa.load(temp_audio_path, sr=16000)
            
            # Clean up
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            duration = len(self.audio_data) / self.sample_rate
            print(f"   ✓ Audio extracted: {duration:.2f}s at {self.sample_rate}Hz")
            return True
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False
    
    def detect_speech_with_webrtc(self):
        """Detect speech segments using WebRTC VAD"""
        print("\n[2/4] Detecting speech with WebRTC VAD...")
        
        # WebRTC VAD requires 10, 20, or 30 ms frames
        frame_duration_ms = 30
        frame_length = int(self.sample_rate * frame_duration_ms / 1000)
        
        # Convert float audio to int16
        audio_int16 = (self.audio_data * 32767).astype(np.int16)
        
        speech_frames = []
        num_frames = len(audio_int16) // frame_length
        
        for i in tqdm(range(num_frames), desc="   Processing frames"):
            start = i * frame_length
            end = start + frame_length
            frame = audio_int16[start:end].tobytes()
            
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                timestamp = start / self.sample_rate
                
                speech_frames.append({
                    'timestamp': timestamp,
                    'is_speech': is_speech
                })
            except:
                continue
        
        # Merge consecutive speech frames into segments
        self.speech_segments = self._merge_speech_frames(speech_frames, min_duration=0.3)
        
        print(f"   ✓ Detected {len(self.speech_segments)} speech segments")
        return True
    
    def detect_speech_with_silero(self):
        """Detect speech segments using Silero VAD"""
        print("\n[2/4] Detecting speech with Silero VAD...")
        
        import torch
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(self.audio_data).float()
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            return_seconds=True
        )
        
        # Convert to our format
        self.speech_segments = []
        for segment in speech_timestamps:
            self.speech_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['end'] - segment['start']
            })
        
        print(f"   ✓ Detected {len(self.speech_segments)} speech segments")
        return True
    
    def _merge_speech_frames(self, speech_frames, min_duration=0.3):
        """Merge consecutive speech frames into segments"""
        if not speech_frames:
            return []
        
        segments = []
        current_segment = None
        
        for frame in speech_frames:
            if frame['is_speech']:
                if current_segment is None:
                    current_segment = {
                        'start': frame['timestamp'],
                        'end': frame['timestamp']
                    }
                else:
                    current_segment['end'] = frame['timestamp']
            else:
                if current_segment is not None:
                    duration = current_segment['end'] - current_segment['start']
                    if duration >= min_duration:
                        current_segment['duration'] = duration
                        segments.append(current_segment)
                    current_segment = None
        
        if current_segment is not None:
            duration = current_segment['end'] - current_segment['start']
            if duration >= min_duration:
                current_segment['duration'] = duration
                segments.append(current_segment)
        
        return segments
    
    def compute_energy_profile(self, window_size=0.02, hop_size=0.01):
        """Compute energy profile of audio signal (legacy energy-based method)"""
        if self.audio_data is None:
            return False
        
        print("\n[2/4] Computing audio energy profile...")
        
        # Frame-based energy calculation
        frame_length = int(window_size * self.sample_rate)
        hop_length = int(hop_size * self.sample_rate)
        
        # Calculate RMS energy for each frame
        energies = []
        timestamps = []
        
        for i in range(0, len(self.audio_data) - frame_length, hop_length):
            frame = self.audio_data[i:i+frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
            timestamps.append(i / self.sample_rate)
        
        self.energy_profile = {
            'energies': np.array(energies),
            'timestamps': np.array(timestamps),
            'threshold': None
        }
        
        # Compute adaptive threshold
        median_energy = np.median(self.energy_profile['energies'])
        mad = np.median(np.abs(self.energy_profile['energies'] - median_energy))
        self.energy_profile['threshold'] = median_energy + 2.5 * mad
        
        print(f"   ✓ Energy profile computed: {len(energies)} frames")
        return True
    
    def detect_speech_onsets(self):
        """Detect speech onset times from energy profile (legacy method)"""
        if self.energy_profile is None:
            return []
        
        energies = self.energy_profile['energies']
        timestamps = self.energy_profile['timestamps']
        threshold = self.energy_profile['threshold']
        
        # Find speech onsets (energy crosses threshold)
        is_speech = energies > threshold
        onsets = []
        
        in_speech = False
        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                onsets.append(timestamps[i])
                in_speech = True
            elif not is_speech[i] and in_speech:
                in_speech = False
        
        print(f"   ✓ Detected {len(onsets)} speech onsets")
        return onsets
    
    def load_subtitle_times(self):
        """Load subtitle start times"""
        try:
            # Handle BOM
            with open(self.srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', 
                                            delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Load subtitles
            subs = pysrt.open(tmp_path, encoding='utf-8')
            
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            # Extract start times
            subtitle_times = []
            for sub in subs:
                start_sec = (sub.start.hours * 3600 + 
                             sub.start.minutes * 60 + 
                             sub.start.seconds + 
                             sub.start.milliseconds / 1000.0)
                subtitle_times.append(start_sec)
            
            print(f"\n[3/4] Loaded {len(subtitle_times)} subtitles")
            return subtitle_times
            
        except Exception as e:
            print(f"Error loading subtitles: {e}")
            return []
    
    def calculate_optimal_offset(self, subtitle_times, speech_onsets, 
                                 max_offset=5.0, step=0.02):
        """Calculate optimal time offset by matching subtitle times to speech onsets"""
        if not subtitle_times or not speech_onsets:
            return 0.0
        
        print(f"\n[4/4] Calculating optimal time offset...")
        print(f"   Testing range: [{-max_offset:.2f}, {max_offset:.2f}]s (step: {step:.3f}s)")
        
        # Try different offsets
        best_offset = 0.0
        best_score = -1
        
        offset_range = np.arange(-max_offset, max_offset + step, step)
        
        for offset in tqdm(offset_range, desc="   Testing offsets"):
            # Apply offset to subtitle times
            adjusted_times = [t + offset for t in subtitle_times]
            
            # Calculate matching score
            score = 0
            tolerance = 0.5  # 500ms tolerance
            
            for sub_time in adjusted_times:
                # Find nearest speech onset
                distances = [abs(sub_time - onset) for onset in speech_onsets]
                min_distance = min(distances) if distances else float('inf')
                
                # Score based on distance (closer is better)
                if min_distance < tolerance:
                    score += 1.0 - (min_distance / tolerance)
            
            # Normalize score
            score = score / len(subtitle_times) if subtitle_times else 0
            
            if score > best_score:
                best_score = score
                best_offset = offset
        
        match_percentage = best_score * 100
        
        print(f"\n   ✓ Optimal offset: {best_offset:.3f} seconds")
        print(f"   Match quality: {match_percentage:.1f}%")
        
        return best_offset
    
    def run_alignment(self):
        """Run the complete alignment process"""
        print("\n" + "="*70)
        print(f"Audio-Subtitle Alignment (Mode: {self.vad_mode.upper()})")
        print("="*70)
        
        # Step 1: Extract audio
        if not self.extract_audio():
            return 0.0
        
        # Step 2: Detect speech
        if self.vad_mode == 'webrtc' and self.vad is not None:
            if not self.detect_speech_with_webrtc():
                return 0.0
            speech_onsets = [seg['start'] for seg in self.speech_segments]
        
        elif self.vad_mode == 'silero' and self.vad_model is not None:
            if not self.detect_speech_with_silero():
                return 0.0
            speech_onsets = [seg['start'] for seg in self.speech_segments]
        
        else:  # energy-based fallback
            if not self.compute_energy_profile():
                return 0.0
            speech_onsets = self.detect_speech_onsets()
        
        # Step 3: Load subtitles
        subtitle_times = self.load_subtitle_times()
        if not subtitle_times:
            return 0.0
        
        # Step 4: Calculate offset
        offset = self.calculate_optimal_offset(subtitle_times, speech_onsets)
        
        print("\n" + "="*70)
        print(f"✓ Alignment Complete!")
        print(f"Recommended time offset: {offset:.3f} seconds")
        print("="*70)
        
        return offset

        print("\n" + "="*70)
        print("AUTOMATIC AUDIO-SUBTITLE ALIGNMENT")
        print("="*70)
        
        # Step 1: Extract audio
        if not self.extract_audio():
            return 0.0
        
        # Step 2: Compute energy profile
        if not self.compute_energy_profile():
            return 0.0
        
        # Step 3: Detect speech onsets
        speech_onsets = self.detect_speech_onsets()
        if not speech_onsets:
            print("Warning: No speech onsets detected")
            return 0.0
        
        # Step 4: Load subtitle times
        subtitle_times = self.load_subtitle_times()
        if not subtitle_times:
            print("Warning: No subtitle times loaded")
            return 0.0
        
        # Step 5: Calculate optimal offset
        optimal_offset = self.calculate_optimal_offset(subtitle_times, speech_onsets)
        
        # Step 6: Visualize alignment
        self.visualize_alignment(subtitle_times, speech_onsets, optimal_offset)
        
        return optimal_offset


class RealTimeFaceDetector:
    """Real-time Face Detection and Recognition"""
    
    def __init__(self, model_dir, centers_data_path):
        self.model_dir = model_dir
        self.centers_data_path = centers_data_path
        self.sess = None
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.centers = None
        self.center_paths = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        
    def initialize(self):
        """Initialize models"""
        print("Initializing face detection and recognition models...")
        
        # Load cluster centers
        with open(self.centers_data_path, 'rb') as f:
            centers_data = pickle.load(f)
        self.centers, self.center_paths = centers_data['cluster_centers']
        print(f"Loaded {len(self.centers)} character centers")
        
        # Initialize TensorFlow
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            
            # Create MTCNN detector
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
            
            # Load FaceNet model
            feature_extraction.load_model(self.sess, os.path.expanduser(self.model_dir))
            
            # Get tensors
            self.images_placeholder = self.graph.get_tensor_by_name("input:0")
            self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        
        print("Models initialized successfully")
    
    def detect_and_identify_faces(self, frame):
        """Detect and identify faces in a frame"""
        with self.graph.as_default():
            # Detect faces
            filtered_bboxes = detect_foreground_faces_in_frame(
                frame, self.pnet, self.rnet, self.onet,
                min_face_size=60,
                min_face_area_ratio=0.008,
                max_faces_per_frame=5
            )
            
            if not filtered_bboxes:
                return []
            
            # Identify each face
            faces = []
            for bbox in filtered_bboxes:
                x1, y1, x2, y2 = bbox[:4]
                
                # Extract face region
                bbox_size = max(x2 - x1, y2 - y1)
                margin = int(bbox_size * 0.2)
                
                x1_crop = max(0, x1 - margin)
                y1_crop = max(0, y1 - margin)
                x2_crop = min(frame.shape[1], x2 + margin)
                y2_crop = min(frame.shape[0], y2 + margin)
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = frame_rgb[y1_crop:y2_crop, x1_crop:x2_crop, :]
                
                if face.size == 0:
                    continue
                
                # Resize and pre-whiten
                face_resized = cv2.resize(face, (160, 160))
                import facenet.src.facenet as facenet
                face_prewhitened = facenet.prewhiten(face_resized)
                
                # Calculate encoding
                feed_dict = {
                    self.images_placeholder: face_prewhitened.reshape(-1, 160, 160, 3),
                    self.phase_train_placeholder: False
                }
                encoding = self.sess.run(self.embeddings, feed_dict=feed_dict)[0]
                
                # Compare with centers
                similarities = np.dot(self.centers, encoding)
                best_idx = np.argmax(similarities)
                best_sim = similarities[best_idx]
                
                # Only accept results above threshold
                character_id = best_idx if best_sim > 0.55 else -1
                
                faces.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'character_id': character_id,
                    'similarity': float(best_sim),
                    'encoding': encoding
                })
            
            return faces
    
    def cleanup(self):
        """Clean up resources"""
        if self.sess:
            self.sess.close()


class ImprovedGroundTruthTool:
    """Enhanced Ground Truth Annotation Tool with Auto-Alignment"""
    
    def __init__(self, video_path, srt_path, model_dir, centers_data_path, 
                 output_dir, character_names=None, time_offset=0.0, 
                 auto_align=False, vad_mode='webrtc', vad_aggressiveness=3):
        self.video_path = video_path
        self.srt_path = srt_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.character_names = character_names or {}
        self.time_offset = time_offset
        self.auto_align = auto_align
        self.vad_mode = vad_mode
        self.vad_aggressiveness = vad_aggressiveness
        
        # Perform auto-alignment if requested
        if self.auto_align and LIBROSA_AVAILABLE:
            self._perform_auto_alignment()
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"\nVideo Information:")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Total frames: {self.total_frames:,}")
        print(f"   Duration: {self.duration:.2f} seconds")
        print(f"   Final time offset: {self.time_offset:.3f} seconds")
        
        # Load subtitles and expand annotation points
        self.annotation_points = self._load_and_expand_subtitles()
        
        # Initialize face detector
        self.detector = RealTimeFaceDetector(model_dir, centers_data_path)
        self.detector.initialize()
        
        # Annotation data
        self.annotations = {}
        self.current_idx = 0
        
        # Cache
        self.frame_cache = {}
    
    def _perform_auto_alignment(self):
        """Perform automatic audio-subtitle alignment"""
        print("\n" + "="*70)
        print("STARTING AUTO-ALIGNMENT PROCESS")
        print("="*70)
        
        aligner = AudioSubtitleAligner(self.video_path, self.srt_path)
        calculated_offset = aligner.run_alignment()
        
        if calculated_offset != 0.0:
            print(f"\nCalculated time offset: {calculated_offset:.3f} seconds")
            
            # Ask user for confirmation
            response = input(f"\nApply this offset? (y/n/custom): ").strip().lower()
            
            if response == 'y':
                self.time_offset = calculated_offset
                print(f"Applied offset: {self.time_offset:.3f} seconds")
            elif response == 'custom':
                try:
                    custom_offset = float(input("Enter custom offset (seconds): ").strip())
                    self.time_offset = custom_offset
                    print(f"Applied custom offset: {self.time_offset:.3f} seconds")
                except:
                    print("Invalid input. Using calculated offset.")
                    self.time_offset = calculated_offset
            else:
                print("Offset not applied. Using original subtitle times.")
                self.time_offset = 0.0
        else:
            print("\nAuto-alignment could not determine an offset.")
            print("Proceeding with original subtitle times.")
    
    def _load_and_expand_subtitles(self):
        """Load subtitles and expand into annotation points"""
        print(f"\nLoading subtitles: {self.srt_path}")
        
        try:
            with open(self.srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', 
                                            delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            subs = pysrt.open(tmp_path, encoding='utf-8')
            
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Falling back to standard loading method")
            subs = pysrt.open(self.srt_path, encoding='utf-8')
        
        annotation_points = []
        
        for sub in subs:
            start_sec = self._time_to_seconds(sub.start) + self.time_offset
            end_sec = self._time_to_seconds(sub.end) + self.time_offset
            
            text = sub.text_without_tags.strip()
            subtitle_id = int(str(sub.index).strip().lstrip('\ufeff'))
            
            annotation_points.append({
                'subtitle_id': subtitle_id,
                'position': 'start',
                'timestamp': start_sec,
                'text': text,
                'duration': end_sec - start_sec
            })
        
        print(f"Loaded {len(subs)} subtitles, expanded to {len(annotation_points)} annotation points")
        return annotation_points
    
    def _time_to_seconds(self, time_obj):
        """Convert time object to seconds"""
        return (time_obj.hours * 3600 + 
                time_obj.minutes * 60 + 
                time_obj.seconds + 
                time_obj.milliseconds / 1000.0)
    
    def get_character_name(self, char_id):
        """Get character name"""
        if char_id == -1:
            return "Unknown"
        return self.character_names.get(char_id, f"ID_{char_id}")
    
    def _get_frame_at_time(self, timestamp):
        """Get frame at specific timestamp"""
        frame_idx = int(timestamp * self.fps)
        
        # Check cache
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]['frame'], self.frame_cache[frame_idx]['faces']
        
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return None, []
        
        # Detect and identify faces
        faces = self.detector.detect_and_identify_faces(frame)
        
        # Cache result
        if len(self.frame_cache) > 100:
            oldest_key = min(self.frame_cache.keys())
            del self.frame_cache[oldest_key]
        
        self.frame_cache[frame_idx] = {'frame': frame.copy(), 'faces': faces}
        
        return frame, faces
    
    def _prepare_display_frame(self, frame, point, faces):
        """Prepare frame for display"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw all detected faces
        for idx, face in enumerate(faces, 1):
            char_id = face['character_id']
            similarity = face['similarity']
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            
            color = (0, 150, 255)  # Orange
            thickness = 3
            label = f"#{idx} {self.get_character_name(char_id)}"
            
            point_key = f"{point['subtitle_id']}_{point['position']}"
            if point_key in self.annotations:
                ann = self.annotations[point_key]
                if char_id in ann.get('all_faces', []) or char_id in ann.get('speaker_ids', []):
                    color = (0, 255, 0)  # Green
                    thickness = 4
                    label = f"#{idx} OK {self.get_character_name(char_id)}"
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            cv2.circle(display_frame, (x1 + 20, y1 + 20), 18, color, -1)
            cv2.putText(display_frame, str(idx), (x1 + 13, y1 + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), 
                          (x1 + text_size[0] + 10, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            sim_text = f"Sim: {similarity:.2f}"
            cv2.putText(display_frame, sim_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if not faces:
            warning = "WARNING: No faces detected in this frame"
            cv2.putText(display_frame, warning, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Create info panel
        panel_height = 300
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        point_key = f"{point['subtitle_id']}_{point['position']}"
        annotation_status = "Not annotated"
        if point_key in self.annotations:
            ann = self.annotations[point_key]
            status = ann['status']
            
            if status == 'chorus':
                speaker_ids = ann.get('speaker_ids', [])
                speaker_names = [self.get_character_name(sid) for sid in speaker_ids]
                annotation_status = f"CHORUS: {', '.join(speaker_names)}"
            elif status == 'all_correct':
                annotation_status = f"OK All correct (Speaker: ID {ann.get('speaker_id', -1)})"
            elif status == 'partially_corrected':
                annotation_status = f"WARNING Corrected (Speaker: ID {ann.get('speaker_id', -1)})"
            elif status == 'unknown':
                annotation_status = "? Unknown"
            elif status == 'narration':
                annotation_status = "NARRATION (Voiceover/Narrator)"
            elif status == 'speaker_not_visible':
                annotation_status = f"WARNING Speaker not visible (Speaker ID: {ann.get('speaker_id', -1)})"
        
        info_lines = [
            f"Progress: {self.current_idx + 1}/{len(self.annotation_points)}",
            f"Subtitle #{point['subtitle_id']} - Position: {point['position']}",
            f"Time: {point['timestamp']:.2f}s | Text: {point['text'][:40]}",
            f"Detected faces: {len(faces)}",
            f"Current status: {annotation_status}",
            "",
            "Instructions:",
            "  c = All Correct | m = Manual Annotation | h = Chorus Mode",
            "  x = Speaker Not Visible | v = Narration",
            "  u = Unknown | s = Skip | n = Next | p = Previous | q = Save & Quit"
        ]
        
        y_offset = 22
        for line in info_lines:
            cv2.putText(panel, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 22
        
        result = np.vstack([display_frame, panel])
        return result
    
    def _handle_key(self, key, point, faces):
        """Handle key presses"""
        try:
            subtitle_id = int(str(point['subtitle_id']).strip().lstrip('\ufeff'))
        except:
            subtitle_id = 0
        
        point_key = f"{subtitle_id}_{point['position']}"
        
        if key == ord('q'):
            return 'quit'
        elif key == 27:  # ESC
            return 'quit_nosave'
        elif key == ord('n'):
            return 'next'
        elif key == ord('p'):
            return 'prev'
        elif key == ord('s'):
            print("-> Skipped")
            return 'next'
        elif key == ord('u'):
            self.annotations[point_key] = {
                'subtitle_id': subtitle_id,
                'position': str(point['position']),
                'timestamp': float(point['timestamp']),
                'text': str(point['text']),
                'speaker_id': -1,
                'speaker_ids': [],
                'all_faces': [],
                'status': 'unknown'
            }
            print("? Marked as unknown")
            return 'next'
        
        elif key == ord('h'):
            # Chorus Mode
            print(f"\n*** CHORUS MODE ***")
            
            if not faces:
                print("ERROR: No faces to annotate")
                return 'stay'
            
            print(f"\nFaces in frame:")
            for idx, face in enumerate(faces, 1):
                char_id = face['character_id']
                print(f"  #{idx}: {self.get_character_name(char_id)} (ID: {char_id})")
            
            try:
                speaker_input = input(f"Enter all speaker numbers (e.g., 1,2,3): ").strip()
                
                if speaker_input == "":
                    return 'stay'
                
                speaker_numbers = [int(x.strip()) for x in speaker_input.split(',')]
                
                if not all(1 <= num <= len(faces) for num in speaker_numbers):
                    print(f"ERROR: Numbers must be between 1 and {len(faces)}")
                    return 'stay'
                
                speaker_ids = [int(faces[num - 1]['character_id']) for num in speaker_numbers]
                all_face_ids = [int(face['character_id']) for face in faces]
                
                speaker_names = [self.get_character_name(sid) for sid in speaker_ids]
                print(f"\nSpeakers: {', '.join(speaker_names)}")
                
                confirm = input(f"Confirm chorus annotation? (y/n): ").strip().lower()
                
                if confirm != 'y':
                    print("Annotation cancelled.")
                    return 'stay'
                
                self.annotations[point_key] = {
                    'subtitle_id': subtitle_id,
                    'position': str(point['position']),
                    'timestamp': float(point['timestamp']),
                    'text': str(point['text']),
                    'speaker_id': speaker_ids[0] if speaker_ids else -1,
                    'speaker_ids': speaker_ids,
                    'all_faces': all_face_ids,
                    'status': 'chorus'
                }
                print(f"OK Chorus annotation complete")
                return 'next'
                
            except ValueError:
                print("ERROR: Invalid input")
                return 'stay'
            except Exception as e:
                print(f"ERROR: {e}")
                return 'stay'
        
        elif key == ord('x'):
            # Speaker not visible
            print(f"\nWARNING Speaker Not Visible Mode")
            
            all_face_ids = [int(face['character_id']) for face in faces]
            
            if faces:
                print(f"\nFaces currently visible:")
                for idx, face in enumerate(faces, 1):
                    print(f"  #{idx}: {self.get_character_name(face['character_id'])}")
            
            try:
                speaker_input = input(f"Enter speaker ID: ").strip()
                
                if speaker_input == "":
                    return 'stay'
                
                speaker_id = int(speaker_input)
                
                confirm = input(f"Confirm: Speaker is ID {speaker_id}, not in frame? (y/n): ").strip().lower()
                
                if confirm != 'y':
                    return 'stay'
                
                self.annotations[point_key] = {
                    'subtitle_id': subtitle_id,
                    'position': str(point['position']),
                    'timestamp': float(point['timestamp']),
                    'text': str(point['text']),
                    'speaker_id': speaker_id,
                    'speaker_ids': [speaker_id],
                    'all_faces': all_face_ids,
                    'status': 'speaker_not_visible'
                }
                print(f"OK Annotated: Speaker ID {speaker_id} not visible")
                return 'next'
                
            except ValueError:
                print("ERROR: Invalid input")
                return 'stay'
            except Exception as e:
                print(f"ERROR: {e}")
                return 'stay'
            
        elif key == ord('v'):
            # Narration mode
            print(f"\nNARRATION MODE")
            
            all_face_ids = [int(face['character_id']) for face in faces]
            
            confirm = input(f"Confirm this is narration/voiceover? (y/n): ").strip().lower()
            
            if confirm != 'y':
                return 'stay'
            
            self.annotations[point_key] = {
                'subtitle_id': subtitle_id,
                'position': str(point['position']),
                'timestamp': float(point['timestamp']),
                'text': str(point['text']),
                'speaker_id': -2,
                'speaker_ids': [-2],
                'all_faces': all_face_ids,
                'status': 'narration'
            }
            print(f"OK Annotated as narration")
            return 'next'
        
        elif key == ord('c'):
            # All correct
            if not faces:
                print("ERROR: No faces to annotate")
                return 'stay'
            
            print(f"\nOK Confirming all face IDs are correct")
            print("Faces in frame:")
            for idx, face in enumerate(faces, 1):
                print(f"  #{idx}: {self.get_character_name(face['character_id'])}")
            
            try:
                speaker_choice = input(f"\nSelect speaker number (1-{len(faces)}): ").strip()
                
                if speaker_choice == "":
                    return 'stay'
                
                speaker_num = int(speaker_choice)
                if 1 <= speaker_num <= len(faces):
                    speaker_id = int(faces[speaker_num - 1]['character_id'])
                    all_face_ids = [int(face['character_id']) for face in faces]
                    
                    self.annotations[point_key] = {
                        'subtitle_id': subtitle_id,
                        'position': str(point['position']),
                        'timestamp': float(point['timestamp']),
                        'text': str(point['text']),
                        'speaker_id': speaker_id,
                        'speaker_ids': [speaker_id],
                        'all_faces': all_face_ids,
                        'status': 'all_correct'
                    }
                    print(f"OK Annotated: Speaker is ID {speaker_id}")
                    return 'next'
                else:
                    print("ERROR: Invalid number")
                    return 'stay'
            except:
                print("ERROR: Invalid input")
                return 'stay'
        
        elif key == ord('m'):
            # Manual annotation
            if not faces:
                print("ERROR: No faces to annotate")
                return 'stay'
            
            print(f"\nManual Annotation Mode")
            
            corrected_faces = []
            
            for idx, face in enumerate(faces, 1):
                detected_id = face['character_id']
                print(f"\nFace #{idx}:")
                print(f"  System ID: {self.get_character_name(detected_id)}")
                
                response = input(f"  Correct? (y/ID/n): ").strip().lower()
                
                if response == 'y':
                    corrected_faces.append(detected_id)
                elif response == 'n':
                    continue
                else:
                    try:
                        correct_id = int(response)
                        corrected_faces.append(correct_id)
                        print(f"  OK Corrected to ID {correct_id}")
                    except:
                        continue
            
            corrected_idx = 0
            for idx, face in enumerate(faces):
                if corrected_idx < len(corrected_faces):
                    faces[idx]['character_id'] = corrected_faces[corrected_idx]
                    corrected_idx += 1
            
            if not corrected_faces:
                print("ERROR: No faces were annotated")
                return 'stay'
            
            try:
                speaker_input = input(f"Enter speaker ID: ").strip()
                
                if speaker_input == "":
                    speaker_id = corrected_faces[0]
                else:
                    speaker_id = int(speaker_input)
                
                self.annotations[point_key] = {
                    'subtitle_id': subtitle_id,
                    'position': str(point['position']),
                    'timestamp': float(point['timestamp']),
                    'text': str(point['text']),
                    'speaker_id': speaker_id,
                    'speaker_ids': [speaker_id],
                    'all_faces': corrected_faces,
                    'status': 'partially_corrected'
                }
                print(f"OK Annotation complete")
                return 'next'
                
            except:
                print("ERROR: Invalid input")
                return 'stay'
        
        return 'stay'
    
    def run_annotation(self):
        """Run annotation process"""
        print("\n" + "="*70)
        print("Enhanced Ground Truth Annotation Tool")
        print("="*70)
        print("\nAnnotation Workflow:")
        print("  c = All Correct | m = Manual | h = Chorus")
        print("  x = Not Visible | v = Narration | u = Unknown")
        print("  s = Skip | n = Next | p = Previous | q = Save & Quit")
        print("  [ESC] = Quit WITHOUT Saving") # Remind user of ESC functionality
        print("="*70)

        input("\nPress Enter to start annotating...")
    
        # We no longer use 'annotation_completed', but a clearer flag
        aborted_no_save = False
    
        while self.current_idx < len(self.annotation_points):
            point = self.annotation_points[self.current_idx]

            frame, faces = self._get_frame_at_time(point['timestamp'])

            if frame is None:
                print(f"WARNING: Could not read frame, skipping")
                self.current_idx += 1
                continue

            display_frame = self._prepare_display_frame(frame, point, faces)
            cv2.imshow('Ground Truth Annotation Tool', display_frame)

            key = cv2.waitKey(0) & 0xFF
            action = self._handle_key(key, point, faces)

            if action == 'quit':
                # Press 'q' (save and exit), we just need to break the loop
                break
            elif action == 'quit_nosave':
                # Press 'ESC' (don't save and exit)
                print("\nABORTED: Exiting without saving")
                aborted_no_save = True # Set flag
                break
            elif action == 'next':
                self.current_idx += 1
                # Check if we've reached the end
                if self.current_idx >= len(self.annotation_points):
                        # Finished, we just need to break the loop
                    print("\n" + "="*70)
                    print("REACHED END OF ANNOTATION POINTS")
                    print("="*70)
                    break
            elif action == 'prev':
                self.current_idx = max(0, self.current_idx - 1)

        # --- Unified handling after the loop ---
        cv2.destroyAllWindows()
        self.cap.release()
        self.detector.cleanup()

        # Only don't save if 'aborted_no_save' (ESC) is True
        if not aborted_no_save:
            print("\nAnnotation session complete. Saving annotations...")
            self._save_annotations()
            self._print_stats()
        else:
            print("\nAnnotation session ended without saving.")
    
    def _save_annotations(self):
        """Save annotations"""
        output_file = self.output_dir / 'speaking_moments_ground_truth_enhanced.json'
        
        serializable_annotations = {}
        for key, value in self.annotations.items():
            speaker_ids = value.get('speaker_ids', [])
            all_faces = value.get('all_faces', [])    

            serializable_annotations[key] = {
                'subtitle_id': int(value.get('subtitle_id', 0)),
                'position': value.get('position', ''),
                'timestamp': float(value.get('timestamp', 0.0)),
                'text': value.get('text', ''),
                'speaker_id': int(value.get('speaker_id', -1)),
                'speaker_ids': [int(x) for x in speaker_ids],
                'all_faces': [int(x) for x in all_faces],
                'status': value.get('status', 'unknown'),
                'note': value.get('note', '')
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved to: {output_file}")
        print(f"   Annotation count: {len(self.annotations)}")
    
    def _print_stats(self):
        """Print statistics"""
        if not self.annotations:
            return
        
        stats = defaultdict(int)
        for ann in self.annotations.values():
            stats[ann['status']] += 1
        
        print("\n" + "="*70)
        print("Annotation Statistics")
        print("="*70)
        print(f"Total annotation points: {len(self.annotation_points)}")
        print(f"Annotated: {len(self.annotations)}")
        print("\nBreakdown by status:")
        
        status_labels = {
            'all_correct': 'All Correct',
            'partially_corrected': 'Partially Corrected',
            'chorus': 'Chorus Mode',
            'speaker_not_visible': 'Speaker Not Visible',
            'narration': 'Narration',
            'unknown': 'Unknown'
        }
        
        for status, count in stats.items():
            label = status_labels.get(status, status)
            print(f"  {label}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Speaking Moment Ground Truth Tool with Auto-Alignment'
    )
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--srt', required=True, help='Subtitle file path')
    parser.add_argument('--model-dir', required=True, help='FaceNet model directory')
    parser.add_argument('--centers-data', required=True, help='Cluster centers data file')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--character-names', help='Character names JSON file path')
    parser.add_argument('--time-offset', type=float, default=0.0, 
                        help='Manual time offset in seconds')
    parser.add_argument('--auto-align', action='store_true',
                        help='Enable automatic audio-subtitle alignment')
    parser.add_argument('--vad-mode', choices=['energy', 'webrtc', 'silero'],
                        default='webrtc',
                        help='VAD method for auto-alignment (default: webrtc)')
    parser.add_argument('--vad-aggressiveness', type=int, default=3,
                        choices=[0, 1, 2, 3],
                        help='WebRTC VAD aggressiveness level 0-3 (default: 3)')
    
    args = parser.parse_args()
    
    # Load character names
    character_names = {}
    if args.character_names:
        try:
            with open(args.character_names, 'r', encoding='utf-8') as f:
                names_data = json.load(f)
                character_names = {int(k): v for k, v in names_data.items()}
        except Exception as e:
            print(f"WARNING: Could not load character names: {e}")
    
    # Create tool instance
    tool = ImprovedGroundTruthTool(
        video_path=args.video,
        srt_path=args.srt,
        model_dir=args.model_dir,
        centers_data_path=args.centers_data,
        output_dir=args.output,
        character_names=character_names,
        time_offset=args.time_offset,
        auto_align=args.auto_align
    )
    
    # Run annotation
    tool.run_annotation()


if __name__ == '__main__':
    exit(main())