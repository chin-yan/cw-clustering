# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
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
import argparse
import shutil
from collections import defaultdict, Counter

# InsightFace imports
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Suppress warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

# Note: feature_extraction import removed as it was not provided in context, 
# assuming functions are self-contained or standard imports are sufficient.


def generate_distinct_colors(n):
    """
    Generate N visually distinct colors using HSV color space
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
        self.colors = generate_distinct_colors(n_characters)
        self.unknown_color = (128, 128, 128)  # Gray for unknown
        self.default_color = (255, 255, 255)  # White for narration/not visible
    
    def get_color(self, character_id):
        if character_id == -1:
            return self.unknown_color
        elif character_id >= 0 and character_id < len(self.colors):
            return self.colors[character_id]
        else:
            return self.unknown_color


class BBoxSmoother:
    """
    Smooth bounding box positions using Exponential Moving Average (EMA)
    """
    
    def __init__(self, initial_bbox, alpha=0.3):
        if initial_bbox is None:
            self.smoothed_bbox = None
        else:
            self.smoothed_bbox = [float(x) for x in initial_bbox]
        self.alpha = alpha
        self.initialized = (initial_bbox is not None)
    
    def update(self, new_bbox):
        if new_bbox is None:
            if self.smoothed_bbox is None:
                return None
            return tuple(int(round(x)) for x in self.smoothed_bbox)
        
        if not self.initialized:
            self.smoothed_bbox = [float(x) for x in new_bbox]
            self.initialized = True
            return tuple(int(round(x)) for x in self.smoothed_bbox)
        
        for i in range(4):
            self.smoothed_bbox[i] = (
                self.alpha * new_bbox[i] + 
                (1 - self.alpha) * self.smoothed_bbox[i]
            )
        
        return tuple(int(round(x)) for x in self.smoothed_bbox)


def load_centers_data(centers_data_path):
    print("Loading cluster center data...")
    with open(centers_data_path, 'rb') as f:
        centers_data = pickle.load(f)
    
    if 'cluster_centers' not in centers_data:
        raise ValueError("Missing cluster center information in the data")
    
    centers, center_paths = centers_data['cluster_centers']
    print(f"Successfully loaded {len(centers)} cluster centers")
    return centers, center_paths, centers_data


def match_face_with_centers(face_encoding, centers, threshold=0.5):
    if len(centers) == 0:
        return -1, 0
    
    similarities = np.dot(centers, face_encoding)
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]
    
    if best_similarity > threshold:
        return best_index, best_similarity
    else:
        return -1, 0


def init_facial_landmark_detector():
    print("Initializing facial landmark detector (Dlib)...")
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print(f"Facial landmark model not found at {model_path}")
        raise FileNotFoundError(f"Missing required model file: {model_path}")
    return dlib.shape_predictor(model_path)


def init_insightface_app():
    print("Initializing InsightFace App...")
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], 
                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace loaded on GPU")
    except Exception as e:
        print(f"InsightFace loading on GPU failed, falling back to CPU: {e}")
        app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def process_frame_faces(frame, app, landmark_predictor, rec_model, min_face_size=30):
    faces_data = []
    try:
        detected_faces = app.get(frame)
    except Exception:
        return []
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for face in detected_faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        if (y2 - y1) < min_face_size:
            continue
            
        try:
            # ArcFace Embedding
            aligned_face = face_align.norm_crop(frame, landmark=face.kps, image_size=112)
            embedding = rec_model.get_feat(aligned_face).flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
        except Exception:
            continue
            
        try:
            # Dlib Landmarks for MAR
            dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
            shape = landmark_predictor(gray_frame, dlib_rect)
            landmarks = []
            for i in range(68):
                landmarks.append((shape.part(i).x, shape.part(i).y))
            
            mouth_landmarks = landmarks[48:68]
            
            face_info = {
                'bbox': (x1, y1, x2, y2),
                'kps': face.kps,
                'landmarks': landmarks,
                'mouth_landmarks': mouth_landmarks,
                'encoding': embedding,
                'det_score': face.det_score
            }
            faces_data.append(face_info)
            
        except Exception:
            continue
            
    return faces_data


def calculate_mouth_aspect_ratio(mouth_landmarks):
    if not mouth_landmarks or len(mouth_landmarks) < 20:
        return 0.0
    
    mouth_landmarks = np.array(mouth_landmarks)
    
    dist_center = abs(mouth_landmarks[9][1] - mouth_landmarks[3][1])
    dist_left = abs(mouth_landmarks[10][1] - mouth_landmarks[2][1])
    dist_right = abs(mouth_landmarks[8][1] - mouth_landmarks[4][1])
    mouth_height = (dist_center + dist_left + dist_right) / 3.0

    mouth_width = abs(mouth_landmarks[6][0] - mouth_landmarks[0][0])
    
    if mouth_width < 1:
        return 0.0
    
    return mouth_height / mouth_width


def detect_speaking_face(prev_faces, curr_faces, threshold=0.03):
    if not prev_faces or not curr_faces:
        return -1
    
    max_mar_change = 0
    speaking_idx = -1
    
    for curr_idx, curr_face in enumerate(curr_faces):
        curr_encoding = curr_face.get('encoding')
        if curr_encoding is None:
            continue
        
        best_match_idx = -1
        best_match_sim = 0
        
        for prev_idx, prev_face in enumerate(prev_faces):
            prev_encoding = prev_face.get('encoding')
            if prev_encoding is None:
                continue
            
            similarity = np.dot(curr_encoding, prev_encoding)
            if similarity > best_match_sim:
                best_match_sim = similarity
                best_match_idx = prev_idx
        
        if best_match_idx >= 0 and best_match_sim > 0.6:
            prev_mouth = prev_faces[best_match_idx].get('mouth_landmarks')
            curr_mouth = curr_face.get('mouth_landmarks')
            
            if not prev_mouth or not curr_mouth:
                continue
            
            prev_mar = calculate_mouth_aspect_ratio(prev_mouth)
            curr_mar = calculate_mouth_aspect_ratio(curr_mouth)
            mar_change = abs(curr_mar - prev_mar)
            
            if mar_change > max_mar_change:
                max_mar_change = mar_change
                speaking_idx = curr_idx
    
    if max_mar_change > threshold:
        return speaking_idx
    else:
        return -1


def synchronize_subtitles(video_path, srt_path):
    """
    Synchronize SRT with Video Audio using ffsubsync.
    Returns the path to the synchronized SRT file.
    """
    print("\n" + "="*70)
    print("PHASE 0: Synchronizing subtitles with audio (ffsubsync)")
    print("="*70)

    # Check if ffsubsync is available
    if shutil.which('ffsubsync') is None:
        print("[WARNING] 'ffsubsync' tool not found!")
        print("To fix synchronization issues, please install it via: pip install ffsubsync")
        print("Proceeding with original subtitles (audio/text mismatch may occur).")
        return srt_path

    # Create output filename
    base_name = os.path.splitext(os.path.basename(srt_path))[0]
    output_dir = os.path.dirname(srt_path)
    if not output_dir:
        output_dir = "."
    synced_srt_path = os.path.join(output_dir, f"{base_name}_synced.srt")

    print(f"Aligning {srt_path} to audio in {video_path}...")
    
    try:
        # Construct command: ffsubsync video.mp4 -i input.srt -o output.srt
        cmd = [
            'ffsubsync', 
            video_path, 
            '-i', srt_path, 
            '-o', synced_srt_path
        ]
        
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[SUCCESS] Synchronized subtitles saved to: {synced_srt_path}")
            return synced_srt_path
        else:
            print(f"[ERROR] Synchronization failed. Error details:\n{result.stderr}")
            print("Falling back to original subtitles.")
            return srt_path
            
    except Exception as e:
        print(f"[ERROR] Exception during synchronization: {e}")
        return srt_path


def parse_subtitle_file(subtitle_path):
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
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame.shape[:2]
    
    wrapped_text = wrap_text(text, max_width=30)
    lines = wrapped_text.split('\n')
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_height = 28
    
    max_text_width = 0
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
    
    padding = 10
    spacing = 20
    subtitle_x = x1 + label_width + spacing
    subtitle_y = y1 - line_height
    
    if subtitle_x + max_text_width + padding * 2 > frame_width:
        subtitle_x = max(padding, x1 - max_text_width - spacing - padding * 2)
    
    subtitle_y = max(line_height + padding, subtitle_y)
    
    bg_x1 = subtitle_x - padding
    bg_y1 = subtitle_y - line_height - padding
    bg_x2 = subtitle_x + max_text_width + padding
    bg_y2 = subtitle_y + (len(lines) - 1) * line_height + padding
    
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(frame_width, bg_x2)
    bg_y2 = min(frame_height, bg_y2)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.rectangle(frame, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), color, 2)
    
    for i, line in enumerate(lines):
        line_y = subtitle_y + i * line_height
        cv2.putText(frame, line, (int(subtitle_x) + 1, int(line_y) + 1),
                   font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (int(subtitle_x), int(line_y)),
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_subtitle_at_bottom(frame, text, color=(255, 255, 255)):
    if not text:
        return frame
    
    wrapped_text = wrap_text(text, max_width=50)
    lines = wrapped_text.split('\n')
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_height = 35
    
    frame_height, frame_width = frame.shape[:2]
    
    max_text_width = 0
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
    
    start_y = frame_height - (line_height * len(lines)) - 40
    
    padding = 12
    bg_y1 = start_y - line_height + 5
    bg_y2 = start_y + (len(lines) - 1) * line_height + padding + 5
    bg_x1 = (frame_width - max_text_width) // 2 - padding
    bg_x2 = (frame_width + max_text_width) // 2 + padding
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    for i, line in enumerate(lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = (frame_width - text_width) // 2
        line_y = start_y + i * line_height
        
        cv2.putText(frame, line, (int(text_x) + 2, int(line_y) + 2),
                   font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (int(text_x), int(line_y)),
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_context_stack(frame, active_contexts):
    if not active_contexts:
        return frame
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_height = 30
    
    start_x = 100
    start_y = 80
    
    overlay = frame.copy()
    current_y = start_y
    
    for text in active_contexts:
        wrapped_text = wrap_text(text, max_width=35)
        lines = wrapped_text.split('\n')
        
        max_w = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_w = max(max_w, w)
            
        padding = 8
        bg_x1 = start_x - padding
        bg_y1 = current_y - line_height + padding
        bg_x2 = start_x + max_w + padding
        bg_y2 = current_y + (len(lines)-1) * line_height + padding * 2
        
        cv2.rectangle(overlay, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), (40, 40, 40), -1)
        next_block_y = bg_y2 + 10 
        
        for i, line in enumerate(lines):
            line_y = current_y + i * line_height
            cv2.putText(overlay, line, (int(start_x), int(line_y)),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        current_y = next_block_y

    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame


def determine_speaker_for_subtitle(subtitle, detection_results, fps):
    start_ms = subtitle.start.ordinal
    end_ms = subtitle.end.ordinal
    
    start_frame = round((start_ms / 1000.0) * fps)
    end_frame = round((end_ms / 1000.0) * fps)
    
    speaker_votes = Counter()
    speaker_bboxes = {}
    all_speaker_ids = []
    all_faces_data = []
    
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx in detection_results:
            result = detection_results[frame_idx]
            speaking_idx = result.get('speaking_idx', -1)
            faces = result.get('faces', [])
            
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
    
    unique_speaker_ids = list(set(all_speaker_ids))
    
    if speaker_votes:
        most_common = speaker_votes.most_common(1)[0]
        winner_id = most_common[0]
        return winner_id, speaker_bboxes.get(winner_id), unique_speaker_ids, all_faces_data
    
    return None, None, unique_speaker_ids, all_faces_data


def merge_audio_to_video(video_without_audio, original_video, output_video):
    print("\nMerging audio from original video...")
    try:
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
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        
        if result.returncode == 0:
            try:
                if os.path.exists(output_video):
                    os.remove(output_video)
                os.rename(temp_ffmpeg_output, output_video)
                print(f"Audio merged successfully!")
                return True
            except Exception as e:
                print(f"Error moving file: {e}")
                return False
        else:
            print(f"FFmpeg failed: {result.stderr}")
            if os.path.exists(temp_ffmpeg_output):
                try:
                    os.remove(temp_ffmpeg_output)
                except:
                    pass
            return False
            
    except Exception as e:
        print(f"Error merging audio: {e}")
        return False


def convert_to_json_serializable(obj):
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
    try:
        serializable_data = convert_to_json_serializable(json_data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] JSON annotation saved: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save JSON annotation: {e}")
        return False


def annotate_video_with_speaker_subtitles(input_video, output_video, centers_data_path, 
                                         subtitle_path, detection_interval=1,
                                         similarity_threshold=0.65, speaking_threshold=1,
                                         preserve_audio=True, smoothing_alpha=0.3,
                                         generate_json=True, subtitle_offset=0.0,
                                         force_sync=True):
    
    # PHASE 0: Synchronization (New Logic)
    final_subtitle_path = subtitle_path
    if force_sync:
        final_subtitle_path = synchronize_subtitles(input_video, subtitle_path)
    
    if speaking_threshold > 0.2:
        print(f"WARNING: speaking_threshold {speaking_threshold} is too high for MAR logic.")
        print(f"         Auto-adjusting to 0.05 for reliable detection.")
        speaking_threshold = 0.05
    else:
        print(f"Using speaking_threshold: {speaking_threshold} (MAR)")
        
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    color_manager = ColorManager(len(centers))
    
    # Load the (potentially synchronized) subtitles
    subtitles = parse_subtitle_file(final_subtitle_path)
    print(f"Loaded {len(subtitles)} subtitles from {final_subtitle_path}")
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    if preserve_audio:
        output_dir = os.path.dirname(output_video)
        output_name = os.path.basename(output_video)
        name_without_ext, ext = os.path.splitext(output_name)
        temp_video = os.path.join(output_dir, f'{name_without_ext}_temp_no_audio{ext}')
        final_output = output_video
    else:
        temp_video = output_video
        final_output = None
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    json_annotations = {}
    
    app = init_insightface_app()
    landmark_predictor = init_facial_landmark_detector()
    rec_model = app.models['recognition']
            
    # PHASE 1: Detect faces
    print("\n" + "="*70)
    print("PHASE 1: Detecting faces and identifying speakers (InsightFace)")
    print("="*70)
    
    detection_results = {}
    prev_faces = []
    
    pbar = tqdm(total=total_frames, desc="Detecting")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % detection_interval == 0:
            faces = process_frame_faces(frame, app, landmark_predictor, rec_model, min_face_size=30)
            
            for face in faces:
                match_idx, similarity = match_face_with_centers(
                    face['encoding'], centers, threshold=similarity_threshold
                )
                face['match_idx'] = match_idx
                face['similarity'] = similarity
            
            speaking_idx = -1
            if prev_faces:
                speaking_idx = detect_speaking_face(prev_faces, faces, threshold=speaking_threshold)
            
            detection_results[frame_count] = {
                'faces': faces,
                'speaking_idx': speaking_idx
            }
            
            prev_faces = faces
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # PHASE 2: Analyze Subtitles
    print("\n" + "="*70)
    print("PHASE 2: Analyzing subtitles (Dialogues vs Context Notes)")
    print("="*70)
    
    subtitle_speakers = {}
    context_subtitles = []
    
    for subtitle in tqdm(subtitles, desc="Processing subtitles"):
        
        duration = subtitle.end.ordinal - subtitle.start.ordinal
        # Use offset if needed (though sync handles most)
        start_timestamp = subtitle.start.ordinal / 1000.0 + subtitle_offset
        
        if duration < 1000:
            start_ms = subtitle.start.ordinal + int(subtitle_offset * 1000)
            end_ms = start_ms + 5000 
            
            context_subtitles.append({
                'text': subtitle.text_without_tags,
                'start_ms': start_ms,
                'end_ms': end_ms
            })
            
            if generate_json:
                start_key = f"{subtitle.index}_start"
                json_annotations[start_key] = {
                    'subtitle_id': int(subtitle.index),
                    'position': 'top_left_context',
                    'timestamp': round(float(start_timestamp), 3),
                    'text': subtitle.text_without_tags,
                    'speaker_id': None,
                    'status': 'context_note'
                }
            continue
            
        speaker_id, speaker_bbox, speaker_ids, all_faces = determine_speaker_for_subtitle(
            subtitle, detection_results, fps
        )
        
        bbox_smoother = None
        
        subtitle_speakers[subtitle.index] = {
            'speaker_id': speaker_id,
            'speaker_bbox': speaker_bbox,
            'speaker_ids': speaker_ids if speaker_ids else [],
            'all_faces': all_faces,
            'bbox_smoother': bbox_smoother,
            'smoothing_alpha': smoothing_alpha,
            'text': subtitle.text_without_tags,
            'start_ms': subtitle.start.ordinal + int(subtitle_offset * 1000),
            'end_ms': subtitle.end.ordinal + int(subtitle_offset * 1000)
        }
        
        if generate_json:
            start_key = f"{subtitle.index}_start"
            json_speaker_id = int(speaker_id) if (speaker_id is not None and speaker_id >= 0) else None
            json_speaker_ids = [int(sid) for sid in speaker_ids] if speaker_ids else []
            
            json_annotations[start_key] = {
                'subtitle_id': int(subtitle.index),
                'position': 'standard',
                'timestamp': round(float(start_timestamp), 3),
                'text': subtitle.text_without_tags,
                'speaker_id': json_speaker_id,
                'speaker_ids': json_speaker_ids,
                'status': 'speaker_identified' if json_speaker_id is not None else 'speaker_not_visible'
            }
            
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
    
    # PHASE 3: Render
    print("\n" + "="*70)
    print("PHASE 3: Rendering final video with subtitles")
    print("="*70)

    cap = cv2.VideoCapture(input_video)
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Rendering")
    
    last_faces_in_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = int((frame_count / fps) * 1000)
        
        active_contexts = []
        for ctx in context_subtitles:
            if ctx['start_ms'] <= timestamp_ms <= ctx['end_ms']:
                active_contexts.append(ctx['text'])
        
        if active_contexts:
            frame = draw_context_stack(frame, active_contexts)
        
        current_subtitle = None
        current_subtitle_info = None
        
        for sub_idx, sub_info in subtitle_speakers.items():
            if sub_info['start_ms'] <= timestamp_ms <= sub_info['end_ms']:
                current_subtitle = sub_info['text']
                current_subtitle_info = sub_info
                break
        
        faces_in_frame = []
        if frame_count in detection_results:
            faces_in_frame = detection_results[frame_count]['faces']
            last_faces_in_frame = faces_in_frame
        else:
            faces_in_frame = last_faces_in_frame
        
        if current_subtitle and current_subtitle_info:
            speaker_id = current_subtitle_info['speaker_id']
            
            if speaker_id is not None and speaker_id >= 0:
                color = color_manager.get_color(speaker_id)
            else:
                color = color_manager.default_color
            
            current_speaker_bbox = None
            if speaker_id is not None and speaker_id >= 0:
                for face in faces_in_frame:
                    if face.get('match_idx') == speaker_id:
                        current_speaker_bbox = face['bbox']
                        break
            
            bbox_smoother = current_subtitle_info.get('bbox_smoother')
            display_bbox = None
            
            if bbox_smoother is None and current_speaker_bbox is not None:
                bbox_smoother = BBoxSmoother(current_speaker_bbox, alpha=smoothing_alpha)
                current_subtitle_info['bbox_smoother'] = bbox_smoother
                display_bbox = current_speaker_bbox
            elif bbox_smoother is not None:
                display_bbox = bbox_smoother.update(current_speaker_bbox)
            
            if display_bbox and speaker_id >= 0:
                x1, y1, x2, y2 = display_bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                
                id_label = f"Character {speaker_id}"
                label_font = cv2.FONT_HERSHEY_DUPLEX
                label_scale = 0.7
                label_thickness = 2
                label_size = cv2.getTextSize(id_label, label_font, label_scale, label_thickness)[0]
                
                label_padding = 8
                label_width = label_size[0] + label_padding * 2
                
                cv2.rectangle(frame, 
                            (int(x1), int(y1) - label_size[1] - label_padding * 2), 
                            (int(x1) + label_width, int(y1)), 
                            color, -1)
                
                cv2.putText(frame, id_label, 
                        (int(x1) + label_padding, int(y1) - label_padding),
                        label_font, label_scale, (255, 255, 255), 
                        label_thickness, cv2.LINE_AA)
                
                frame = draw_subtitle_beside_label(frame, current_subtitle, 
                                                display_bbox, color, label_width)
            else:
                frame = draw_subtitle_at_bottom(frame, current_subtitle, color)
            
            for face in faces_in_frame:
                face_id = face.get('match_idx', -1)
                if face_id >= 0 and face_id != speaker_id:
                    face_color = color_manager.get_color(face_id)
                    fx1, fy1, fx2, fy2 = face['bbox']
                    cv2.rectangle(frame, (int(fx1), int(fy1)), (int(fx2), int(fy2)), face_color, 1)
        
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    
    print(f"\nVideo rendering complete: {temp_video}")
    
    if generate_json and json_annotations:
        json_output_path = output_video.replace('.mp4', '_annotation_3.0.json')
        save_annotation_json(json_annotations, json_output_path)
    
    if preserve_audio and final_output:
        print("\n" + "="*70)
        print("PHASE 4: Merging audio from original video")
        print("="*70)
        
        success = merge_audio_to_video(temp_video, input_video, final_output)
        
        if success:
            try:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            except:
                pass
            print(f"\nFINAL OUTPUT WITH AUDIO: {final_output}")
    else:
        print(f"\nFINAL OUTPUT (NO AUDIO): {temp_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Speaker Subtitle Annotation with Auto-Sync")
    parser.add_argument("--input_video", required=True, help="Path to input video")
    parser.add_argument("--output_video", required=True, help="Path to output video")
    parser.add_argument("--centers_data", required=True, help="Path to centers_data.pkl from clustering")
    parser.add_argument("--srt", required=True, help="Path to SRT subtitle file")
    
    parser.add_argument("--detection_interval", type=int, default=1, help="Face detection interval")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Matching threshold")
    parser.add_argument("--speaking_threshold", type=float, default=0.05, help="MAR threshold")
    parser.add_argument("--subtitle_offset", type=float, default=0.0, help="Manual subtitle offset (if sync fails)")
    parser.add_argument("--no_audio", action="store_true", help="Disable audio preservation")
    parser.add_argument("--no_json", action="store_true", help="Disable JSON output")
    parser.add_argument("--skip_sync", action="store_true", help="Skip automatic audio synchronization")
    
    args = parser.parse_args()
    
    annotate_video_with_speaker_subtitles(
        input_video=args.input_video,
        output_video=args.output_video,
        centers_data_path=args.centers_data,
        subtitle_path=args.srt,
        detection_interval=args.detection_interval,
        similarity_threshold=args.similarity_threshold,
        speaking_threshold=args.speaking_threshold,
        preserve_audio=not args.no_audio,
        generate_json=not args.no_json,
        subtitle_offset=args.subtitle_offset,
        force_sync=not args.skip_sync
    )