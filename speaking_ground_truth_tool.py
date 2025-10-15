#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Speaking Moment Ground Truth Annotation Tool
Improvements:
1. Real-time face detection with bounding box display.
2. Fixes the contradiction of having recognition results without faces.
3. Supports annotating at the start, middle, and end points of each subtitle.
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

try:
    import pysrt
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysrt"])
    import pysrt

# Import your modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import feature_extraction
from enhanced_face_preprocessing import detect_foreground_faces_in_frame
import facenet.src.align.detect_face as detect_face


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
        """Initializes the models"""
        print("🔧 Initializing face detection and recognition models...")
        
        # Load cluster centers
        with open(self.centers_data_path, 'rb') as f:
            centers_data = pickle.load(f)
        self.centers, self.center_paths = centers_data['cluster_centers']
        print(f"✅ Loaded {len(self.centers)} character centers")
        
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
        
        print("✅ Models initialized successfully")
    
    def detect_and_identify_faces(self, frame):
        """
        Detects and identifies faces in a frame.
        
        Returns:
            List[dict]: Information for each face {bbox, character_id, similarity, encoding}
        """
        with self.graph.as_default():
            # 1. Detect faces
            filtered_bboxes = detect_foreground_faces_in_frame(
                frame, self.pnet, self.rnet, self.onet,
                min_face_size=60,
                min_face_area_ratio=0.008,
                max_faces_per_frame=5
            )
            
            if not filtered_bboxes:
                return []
            
            # 2. Identify each face
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
                
                # Only accept results above the threshold
                character_id = best_idx if best_sim > 0.55 else -1
                
                faces.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'character_id': character_id,
                    'similarity': float(best_sim),
                    'encoding': encoding
                })
            
            return faces
    
    def cleanup(self):
        """Cleans up resources"""
        if self.sess:
            self.sess.close()


class ImprovedGroundTruthTool:
    """Improved Ground Truth Annotation Tool"""
    
    def __init__(self, video_path, srt_path, model_dir, centers_data_path, 
                 output_dir, character_names=None):
        self.video_path = video_path
        self.srt_path = srt_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.character_names = character_names or {}
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"\n🎬 Video Information:")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Total frames: {self.total_frames:,}")
        print(f"   Duration: {self.duration:.2f} seconds")
        
        # Load subtitles and expand annotation points
        self.annotation_points = self._load_and_expand_subtitles()
        
        # Initialize face detector
        self.detector = RealTimeFaceDetector(model_dir, centers_data_path)
        self.detector.initialize()
        
        # Annotation data
        self.annotations = {}
        self.current_idx = 0
        
        # Cache: to avoid re-detecting the same frame
        self.frame_cache = {}
        
    def _load_and_expand_subtitles(self):
        """
        Loads subtitles and expands them into multiple annotation points.
        Each subtitle generates 3 points: start, middle, and end.
        """
        print(f"\n📖 Loading subtitles: {self.srt_path}")
        
        # Handle BOM issue: first read the file content and remove the BOM
        try:
            with open(self.srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            # Write to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', 
                                             delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Load from the temporary file
            subs = pysrt.open(tmp_path, encoding='utf-8')
            
            # Delete the temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            print(f"⚠️  Falling back to standard subtitle loading method")
            subs = pysrt.open(self.srt_path, encoding='utf-8')
        
        annotation_points = []
        
        for sub in subs:
            start_sec = self._time_to_seconds(sub.start)
            end_sec = self._time_to_seconds(sub.end)
            mid_sec = (start_sec + end_sec) / 2
            
            text = sub.text_without_tags.strip()
            
            # Clean up subtitle_id, removing potential BOM or other special characters
            subtitle_id = int(str(sub.index).strip().lstrip('\ufeff'))
            
            # Create 3 annotation points for each subtitle
            for position, timestamp in [
                ('start', start_sec),
                ('mid', mid_sec),
                ('end', end_sec)
            ]:
                annotation_points.append({
                    'subtitle_id': subtitle_id,  # Already a clean int
                    'position': position,
                    'timestamp': timestamp,
                    'text': text,
                    'duration': end_sec - start_sec
                })
        
        print(f"✅ Loaded {len(subs)} subtitles, expanded to {len(annotation_points)} annotation points")
        return annotation_points
    
    def _time_to_seconds(self, time_obj):
        """Converts a time object to seconds"""
        return (time_obj.hours * 3600 + 
                time_obj.minutes * 60 + 
                time_obj.seconds + 
                time_obj.milliseconds / 1000.0)
    
    def get_character_name(self, char_id):
        """Gets the character name"""
        if char_id == -1:
            return "Unknown"
        return self.character_names.get(char_id, f"ID_{char_id}")
    
    def _get_frame_at_time(self, timestamp):
        """Gets the frame at a specific timestamp"""
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
        
        # Cache the result (limit cache size)
        if len(self.frame_cache) > 100:
            # Remove the oldest cache entry
            oldest_key = min(self.frame_cache.keys())
            del self.frame_cache[oldest_key]
        
        self.frame_cache[frame_idx] = {'frame': frame.copy(), 'faces': faces}
        
        return frame, faces
    
    def _prepare_display_frame(self, frame, point, faces):
        """Prepares the frame for display"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw all detected faces
        for idx, face in enumerate(faces, 1):
            char_id = face['character_id']
            similarity = face['similarity']
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            
            # Default color: Orange (not annotated)
            color = (0, 150, 255) # Orange
            thickness = 3
            label = f"#{idx} {self.get_character_name(char_id)}"
            
            # Check if this face has been annotated
            point_key = f"{point['subtitle_id']}_{point['position']}"
            if point_key in self.annotations:
                # If this face's ID is in the annotation
                if char_id in self.annotations[point_key].get('all_faces', []):
                    color = (0, 255, 0)  # Green: Annotated and correct
                    thickness = 4
                    label = f"#{idx} ✓ {self.get_character_name(char_id)}"
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw number circle
            cv2.circle(display_frame, (x1 + 20, y1 + 20), 18, color, -1)
            cv2.putText(display_frame, str(idx), (x1 + 13, y1 + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), 
                          (x1 + text_size[0] + 10, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display similarity
            sim_text = f"Sim: {similarity:.2f}"
            cv2.putText(display_frame, sim_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # If no faces are detected
        if not faces:
            warning = "⚠️ No faces detected in this frame"
            cv2.putText(display_frame, warning, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Create info panel
        panel_height = 250
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Check current annotation status
        point_key = f"{point['subtitle_id']}_{point['position']}"
        annotation_status = "Not annotated"
        if point_key in self.annotations:
            ann = self.annotations[point_key]
            if ann['status'] == 'all_correct':
                annotation_status = f"✓ All correct (Speaker: ID {ann['speaker_id']})"
            elif ann['status'] == 'partially_corrected':
                annotation_status = f"⚠️ Corrected (Speaker: ID {ann['speaker_id']})"
            elif ann['status'] == 'unknown':
                annotation_status = "? Unknown"
        
        info_lines = [
            f"Progress: {self.current_idx + 1}/{len(self.annotation_points)}",
            f"Subtitle #{point['subtitle_id']} - Position: {point['position']}",
            f"Time: {point['timestamp']:.2f}s | Text: {point['text'][:40]}",
            f"Detected faces: {len(faces)}",
            f"Current status: {annotation_status}",
            "",
            "Instructions:",
            "  c = All Correct (all face IDs are correct, then specify speaker)",
            "  m = Manual Annotation (confirm/correct each face ID one by one)",
            "  u = Unknown | s = Skip | n = Next | p = Previous | q = Save & Quit"
        ]
        
        y_offset = 22
        for line in info_lines:
            cv2.putText(panel, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 22
        
        # Combine frame and panel
        result = np.vstack([display_frame, panel])
        return result
    
    def _handle_key(self, key, point, faces):
        """Handles key presses - improved annotation workflow"""
        # Ensure subtitle_id is a clean integer
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
            print("→ Skipped")
            return 'next'
        elif key == ord('u'):
            # Unknown
            self.annotations[point_key] = {
                'subtitle_id': subtitle_id,
                'position': str(point['position']),
                'timestamp': float(point['timestamp']),
                'text': str(point['text']),
                'speaker_id': -1,
                'all_faces': [],
                'status': 'unknown'
            }
            print("? Marked as unknown")
            return 'next'
        
        elif key == ord('c'):
            # All correct: assumes all face IDs are correct, let user specify the speaker
            if not faces:
                print("❌ No faces to annotate.")
                return 'stay'
            
            print(f"\n✓ Confirming all face IDs are correct.")
            print("Faces in frame:")
            for idx, face in enumerate(faces, 1):
                print(f"  #{idx}: {self.get_character_name(face['character_id'])} (ID: {face['character_id']})")
            
            # Let user specify the speaker
            try:
                speaker_choice = input(f"\nPlease select the speaker's number (1-{len(faces)}, or press Enter to skip): ").strip()
                
                if speaker_choice == "":
                    print("Speaker annotation skipped.")
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
                        'all_faces': all_face_ids,
                        'status': 'all_correct'
                    }
                    print(f"✓ Annotated: Speaker is ID {speaker_id}")
                    return 'next'
                else:
                    print("❌ Invalid number.")
                    return 'stay'
            except:
                print("❌ Invalid input.")
                return 'stay'
        
        elif key == ord('m'):
            # Manual annotation: confirm each face one by one
            if not faces:
                print("❌ No faces to annotate.")
                return 'stay'
            
            print(f"\n🖊️  Manual Annotation Mode")
            print("System detected faces:")
            
            corrected_faces = []
            
            for idx, face in enumerate(faces, 1):
                detected_id = face['character_id']
                print(f"\nFace #{idx}:")
                print(f"  System ID: {self.get_character_name(detected_id)} (ID: {detected_id})")
                
                # Let user confirm or correct
                response = input(f"  Is this correct? (y=yes / enter correct ID / n=skip): ").strip().lower()
                
                if response == 'y':
                    corrected_faces.append(detected_id)
                    print(f"  ✓ Confirmed as ID {detected_id}")
                elif response == 'n':
                    print(f"  ✗ Skipped this face")
                    continue
                else:
                    # Try to parse as ID
                    try:
                        correct_id = int(response)
                        corrected_faces.append(correct_id)
                        print(f"  ✓ Corrected to ID {correct_id}")
                    except:
                        print(f"  ✗ Invalid input, skipping this face")
                        continue
            
            if not corrected_faces:
                print("❌ No faces were annotated.")
                return 'stay'
            
            # Let user specify the speaker from the corrected list
            print(f"\nAnnotated faces: {corrected_faces}")
            try:
                speaker_input = input(f"Please enter the speaker's ID (or press Enter to skip): ").strip()
                
                if speaker_input == "":
                    speaker_id = corrected_faces[0] if corrected_faces else -1
                else:
                    speaker_id = int(speaker_input)
                    if speaker_id not in corrected_faces:
                        print(f"⚠️  Warning: ID {speaker_id} is not in the list of annotated faces.")
                
                self.annotations[point_key] = {
                    'subtitle_id': subtitle_id,
                    'position': str(point['position']),
                    'timestamp': float(point['timestamp']),
                    'text': str(point['text']),
                    'speaker_id': speaker_id,
                    'all_faces': corrected_faces,
                    'status': 'partially_corrected'
                }
                print(f"✓ Annotation complete.")
                return 'next'
                
            except:
                print("❌ Invalid input.")
                return 'stay'
        
        return 'stay'
    
    def run_annotation(self):
        """Runs the annotation process"""
        print("\n" + "="*70)
        print("🏷️  Improved Ground Truth Annotation Tool")
        print("="*70)
        print("\n📋 Annotation Workflow Instructions:")
        print("1. Observe all faces in the frame (Orange box = Not annotated, Green box = Annotated).")
        print("2. Choose an action:")
        print("   c = All Correct (if all face IDs are correct) → You will be asked to specify the speaker.")
        print("   m = Manual Annotation (to confirm/correct each face ID) → You will be prompted for each face.")
        print("   u = Unknown / s = Skip / n = Next / p = Previous")
        print("   q = Save & Quit / ESC = Quit without saving")
        print("\n💡 Important Tips:")
        print("   - If there are 2 faces and both are correct: Press 'c' then select the speaker.")
        print("   - If any face ID is incorrect: Press 'm' to enter Manual Annotation mode.")
        print("   - The annotation data includes: The speaker's ID + all visible face IDs.")
        print("="*70)
        
        input("\nPress Enter to start annotating...")
        
        while self.current_idx < len(self.annotation_points):
            point = self.annotation_points[self.current_idx]
            
            # Get frame and faces
            frame, faces = self._get_frame_at_time(point['timestamp'])
            
            if frame is None:
                print(f"⚠️  Could not read frame, skipping.")
                self.current_idx += 1
                continue
            
            # Prepare for display
            display_frame = self._prepare_display_frame(frame, point, faces)
            cv2.imshow('Ground Truth Annotation Tool', display_frame)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            action = self._handle_key(key, point, faces)
            
            if action == 'quit':
                self._save_annotations()
                break
            elif action == 'quit_nosave':
                print("\n❌ Exiting without saving.")
                break
            elif action == 'next':
                self.current_idx += 1
            elif action == 'prev':
                self.current_idx = max(0, self.current_idx - 1)
        
        cv2.destroyAllWindows()
        self.cap.release()
        self.detector.cleanup()
        
        self._print_stats()
    
    def _save_annotations(self):
        """Saves annotations"""
        output_file = self.output_dir / 'speaking_moments_ground_truth_improved.json'
        
        # Convert to a JSON-serializable format
        serializable_annotations = {}
        for key, value in self.annotations.items():
            # This block might need adjustment to match the new data structure from _handle_key
            serializable_annotations[key] = {
                'subtitle_id': int(value.get('subtitle_id', 0)),
                'position': value.get('position', ''),
                'timestamp': float(value.get('timestamp', 0.0)),
                'text': value.get('text', ''),
                'speaker_id': int(value.get('speaker_id', -1)),
                'all_faces': value.get('all_faces', []),
                'status': value.get('status', 'unknown')
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_file}")
        print(f"   Number of annotations: {len(self.annotations)}")
    
    def _print_stats(self):
        """Prints statistics"""
        if not self.annotations:
            return
        
        stats = defaultdict(int)
        for ann in self.annotations.values():
            stats[ann['status']] += 1
        
        print("\n" + "="*70)
        print("📊 Annotation Statistics")
        print("="*70)
        print(f"Total annotation points: {len(self.annotation_points)}")
        print(f"Annotated: {len(self.annotations)}")
        for status, count in stats.items():
            print(f"  - {status}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Improved Speaking Moment Ground Truth Annotation Tool')
    parser.add_argument('--video', required=True, help='Path to the video file')
    parser.add_argument('--srt', required=True, help='Path to the subtitle file')
    parser.add_argument('--model-dir', required=True, help='Path to the FaceNet model directory')
    parser.add_argument('--centers-data', required=True, help='Path to the cluster centers data file')
    parser.add_argument('--output', default='./ground_truth_improved', help='Output directory')
    parser.add_argument('--character-names', help='Path to the character names JSON file')
    
    args = parser.parse_args()
    
    # Load character names
    character_names = {}
    if args.character_names:
        try:
            with open(args.character_names, 'r', encoding='utf-8') as f:
                names_data = json.load(f)
                character_names = {int(k): v for k, v in names_data.items()}
        except Exception as e:
            print(f"⚠️  Could not load character names: {e}")
    
    # Create the tool instance
    tool = ImprovedGroundTruthTool(
        video_path=args.video,
        srt_path=args.srt,
        model_dir=args.model_dir,
        centers_data_path=args.centers_data,
        output_dir=args.output,
        character_names=character_names
    )
    
    # Run annotation
    tool.run_annotation()


if __name__ == '__main__':
    exit(main())