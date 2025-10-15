#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised Speaking Moment Ground Truth Annotation Tool
Fixed issues with RetrievalResultsLoader
"""

import os
import cv2
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import pysrt
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysrt"])
    import pysrt

try:
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SpeakingMomentExtractor:
    """Extracts speaking moments"""
    
    def __init__(self, srt_path):
        self.srt_path = srt_path
        self.speaking_moments = []
    
    def extract_speaking_moments(self):
        """Extracts all speaking moments"""
        print(f"📖 Reading subtitle file: {self.srt_path}")
        
        try:
            subs = pysrt.open(self.srt_path, encoding='utf-8')
            print(f"✅ Loaded {len(subs)} subtitles")
            
            for sub in subs:
                start = self._time_to_seconds(sub.start)
                end = self._time_to_seconds(sub.end)
                
                moment = {
                    'subtitle_id': sub.index,
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'text': sub.text_without_tags.strip(),
                    'mid_time': (start + end) / 2
                }
                
                self.speaking_moments.append(moment)
            
            print(f"✅ Extracted {len(self.speaking_moments)} speaking moments")
            self._print_statistics()
            
            return self.speaking_moments
            
        except Exception as e:
            print(f"❌ Failed to read subtitles: {e}")
            return []
    
    def _time_to_seconds(self, time_obj):
        """Converts a time object to seconds"""
        return (time_obj.hours * 3600 + 
                time_obj.minutes * 60 + 
                time_obj.seconds + 
                time_obj.milliseconds / 1000.0)
    
    def _print_statistics(self):
        """Prints statistical information"""
        if not self.speaking_moments:
            return
        
        total_duration = sum(m['duration'] for m in self.speaking_moments)
        avg_duration = total_duration / len(self.speaking_moments)
        
        print(f"\n📊 Speaking Moment Statistics:")
        print(f"   Total count: {len(self.speaking_moments)}")
        print(f"   Total duration: {total_duration:.1f} seconds")
        print(f"   Average duration: {avg_duration:.2f} sec/segment")
        
        print(f"\n📝 First 3 examples:")
        for i, moment in enumerate(self.speaking_moments[:3], 1):
            print(f"   {i}. [{moment['start']:.2f}s-{moment['end']:.2f}s] {moment['text'][:40]}...")


class RetrievalResultsLoader:
    """Loads and processes character recognition results"""
    
    def __init__(self, results_path):
        self.results_path = results_path
        self.results = {}
    
    def load_results(self):
        """Loads the recognition results"""
        print(f"\n🔍 Loading character recognition results: {self.results_path}")
        
        try:
            path = Path(self.results_path)
            
            if path.suffix == '.pkl':
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # Normalize format
            self.results = self._normalize_results(data)
            
            print(f"✅ Loaded recognition results for {len(self.results)} frames")
            self._print_statistics()
            
            return self.results
            
        except Exception as e:
            print(f"❌ Load failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _normalize_results(self, data):
        """Normalizes the results format to {frame_idx: {...}}"""
        normalized = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Convert key to integer (frame index)
                try:
                    frame_idx = int(key)
                except:
                    continue
                
                # Ensure necessary fields exist
                if isinstance(value, dict):
                    normalized[frame_idx] = {
                        'character_id': value.get('character_id', value.get('match_idx', -1)),
                        'similarity': value.get('similarity', value.get('confidence', 0.0)),
                        'bbox': value.get('bbox', None),
                        'path': value.get('path', '')
                    }
        
        return normalized
    
    def _print_statistics(self):
        """Prints statistical information"""
        if not self.results:
            return
        
        # Count character distribution
        character_counts = defaultdict(int)
        for result in self.results.values():
            char_id = result['character_id']
            character_counts[char_id] += 1
        
        print(f"\n📊 Recognition Result Statistics:")
        for char_id in sorted(character_counts.keys()):
            count = character_counts[char_id]
            percentage = count / len(self.results) * 100
            print(f"   Character ID {char_id}: {count} frames ({percentage:.1f}%)")
    
    def get_character_at_time(self, timestamp, fps):
        """
        ✨ Fixed: Retrieves the character recognition result for a given timestamp
        
        Args:
            timestamp: Time in seconds
            fps: Video FPS
            
        Returns:
            A dictionary of the recognition result or None
        """
        frame_idx = int(timestamp * fps)
        
        # First, check for an exact match
        if frame_idx in self.results:
            return self.results[frame_idx]
        
        # Search nearby frames (±0.5 seconds)
        search_range = int(fps * 0.5)
        
        for offset in range(1, search_range + 1):
            # Check forward first
            check_idx = frame_idx + offset
            if check_idx in self.results:
                return self.results[check_idx]
            
            # Then check backward
            check_idx = frame_idx - offset
            if check_idx in self.results:
                return self.results[check_idx]
        
        # Return None if not found
        return None


class GroundTruthAnnotationTool:
    """Ground Truth Annotation Tool"""
    
    def __init__(self, video_path, speaking_moments, retrieval_results, 
                 output_dir, character_names=None):
        """
        Initializes the annotation tool
        
        Args:
            video_path: Path to the video file
            speaking_moments: A list of speaking moments
            retrieval_results: A RetrievalResultsLoader object (fixed)
            output_dir: Output directory
            character_names: A mapping of character IDs to names, e.g., {0: 'Jessica', 1: 'Eddie', ...}
        """
        self.video_path = video_path
        self.speaking_moments = speaking_moments
        self.retrieval_results = retrieval_results  # This is now an object
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.character_names = character_names or {}
        
        # Load the video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"\n🎬 Video Information:")
        print(f"   Path: {video_path}")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Total frames: {self.total_frames:,}")
        print(f"   Duration: {self.duration:.2f} seconds")
        
        # Annotation data
        self.annotations = {}
        self.current_idx = 0
        
        # Statistics
        self.stats = {
            'correct': 0,
            'wrong': 0,
            'unknown': 0,
            'skipped': 0
        }
    
    def get_character_name(self, char_id):
        """Gets the character name"""
        if char_id == -1:
            return "Unknown"
        return self.character_names.get(char_id, f"ID_{char_id}")
    
    def run_annotation(self):
        """Runs the annotation process"""
        print("\n" + "="*70)
        print("🏷️  Ground Truth Annotation Tool")
        print("="*70)
        print("\nInstructions:")
        print("  c: System identification is Correct")
        print("  w: System identification is Wrong - select another face")
        print("  1-9: Select face #1-9 in the frame as the speaker")
        print("  0: No one is speaking / Cannot determine")
        print("  u: Unknown/Uncertain")
        print("  s: Skip this moment")
        print("  n: Next moment")
        print("  p: Previous moment")
        print("  v: View nearby frames (when no faces detected)")
        print("  q: Save and quit")
        print("  ESC: Quit without saving")
        print("="*70)
        
        while self.current_idx < len(self.speaking_moments):
            moment = self.speaking_moments[self.current_idx]
            
            # Get the frame at the midpoint time
            mid_time = moment['mid_time']
            frame_idx = int(mid_time * self.fps)
            
            # Read the frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"⚠️  Could not read frame {frame_idx}, skipping")
                self.current_idx += 1
                continue
            
            # Get the system's recognition result
            auto_result = self.retrieval_results.get_character_at_time(mid_time, self.fps)
            auto_char_id = auto_result['character_id'] if auto_result else -1
            auto_similarity = auto_result['similarity'] if auto_result else 0.0
            auto_bbox = auto_result['bbox'] if auto_result else None
            
            # Get all faces in the frame
            all_faces = self._get_all_faces_in_frame(frame_idx)
            
            # Display the frame
            display_frame = self._prepare_display_frame(
                frame, moment, auto_char_id, auto_similarity, auto_bbox, all_faces
            )
            
            cv2.imshow('Ground Truth Annotation Tool', display_frame)
            
            # Wait for a key press
            key = cv2.waitKey(0) & 0xFF
            
            action = self._handle_key(key, moment, auto_char_id, all_faces)
            
            if action == 'quit':
                self._save_annotations()
                break
            elif action == 'quit_nosave':
                print("\n❌ Exiting without saving")
                break
            elif action == 'next':
                self.current_idx += 1
            elif action == 'prev':
                self.current_idx = max(0, self.current_idx - 1)
        
        cv2.destroyAllWindows()
        self.cap.release()
        
        # Display final statistics
        self._print_final_stats()
    
    def _prepare_display_frame(self, frame, moment, auto_char_id, auto_similarity, auto_bbox, all_faces):
        """Prepares the display frame - shows all faces and highlights the speaker"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw all detected faces with numbering
        for idx, face_info in enumerate(all_faces, 1):
            char_id = face_info['character_id']
            similarity = face_info['similarity']
            bbox = face_info['bbox']
            
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Check if this is the system-identified speaker
            is_speaking = (char_id == auto_char_id)
            
            if is_speaking:
                # Speaker: Thick green box + "SPEAKING" label
                color = (0, 255, 0)  # Green
                thickness = 4  # Thick
                label = f"#{idx} SPEAKING: {self.get_character_name(char_id)}"
                label_bg_color = (0, 200, 0)
            else:
                # Others: Thin gray box
                color = (200, 200, 200)  # Gray
                thickness = 2  # Thin
                label = f"#{idx} {self.get_character_name(char_id)}"
                label_bg_color = (100, 100, 100)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw number circle (top-left corner)
            circle_center = (x1 + 20, y1 + 20)
            cv2.circle(display_frame, circle_center, 18, color, -1)
            cv2.putText(display_frame, str(idx), (x1 + 13, y1 + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, 
                          (x1, y1 - text_size[1] - 10), 
                          (x1 + text_size[0] + 10, y1),
                          label_bg_color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display similarity (small text)
            sim_text = f"Sim: {similarity:.2f}"
            cv2.putText(display_frame, sim_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # If no faces are detected, show a detailed warning
        if not all_faces:
            # Check if faces are detected in nearby frames
            nearby_frames_with_faces = self._check_nearby_frames_for_faces(int(moment['mid_time'] * self.fps))
            
            if nearby_frames_with_faces:
                warning_text = f"⚠️ No faces in this frame, but found {nearby_frames_with_faces} faces nearby"
                hint_text = "(Speaker might be in another frame - Press 'v' to check)"
                cv2.putText(display_frame, warning_text, 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(display_frame, hint_text, 
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                warning_text = "⚠️ No faces detected in this segment"
                hint_text = "(Likely narrator/off-screen speaker - Press '0' for no speaker)"
                cv2.putText(display_frame, warning_text, 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display_frame, hint_text, 
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Create information panel
        frame_idx = int(moment['mid_time'] * self.fps)
        panel_height = 250
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Display information
        info_lines = [
            f"Progress: {self.current_idx + 1}/{len(self.speaking_moments)}",
            f"Time: {moment['start']:.2f}s - {moment['end']:.2f}s (Frame: {frame_idx})",
            f"Subtitle: {moment['text'][:50]}",
            f"",
            f"System thinks SPEAKING: {self.get_character_name(auto_char_id)} (Sim: {auto_similarity:.3f})",
            f"Total faces in frame: {len(all_faces)}",
            f"",
            f"Press 1-{len(all_faces)} to select that face, or c/w/u/s",
        ]
        
        # Display existing annotation
        moment_key = f"{moment['subtitle_id']}"
        if moment_key in self.annotations:
            ann = self.annotations[moment_key]
            info_lines.append(f"Your Annotation: {self.get_character_name(ann['ground_truth_id'])}")
        else:
            info_lines.append(f"Your Annotation: (Not yet annotated)")
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(panel, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30
        
        # Combine the frame and panel
        result = np.vstack([display_frame, panel])
        
        return result
    
    def _check_nearby_frames_for_faces(self, frame_idx):
        """
        Checks if faces are detected in nearby frames.
        Helps to determine if it's a narrator or a detection failure.
        
        Returns:
            The total number of faces detected in nearby frames.
        """
        nearby_face_count = 0
        search_range = int(self.fps * 2)  # ±2 seconds
        
        for offset in range(-search_range, search_range + 1):
            check_idx = frame_idx + offset
            if check_idx in self.retrieval_results.results:
                nearby_face_count += 1
        
        return nearby_face_count
    
    def _get_all_faces_in_frame(self, frame_idx):
        """
        Gets all detected faces in a specific frame.
        
        Returns:
            A list of face info dicts with character_id, similarity, bbox.
        """
        all_faces = []
        
        # Search in nearby frames (±5 frames)
        search_range = 5
        
        for offset in range(-search_range, search_range + 1):
            check_idx = frame_idx + offset
            
            if check_idx in self.retrieval_results.results:
                result = self.retrieval_results.results[check_idx]
                
                # Simple deduplication: If the same character is in a similar position, keep only one
                is_duplicate = False
                for existing_face in all_faces:
                    if existing_face['character_id'] == result['character_id']:
                        # If bboxes are close, consider it a duplicate
                        if result.get('bbox') and existing_face.get('bbox'):
                            # Calculate the distance between center points
                            ex1, ey1, ex2, ey2 = existing_face['bbox']
                            rx1, ry1, rx2, ry2 = result['bbox']
                            
                            ex_center = ((ex1 + ex2) / 2, (ey1 + ey2) / 2)
                            rx_center = ((rx1 + rx2) / 2, (ry1 + ry2) / 2)
                            
                            distance = np.sqrt((ex_center[0] - rx_center[0])**2 + 
                                               (ex_center[1] - rx_center[1])**2)
                            
                            if distance < 50:  # Within 50 pixels is considered the same person
                                is_duplicate = True
                                break
                
                if not is_duplicate:
                    all_faces.append({
                        'character_id': result['character_id'],
                        'similarity': result['similarity'],
                        'bbox': result.get('bbox'),
                        'path': result.get('path', '')
                    })
        
        return all_faces
    
    def _handle_key(self, key, moment, auto_char_id, all_faces):
        """Handles key press events - supports selecting faces in the frame"""
        moment_key = f"{moment['subtitle_id']}"
        
        if key == ord('q'):
            return 'quit'
        elif key == 27:  # ESC
            return 'quit_nosave'
        elif key == ord('n'):
            return 'next'
        elif key == ord('p'):
            return 'prev'
        elif key == ord('c'):
            # Mark as correct
            self.annotations[moment_key] = {
                'subtitle_id': moment['subtitle_id'],
                'start': moment['start'],
                'end': moment['end'],
                'text': moment['text'],
                'ground_truth_id': auto_char_id,
                'system_predicted_id': auto_char_id,
                'status': 'correct'
            }
            self.stats['correct'] += 1
            print(f"✓ Marked as correct (ID: {auto_char_id})")
            return 'next'
        
        elif key == ord('w'):
            # Mark as wrong, let the user select the correct face
            print(f"\nSystem identified: {self.get_character_name(auto_char_id)}")
            print(f"Available faces in frame:")
            for idx, face in enumerate(all_faces, 1):
                print(f"  {idx}. {self.get_character_name(face['character_id'])} (ID: {face['character_id']})")
            
            try:
                choice = input(f"Select face number (1-{len(all_faces)}) or enter custom ID: ")
                
                # Check if the input is a numeric choice
                if choice.isdigit():
                    choice_num = int(choice)
                    
                    # Check if it's within the range
                    if 1 <= choice_num <= len(all_faces):
                        # Select a face from the frame
                        selected_face = all_faces[choice_num - 1]
                        correct_id = selected_face['character_id']
                    else:
                        # Treat as a custom ID
                        correct_id = choice_num
                else:
                    print("❌ Invalid input. Skipping.")
                    return 'stay'
                
                self.annotations[moment_key] = {
                    'subtitle_id': moment['subtitle_id'],
                    'start': moment['start'],
                    'end': moment['end'],
                    'text': moment['text'],
                    'ground_truth_id': correct_id,
                    'system_predicted_id': auto_char_id,
                    'status': 'corrected'
                }
                self.stats['wrong'] += 1
                print(f"✓ Annotated with correct ID: {correct_id}")
            except (ValueError, IndexError):
                print("❌ Invalid input. Skipping.")
                return 'stay'
            return 'next'
        
        elif key == ord('u'):
            # Mark as unknown
            self.annotations[moment_key] = {
                'subtitle_id': moment['subtitle_id'],
                'start': moment['start'],
                'end': moment['end'],
                'text': moment['text'],
                'ground_truth_id': -1,
                'system_predicted_id': auto_char_id,
                'status': 'unknown'
            }
            self.stats['unknown'] += 1
            print("? Marked as unknown")
            return 'next'
        
        elif key == ord('s'):
            # Skip
            self.stats['skipped'] += 1
            print("→ Skipped")
            return 'next'
        
        elif ord('1') <= key <= ord('9'):
            # Quickly select a face in the frame (1-9)
            face_num = key - ord('0')
            
            if face_num <= len(all_faces):
                selected_face = all_faces[face_num - 1]
                correct_id = selected_face['character_id']
                
                self.annotations[moment_key] = {
                    'subtitle_id': moment['subtitle_id'],
                    'start': moment['start'],
                    'end': moment['end'],
                    'text': moment['text'],
                    'ground_truth_id': correct_id,
                    'system_predicted_id': auto_char_id,
                    'status': 'manual_select'
                }
                
                if correct_id == auto_char_id:
                    self.stats['correct'] += 1
                else:
                    self.stats['wrong'] += 1
                
                print(f"✓ Selected face #{face_num}: {self.get_character_name(correct_id)}")
                return 'next'
            else:
                print(f"❌ Face #{face_num} not found. Only {len(all_faces)} faces in frame.")
                return 'stay'
        
        elif key == ord('0'):
            # No speaker or cannot determine
            self.annotations[moment_key] = {
                'subtitle_id': moment['subtitle_id'],
                'start': moment['start'],
                'end': moment['end'],
                'text': moment['text'],
                'ground_truth_id': -1,
                'system_predicted_id': auto_char_id,
                'status': 'no_speaker'
            }
            self.stats['unknown'] += 1
            print("⓪ Marked as no speaker")
            return 'next'
            
        elif key == ord('v'):
            # View nearby frames (when no faces are detected)
            if not all_faces:
                print(f"\n🔍 Checking nearby frames...")
                mid_time = moment['mid_time']
                frame_idx = int(mid_time * self.fps)
                
                # Check frames within ±1 second
                search_range = int(self.fps)
                found_faces = []
                
                # Check every 5 frames
                for offset in range(-search_range, search_range + 1, 5):
                    check_idx = frame_idx + offset
                    if check_idx in self.retrieval_results.results:
                        result = self.retrieval_results.results[check_idx]
                        time_offset = offset / self.fps
                        found_faces.append({
                            'frame': check_idx,
                            'time_offset': time_offset,
                            'character_id': result['character_id'],
                            'character_name': self.get_character_name(result['character_id'])
                        })
                
                if found_faces:
                    print(f"Found {len(found_faces)} faces in nearby frames:")
                    for face in found_faces[:10]:  # Show first 10
                        print(f"  {face['time_offset']:+.2f}s: {face['character_name']} (ID: {face['character_id']})")
                    
                    choice = input(f"\nSelect character ID or press Enter to skip: ")
                    if choice.isdigit():
                        correct_id = int(choice)
                        self.annotations[moment_key] = {
                            'subtitle_id': moment['subtitle_id'],
                            'start': moment['start'],
                            'end': moment['end'],
                            'text': moment['text'],
                            'ground_truth_id': correct_id,
                            'system_predicted_id': auto_char_id,
                            'status': 'from_nearby_frame'
                        }
                        print(f"✓ Annotated as: {self.get_character_name(correct_id)}")
                        return 'next'
                else:
                    print("No faces found in nearby frames - likely narrator/off-screen")
                    print("Press '0' to mark as no speaker")
            else:
                print("Faces already detected in current frame. Use 1-9 or 'w' to select.")
            
            return 'stay'
        
        return 'stay'

    def _save_annotations(self):
        """Saves the annotation results"""
        output_file = self.output_dir / 'speaking_moments_ground_truth.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Ground truth saved to: {output_file}")
        print(f"   Annotated a total of {len(self.annotations)} speaking moments")
    
    def _print_final_stats(self):
        """Prints the final statistics"""
        print("\n" + "="*70)
        print("📊 Annotation Statistics")
        print("="*70)
        print(f"Total speaking moments: {len(self.speaking_moments)}")
        print(f"Annotated: {len(self.annotations)}")
        print(f"  - System correct: {self.stats['correct']}")
        print(f"  - System wrong: {self.stats['wrong']}")
        print(f"  - Unknown: {self.stats['unknown']}")
        print(f"  - Skipped: {self.stats['skipped']}")
        
        if self.stats['correct'] + self.stats['wrong'] > 0:
            accuracy = self.stats['correct'] / (self.stats['correct'] + self.stats['wrong'])
            print(f"\nPreliminary Accuracy: {accuracy*100:.2f}%")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Speaking Moment Ground Truth Annotation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', required=True, help='Path to the video file')
    parser.add_argument('--srt', required=True, help='Path to the SRT subtitle file')
    parser.add_argument('--results', required=True, help='System recognition result file (.pkl or .json)')
    parser.add_argument('--output', default='./ground_truth', help='Output directory')
    parser.add_argument('--character-names', help='JSON file for mapping character IDs to names (optional)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🏷️  Speaking Moment Ground Truth Annotation Tool")
    print("="*70)
    
    try:
        # Step 1: Extract Speaking Moments
        print("\n【Step 1】Extracting Speaking Moments")
        print("-"*70)
        extractor = SpeakingMomentExtractor(args.srt)
        speaking_moments = extractor.extract_speaking_moments()
        
        if not speaking_moments:
            print("❌ No speaking moments found.")
            return 1
        
        # Step 2: Load System Recognition Results
        print("\n【Step 2】Loading System Recognition Results")
        print("-"*70)
        loader = RetrievalResultsLoader(args.results)
        loader.load_results()  # Load first, the loader object itself contains the results
        
        if not loader.results:
            print("❌ Failed to load retrieval results.")
            return 1

        # Step 3: Load Character Names (Optional)
        character_names = {}
        if args.character_names:
            try:
                with open(args.character_names, 'r', encoding='utf-8') as f:
                    character_names_str_keys = json.load(f)
                    character_names = {int(k): v for k, v in character_names_str_keys.items()}
                print(f"✅ Loaded {len(character_names)} character names.")
            except Exception as e:
                print(f"⚠️  Could not load character names file: {e}")

        # Step 4: Start Annotation
        print("\n【Step 3】Starting Annotation")
        print("-"*70)
        tool = GroundTruthAnnotationTool(
            video_path=args.video,
            speaking_moments=speaking_moments,
            retrieval_results=loader,  # Pass the entire loader object
            output_dir=args.output,
            character_names=character_names
        )
        
        tool.run_annotation()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())