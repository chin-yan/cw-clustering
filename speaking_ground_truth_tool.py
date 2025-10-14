#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speaking Moment Ground Truth Annotation Tool
Used to evaluate whether a character recognition system correctly identifies the speaker
when subtitles appear.
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
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle

# Set up fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SpeakingMomentExtractor:
    """Extracts speaking moments from an SRT subtitle file."""
    
    def __init__(self, srt_path):
        self.srt_path = srt_path
        self.speaking_moments = []
    
    def extract_speaking_moments(self):
        """Extracts all speaking moments."""
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
                    'mid_time': (start + end) / 2  # Midpoint time
                }
                
                self.speaking_moments.append(moment)
            
            print(f"✅ Extracted {len(self.speaking_moments)} speaking moments")
            self._print_statistics()
            
            return self.speaking_moments
            
        except Exception as e:
            print(f"❌ Failed to read subtitles: {e}")
            return []
    
    def _time_to_seconds(self, time_obj):
        """Converts a time object to seconds."""
        return (time_obj.hours * 3600 + 
                time_obj.minutes * 60 + 
                time_obj.seconds + 
                time_obj.milliseconds / 1000.0)
    
    def _print_statistics(self):
        """Displays statistics."""
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
    """Loads your character recognition results."""
    
    def __init__(self, results_path):
        """
        Args:
            results_path: Path to the recognition results file.
                         Can be .pkl (pickle) or .json.
        """
        self.results_path = results_path
        self.results = {}
    
    def load_results(self):
        """Loads the recognition results."""
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
            
            # Convert to a unified format
            # Expected format: {frame_idx: {'character_id': int, 'similarity': float, 'bbox': tuple}}
            self.results = self._normalize_results(data)
            
            print(f"✅ Loaded recognition results for {len(self.results)} frames")
            self._print_statistics()
            
            return self.results
            
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return {}
    
    def _normalize_results(self, data):
        """Normalizes the result format."""
        # If it's a dict and keys are strings, convert them to integers
        normalized = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Convert key to integer (frame index)
                try:
                    frame_idx = int(key)
                except:
                    continue
                
                # Ensure necessary fields are present
                if isinstance(value, dict):
                    normalized[frame_idx] = {
                        'character_id': value.get('character_id', value.get('match_idx', -1)),
                        'similarity': value.get('similarity', value.get('confidence', 0.0)),
                        'bbox': value.get('bbox', None)
                    }
        
        return normalized
    
    def _print_statistics(self):
        """Displays statistics."""
        if not self.results:
            return
        
        # Tally the identified characters
        character_counts = defaultdict(int)
        for result in self.results.values():
            char_id = result['character_id']
            character_counts[char_id] += 1
        
        print(f"\n📊 Recognition Result Statistics:")
        for char_id in sorted(character_counts.keys()):
            count = character_counts[char_id]
            print(f"   Character ID {char_id}: {count} frames ({count/len(self.results)*100:.1f}%)")
    
    def get_character_at_time(self, timestamp, fps):
        """
        Gets the character recognition result at a specific timestamp.
        
        Args:
            timestamp: Time in seconds.
            fps: Video FPS.
            
        Returns:
            A dictionary of the recognition result or None.
        """
        frame_idx = int(timestamp * fps)
        
        # If no result for this frame, search nearby frames
        search_range = int(fps * 0.5)  # Search +/- 0.5 seconds
        
        for offset in range(0, search_range):
            # Check current, then check before and after
            for check_idx in [frame_idx, frame_idx + offset, frame_idx - offset]:
                if check_idx in self.results:
                    return self.results[check_idx]
        
        return None


class GroundTruthAnnotationTool:
    """Ground Truth Annotation Tool."""
    
    def __init__(self, video_path, speaking_moments, retrieval_results, 
                 output_dir, character_names=None):
        """
        Initializes the annotation tool.
        
        Args:
            video_path: Path to the video.
            speaking_moments: List of speaking moments.
            retrieval_results: Dictionary of recognition results.
            output_dir: Output directory.
            character_names: Mapping of character IDs to names {0: 'Jessica', 1: 'Eddie', ...}.
        """
        self.video_path = video_path
        self.speaking_moments = speaking_moments
        self.retrieval_results = retrieval_results
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
        """Gets the character name."""
        if char_id == -1:
            return "Unknown"
        return self.character_names.get(char_id, f"ID_{char_id}")
    
    def run_annotation(self):
        """Runs the annotation process."""
        print("\n" + "="*70)
        print("🏷️  Ground Truth Annotation Tool")
        print("="*70)
        print("\nInstructions:")
        print("  Number keys 0-9: Annotate as Character ID")
        print("  c: System identification is Correct")
        print("  w: System identification is Wrong - then enter correct ID")
        print("  u: Cannot determine (Unknown)")
        print("  s: Skip this moment")
        print("  n: Next moment")
        print("  p: Previous moment")
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
            
            # Display the frame
            display_frame = self._prepare_display_frame(
                frame, moment, auto_char_id, auto_similarity, auto_bbox
            )
            
            cv2.imshow('Ground Truth Annotation Tool', display_frame)
            
            # Wait for keypress
            key = cv2.waitKey(0) & 0xFF
            
            action = self._handle_key(key, moment, auto_char_id)
            
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
        
        # Display final stats
        self._print_final_stats()
    
    def _prepare_display_frame(self, frame, moment, auto_char_id, auto_similarity, auto_bbox):
        """Prepares the display frame with info."""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw the system's identified bounding box
        if auto_bbox:
            x1, y1, x2, y2 = [int(coord) for coord in auto_bbox]
            color = (0, 255, 0) if auto_char_id >= 0 else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Display character ID
            label = self.get_character_name(auto_char_id)
            cv2.putText(display_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Create an information panel
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Display information
        info_lines = [
            f"Progress: {self.current_idx + 1}/{len(self.speaking_moments)}",
            f"Time: {moment['start']:.2f}s - {moment['end']:.2f}s",
            f"Subtitle: {moment['text'][:50]}",
            f"",
            f"System ID: {self.get_character_name(auto_char_id)} (Similarity: {auto_similarity:.3f})",
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
        
        # Combine
        result = np.vstack([display_frame, panel])
        
        return result
    
    def _handle_key(self, key, moment, auto_char_id):
        """Handles keypress events."""
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
            # Mark as wrong, prompt for correct ID
            print(f"\nSystem identified: {self.get_character_name(auto_char_id)}")
            try:
                correct_id_str = input("Please enter the correct Character ID: ")
                correct_id = int(correct_id_str)
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
            except ValueError:
                print("❌ Invalid input, please enter a number. Skipping.")
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
        elif ord('0') <= key <= ord('9'):
            # Direct ID input
            char_id = key - ord('0')
            self.annotations[moment_key] = {
                'subtitle_id': moment['subtitle_id'],
                'start': moment['start'],
                'end': moment['end'],
                'text': moment['text'],
                'ground_truth_id': char_id,
                'system_predicted_id': auto_char_id,
                'status': 'manual'
            }
            print(f"✓ Annotated as ID: {char_id}")
            return 'next'
        
        return 'stay'
    
    def _save_annotations(self):
        """Saves the annotation results."""
        output_file = self.output_dir / 'speaking_moments_ground_truth.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Ground truth saved to: {output_file}")
        print(f"   Annotated a total of {len(self.annotations)} speaking moments")
    
    def _print_final_stats(self):
        """Displays the final statistics."""
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
    """Main program."""
    parser = argparse.ArgumentParser(
        description='Speaking Moment Ground Truth Annotation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Prepare an SRT subtitle file (provides speaking times).
  2. Prepare the system's recognition result file (.pkl or .json).
  3. Run the annotation tool.
  4. Generate the ground truth file.

Example:
  python %(prog)s \\
      --video video.mp4 \\
      --srt subtitles.srt \\
      --results retrieval_results.pkl \\
      --output ./ground_truth
        """
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
        # Step 1: Extract speaking moments
        print("\n【Step 1】Extracting Speaking Moments")
        print("-"*70)
        extractor = SpeakingMomentExtractor(args.srt)
        speaking_moments = extractor.extract_speaking_moments()
        
        if not speaking_moments:
            print("❌ No speaking moments found.")
            return 1
        
        # Step 2: Load recognition results
        print("\n【Step 2】Loading System Recognition Results")
        print("-"*70)
        loader = RetrievalResultsLoader(args.results)
        retrieval_results = loader.load_results()
        
        if not retrieval_results:
            print("❌ Failed to load retrieval results.")
            return 1

        # Step 3: Load character names (optional)
        character_names = {}
        if args.character_names:
            try:
                with open(args.character_names, 'r', encoding='utf-8') as f:
                    # Load and convert keys to integers
                    character_names_str_keys = json.load(f)
                    character_names = {int(k): v for k, v in character_names_str_keys.items()}
                print(f"✅ Loaded {len(character_names)} character names.")
            except Exception as e:
                print(f"⚠️  Could not load character names file: {e}")

        # Step 4: Run annotation
        print("\n【Step 3】Starting Annotation") # Renaming to Step 3 for user clarity
        print("-"*70)
        tool = GroundTruthAnnotationTool(
            video_path=args.video,
            speaking_moments=speaking_moments,
            retrieval_results=retrieval_results,
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