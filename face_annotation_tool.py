import os
import cv2
import numpy as np
import re
import json
from tkinter import Tk, simpledialog, messagebox
import matplotlib.pyplot as plt

def extract_frame_info(image_path):
    """
    Extract frame number and face index from image path
    
    Args:
        image_path: Path to the face image
        
    Returns:
        Frame number and face index
    """
    # Extract the base filename
    basename = os.path.basename(image_path)
    # Expected format: frame_XXXXXX_face_Y.jpg
    match = re.match(r'frame_(\d+)_face_(\d+)\.jpg', basename)
    
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    
    # Also try to match sideface pattern
    match = re.match(r'frame_(\d+)_sideface_(\d+)\.jpg', basename)
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    
    # If the format doesn't match, return defaults
    return -1, -1

def create_face_annotation_template(faces_dir, frames_dir, output_json_path, video_path=None):
    """
    Create annotation template using individual face images and store original video dimensions
    
    Args:
        faces_dir: Directory containing extracted face images
        frames_dir: Directory containing frame images
        output_json_path: Path to save the annotation template
        video_path: Optional path to the original video file to extract dimensions
    """
    # Get all face files
    face_files = [f for f in os.listdir(faces_dir) if f.endswith(".jpg") and ("face_" in f or "sideface_" in f)]
    face_files.sort()  # Sort by name
    
    print(f"Found {len(face_files)} face images in {faces_dir}")
    
    # Get video dimensions
    video_width = 0
    video_height = 0
    
    if video_path and os.path.exists(video_path):
        # Extract original video dimensions
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"Original video dimensions: {video_width}x{video_height}")
    
    # If video not provided, try to get dimensions from a frame
    if video_width == 0 or video_height == 0:
        frame_files = os.listdir(frames_dir)
        if frame_files:
            sample_frame = os.path.join(frames_dir, frame_files[0])
            frame = cv2.imread(sample_frame)
            if frame is not None:
                video_height, video_width = frame.shape[:2]
                print(f"Extracted frame dimensions: {video_width}x{video_height}")
    
    # Initialize annotation data with coordinate system metadata
    annotation_data = {
        "video_name": os.path.basename(os.path.dirname(faces_dir)),
        "coordinate_system": {
            "width": video_width,
            "height": video_height
        },
        "faces": []
    }
    
    # Process each face image
    for face_file in face_files:
        face_path = os.path.join(faces_dir, face_file)
        face_img = cv2.imread(face_path)
        
        if face_img is None:
            print(f"Warning: Could not read face image {face_path}")
            continue
        
        # Extract frame and face information
        frame_id, face_idx = extract_frame_info(face_path)
        if frame_id == -1:
            print(f"Warning: Could not extract frame info from {face_file}")
            continue
        
        # Add face to annotations
        annotation_data["faces"].append({
            "face_path": face_path,
            "frame_id": frame_id,
            "face_idx": face_idx,
            "face_id": -1,  # To be assigned manually
            "person_name": ""
        })
    
    # Save annotation template
    with open(output_json_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"Annotation template saved to {output_json_path}")
    print(f"Total faces: {len(annotation_data['faces'])}")
    print(f"Coordinate system: {video_width}x{video_height}")

class FaceIDAssigner:
    def __init__(self, template_json_path):
        # Load the template JSON file
        with open(template_json_path, 'r') as f:
            self.annotation_data = json.load(f)
        
        self.current_face_idx = 0
        self.current_person_id = 0
        self.person_colors = {}  # To maintain consistent colors for person IDs
        
        # Store coordinate system info
        self.coordinate_system = self.annotation_data.get("coordinate_system", {"width": 0, "height": 0})
        print(f"Using coordinate system: {self.coordinate_system['width']}x{self.coordinate_system['height']}")
        
    def start_annotation(self):
        # ... (rest of the code remains the same)
        pass
    
    def _save_annotations(self):
        # Save annotations to the JSON file
        output_path = "ground_truth.json"
        with open(output_path, 'w') as f:
            json.dump(self.annotation_data, f, indent=2)

if __name__ == "__main__":
    # Directory containing face images (single faces)
    faces_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\faces"
    frames_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\frames"
    video_path = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    
    # Create annotation template with video dimensions
    template_path = "single_face_template.json"
    create_face_annotation_template(faces_dir, frames_dir, template_path, video_path)
    
    # Start manual ID assignment
    id_assigner = FaceIDAssigner(template_path)
    id_assigner.start_annotation()