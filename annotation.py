import os
import cv2
import numpy as np
import tensorflow as tf
import json
from tqdm import tqdm
import facenet.src.align.detect_face as detect_face

def create_annotation_template(frames_dir, output_json_path):
    """
    Create annotation template using existing frames and MTCNN
    
    Args:
        frames_dir: Directory containing extracted frames
        output_json_path: Path to save the annotation template
    """
    # Get all frame files
    frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    frame_files.sort()  # Sort by name
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Initialize annotation data
    annotation_data = {
        "video_name": os.path.basename(os.path.dirname(frames_dir)),
        "frames": []
    }
    
    # Initialize MTCNN
    print("Initializing MTCNN face detector...")
    with tf.Session() as sess:
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        min_face_size = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        
        # Process each frame
        for frame_file in tqdm(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue
            
            # Extract frame ID from filename
            # Expected format: frame_XXXXXX.jpg
            frame_id = int(frame_file.split('_')[1].split('.')[0])
            
            # Detect faces using MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bounding_boxes, _ = detect_face.detect_face(
                frame_rgb, min_face_size, pnet, rnet, onet, threshold, factor
            )
            
            # Convert bounding boxes to the right format
            faces = []
            for i, bbox in enumerate(bounding_boxes):
                # Use int() to convert numpy data types to native Python int
                x1 = int(max(0, bbox[0]))
                y1 = int(max(0, bbox[1]))
                x2 = int(min(frame.shape[1], bbox[2]))
                y2 = int(min(frame.shape[0], bbox[3]))
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                # Assign a temporary face_id (-1)
                faces.append({
                    "face_id": -1,  # To be assigned manually
                    "bbox": [x1, y1, width, height],
                    "person_name": ""
                })
            
            # Add frame to annotations
            annotation_data["frames"].append({
                "frame_id": frame_id,
                "path": frame_path,
                "faces": faces
            })
    
    # Save annotation template
    with open(output_json_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"Annotation template saved to {output_json_path}")
    print(f"Total frames: {len(annotation_data['frames'])}")
    
    # Count total faces
    total_faces = sum(len(frame["faces"]) for frame in annotation_data["frames"])
    print(f"Total detected faces: {total_faces}")

if __name__ == "__main__":
    # Change these paths to match your project structure
    frames_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\faces"  # Directory containing the extracted frames
    output_json_path = "ground_truth_template.json"
    
    create_annotation_template(frames_dir, output_json_path)