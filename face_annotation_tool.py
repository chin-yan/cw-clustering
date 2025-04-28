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
    else:
        # If the format doesn't match, return defaults
        return -1, -1

def create_face_annotation_template(faces_dir, output_json_path):
    """
    Create annotation template using only individual face images
    
    Args:
        faces_dir: Directory containing extracted face images
        output_json_path: Path to save the annotation template
    """
    # Get all face files
    face_files = [f for f in os.listdir(faces_dir) if f.endswith(".jpg") and "face_" in f]
    face_files.sort()  # Sort by name
    
    print(f"Found {len(face_files)} face images in {faces_dir}")
    
    # Initialize annotation data
    annotation_data = {
        "video_name": os.path.basename(os.path.dirname(faces_dir)),
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

class FaceIDAssigner:
    def __init__(self, template_json_path):
        # Load the template JSON file
        with open(template_json_path, 'r') as f:
            self.annotation_data = json.load(f)
        
        self.current_face_idx = 0
        self.current_person_id = 0
        self.person_colors = {}  # To maintain consistent colors for person IDs
        
    def start_annotation(self):
        # Initialize tkinter for dialogs
        self.root = Tk()
        self.root.withdraw()  # Hide the main window
        
        while self.current_face_idx < len(self.annotation_data["faces"]):
            face_info = self.annotation_data["faces"][self.current_face_idx]
            face_path = face_info["face_path"]
            
            if not os.path.exists(face_path):
                print(f"Warning: Face image file {face_path} not found")
                self.current_face_idx += 1
                continue
                
            # Read and display the face
            img = cv2.imread(face_path)
            if img is None:
                print(f"Error: Could not read image {face_path}")
                self.current_face_idx += 1
                continue
            
            # Display information about this face
            face_id = face_info["face_id"]
            if face_id == -1:
                color = (0, 0, 255)  # Red for unassigned
                face_text = "Unassigned"
            else:
                # Get a consistent color for this person
                if face_id not in self.person_colors:
                    # Generate a random color for this person
                    color = tuple(map(int, np.random.randint(0, 255, 3).tolist()))
                    self.person_colors[face_id] = color
                else:
                    color = self.person_colors[face_id]
                face_text = f"ID: {face_id}"
            
            # Add text to the image
            display_img = img.copy()
            cv2.putText(display_img, face_text, 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Set up window
            window_name = f"Face {self.current_face_idx+1}/{len(self.annotation_data['faces'])} - Frame {face_info['frame_id']} Face {face_info['face_idx']}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, display_img)
            
            # Instructions
            print("\nFace", self.current_face_idx+1, "of", len(self.annotation_data["faces"]), "- Instructions:")
            print("- Press 'i' to assign an ID to this face")
            print("- Press 'n' to move to the next face")
            print("- Press 'p' to move to the previous face")
            print("- Press 's' to save annotations")
            print("- Press 'q' to quit")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('i'):
                    # Assign ID to the face
                    person_id = simpledialog.askinteger("Person ID", 
                                                      "Enter Person ID (use same ID for same person)",
                                                      initialvalue=self.current_person_id,
                                                      parent=self.root)
                    
                    if person_id is not None:
                        self.current_person_id = max(self.current_person_id, person_id)
                        
                        # Update the face ID
                        face_info["face_id"] = person_id
                        face_info["person_name"] = f"Person {person_id}"
                        
                        # Update display
                        display_img = img.copy()
                        
                        if person_id not in self.person_colors:
                            color = tuple(map(int, np.random.randint(0, 255, 3).tolist()))
                            self.person_colors[person_id] = color
                        else:
                            color = self.person_colors[person_id]
                        
                        cv2.putText(display_img, f"ID: {person_id}", 
                                   (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        cv2.imshow(window_name, display_img)
                        
                        # Save after each assignment
                        self._save_annotations()
                        
                elif key == ord('n'):  # Next face
                    self.current_face_idx += 1
                    break
                    
                elif key == ord('p'):  # Previous face
                    self.current_face_idx = max(0, self.current_face_idx - 1)
                    break
                    
                elif key == ord('s'):  # Save annotations
                    self._save_annotations()
                    print("Annotations saved")
                    
                elif key == ord('q'):  # Quit
                    self._save_annotations()
                    cv2.destroyAllWindows()
                    self.root.destroy()
                    return
            
            cv2.destroyWindow(window_name)
            
        # After all faces are processed
        messagebox.showinfo("Complete", "All faces have been processed!")
        self.root.destroy()
    
    def _save_annotations(self):
        # Save annotations to the JSON file
        output_path = "ground_truth.json"
        with open(output_path, 'w') as f:
            json.dump(self.annotation_data, f, indent=2)

if __name__ == "__main__":
    # Directory containing face images (single faces)
    faces_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\faces"
    
    # Create annotation template
    template_path = "single_face_template.json"
    create_face_annotation_template(faces_dir, template_path)
    
    # Start manual ID assignment
    id_assigner = FaceIDAssigner(template_path)
    id_assigner.start_annotation()