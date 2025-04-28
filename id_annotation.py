import cv2
import json
import os
import numpy as np
from tkinter import Tk, simpledialog, messagebox

class FaceIDAssigner:
    def __init__(self, template_json_path):
        # Load the template JSON file
        with open(template_json_path, 'r') as f:
            self.annotation_data = json.load(f)
        
        self.current_frame_idx = 0
        self.current_person_id = 0
        self.person_colors = {}  # To maintain consistent colors for person IDs
        
    def start_annotation(self):
        # Initialize tkinter for dialogs
        self.root = Tk()
        self.root.withdraw()  # Hide the main window
        
        while self.current_frame_idx < len(self.annotation_data["frames"]):
            frame_info = self.annotation_data["frames"][self.current_frame_idx]
            frame_path = frame_info["path"]
            
            if not os.path.exists(frame_path):
                print(f"Warning: Frame file {frame_path} not found")
                self.current_frame_idx += 1
                continue
                
            # Read and display the frame
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Error: Could not read image {frame_path}")
                self.current_frame_idx += 1
                continue
            
            # Display MTCNN-detected faces
            display_img = img.copy()
            for i, face in enumerate(frame_info["faces"]):
                bbox = face["bbox"]
                face_id = face["face_id"]
                
                # Get color based on face_id
                if face_id == -1:  # Unassigned
                    color = (0, 0, 255)  # Red for unassigned
                else:
                    # Get a consistent color for this person
                    if face_id not in self.person_colors:
                        # Generate a random color for this person
                        color = tuple(map(int, np.random.randint(0, 255, 3).tolist()))
                        self.person_colors[face_id] = color
                    else:
                        color = self.person_colors[face_id]
                
                # Draw the bounding box
                cv2.rectangle(display_img, 
                             (bbox[0], bbox[1]), 
                             (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                             color, 2)
                
                # Draw the face number and ID
                if face_id == -1:
                    face_label = f"#{i}: Unassigned"
                else:
                    face_label = f"#{i}: ID {face_id}"
                
                cv2.putText(display_img, face_label, 
                           (bbox[0], bbox[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Set up window
            window_name = f"Frame {frame_info['frame_id']} - Assign IDs"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, display_img)
            
            # Instructions
            print("\nFrame", frame_info['frame_id'], "- Instructions:")
            print("- Enter face number (0-based index) to assign an ID")
            print("- Press 'n' to move to the next frame")
            print("- Press 'p' to move to the previous frame")
            print("- Press 's' to save annotations")
            print("- Press 'q' to quit")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key >= ord('0') and key <= ord('9'):
                    # Assign ID to a face
                    face_idx = key - ord('0')
                    if face_idx < len(frame_info["faces"]):
                        # Ask for person ID
                        person_id = simpledialog.askinteger("Person ID", 
                                                           f"Enter Person ID for face #{face_idx}",
                                                           initialvalue=self.current_person_id,
                                                           parent=self.root)
                        
                        if person_id is not None:
                            self.current_person_id = max(self.current_person_id, person_id)
                            
                            # Update the face ID
                            frame_info["faces"][face_idx]["face_id"] = person_id
                            frame_info["faces"][face_idx]["person_name"] = f"Person {person_id}"
                            
                            # Update display
                            display_img = img.copy()
                            for i, face in enumerate(frame_info["faces"]):
                                bbox = face["bbox"]
                                face_id = face["face_id"]
                                
                                if face_id == -1:
                                    color = (0, 0, 255)  # Red for unassigned
                                else:
                                    if face_id not in self.person_colors:
                                        color = tuple(map(int, np.random.randint(0, 255, 3).tolist()))
                                        self.person_colors[face_id] = color
                                    else:
                                        color = self.person_colors[face_id]
                                
                                cv2.rectangle(display_img, 
                                             (bbox[0], bbox[1]), 
                                             (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                             color, 2)
                                
                                if face_id == -1:
                                    face_label = f"#{i}: Unassigned"
                                else:
                                    face_label = f"#{i}: ID {face_id}"
                                
                                cv2.putText(display_img, face_label, 
                                           (bbox[0], bbox[1] - 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            cv2.imshow(window_name, display_img)
                            
                            # Save after each assignment
                            self._save_annotations()
                    else:
                        print(f"No face with index {face_idx}")
                        
                elif key == ord('n'):  # Next frame
                    self.current_frame_idx += 1
                    break
                    
                elif key == ord('p'):  # Previous frame
                    self.current_frame_idx = max(0, self.current_frame_idx - 1)
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
            
        # After all frames are processed
        messagebox.showinfo("Complete", "All frames have been processed!")
        self.root.destroy()
    
    def _save_annotations(self):
        # Save annotations to the JSON file
        output_path = "ground_truth.json"
        with open(output_path, 'w') as f:
            json.dump(self.annotation_data, f, indent=2)

if __name__ == "__main__":
    template_path = "ground_truth_template.json"
    id_assigner = FaceIDAssigner(template_path)
    id_assigner.start_annotation()