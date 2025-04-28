import json
import os
import cv2
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

class FaceDetectionEvaluator:
    def __init__(self, ground_truth_path, detection_results_path):
        print(f"Loading ground truth from: {ground_truth_path}")
        print(f"Loading detection results from: {detection_results_path}")
        
        # Load ground truth (JSON file)
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
            
        # Load detection results (pickle file)
        with open(detection_results_path, 'rb') as f:  # Changed 'r' to 'rb' for binary mode
            self.detection_results = pickle.load(f)    # Changed json.load to pickle.load
        
        # Convert to frame-indexed dictionaries for easier access
        if "frames" in self.ground_truth:
            self.gt_by_frame = {frame["frame_id"]: frame for frame in self.ground_truth["frames"]}
            print(f"Ground truth format: Frame-based with {len(self.gt_by_frame)} frames")
        else:
            # Single-face format - convert to frame-based format
            print("Ground truth format: Single-face format, converting to frame-based")
            self.gt_by_frame = self._convert_single_face_to_frame_based()
            
        self.det_by_frame = self.detection_results  # Assuming this is already keyed by frame_id
        
        # Print detailed debug information
        print(f"Loaded {len(self.gt_by_frame)} frames from ground truth")
        print(f"Loaded {len(self.det_by_frame)} frames from detection results")
        
        # Print some sample frame IDs from both datasets
        gt_sample_keys = list(self.gt_by_frame.keys())[:5]
        det_sample_keys = list(self.det_by_frame.keys())[:5]
        print(f"Sample ground truth frame IDs: {gt_sample_keys}")
        print(f"Sample detection frame IDs: {det_sample_keys}")
        
        # Check for type mismatches
        if len(gt_sample_keys) > 0 and len(det_sample_keys) > 0:
            print(f"Ground truth frame ID type: {type(gt_sample_keys[0])}")
            print(f"Detection frame ID type: {type(det_sample_keys[0])}")
            
        # Check bbox format
        if len(self.gt_by_frame) > 0:
            sample_frame_id = next(iter(self.gt_by_frame))
            if len(self.gt_by_frame[sample_frame_id]["faces"]) > 0:
                sample_bbox = self.gt_by_frame[sample_frame_id]["faces"][0]["bbox"]
                print(f"Ground truth bbox format: {sample_bbox} (type: {type(sample_bbox)})")
                
        if len(self.det_by_frame) > 0:
            sample_frame_id = next(iter(self.det_by_frame))
            if len(self.det_by_frame[sample_frame_id]) > 0:
                sample_bbox = self.det_by_frame[sample_frame_id][0]["bbox"]
                print(f"Detection bbox format: {sample_bbox} (type: {type(sample_bbox)})")
                print(f"Sample detection face fields: {self.det_by_frame[sample_frame_id][0].keys()}")
                
    def _convert_single_face_to_frame_based(self):
        """Convert single-face format ground truth to frame-based format"""
        frames_dict = {}
        
        # Group faces by frame_id
        for face in self.ground_truth["faces"]:
            frame_id = face["frame_id"]
            if frame_id not in frames_dict:
                frames_dict[frame_id] = {"frame_id": frame_id, "faces": []}
                
            # Add face data to the frame
            x1, y1, x2, y2 = 0, 0, 0, 0  # Default values
            if "bbox" in face:
                bbox = face["bbox"]
                x1, y1, x2, y2 = bbox  # Assume [x1, y1, x2, y2] format
            elif "face_path" in face:
                # Try to extract from filename - might need adjustment based on your naming convention
                # This is a placeholder and might need to be customized
                pass
                
            frames_dict[frame_id]["faces"].append({
                "face_id": face["face_id"],
                "bbox": [x1, y1, x2-x1, y2-y1],  # Convert to [x, y, width, height] format
                "person_name": face.get("person_name", "")
            })
            
        print(f"Converted {len(self.ground_truth['faces'])} faces to {len(frames_dict)} frames")
        return frames_dict
        
    def evaluate_detection(self, iou_threshold=0.5):
        """
        Evaluate face detection performance using IoU (Intersection over Union)
        
        Args:
            iou_threshold: Threshold for considering a detection as correct
            
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # For debugging - collect mismatched frames
        missing_gt_frames = []
        missing_det_frames = []
        processed_frames = 0
        matched_frames = 0
        
        for frame_id, gt_frame in self.gt_by_frame.items():
            processed_frames += 1
            
            # Try different frame_key formats
            found_match = False
            frame_keys_to_try = [
                frame_id,            # Original format
                str(frame_id),       # String format
                int(str(frame_id)) if isinstance(frame_id, str) and str(frame_id).isdigit() else frame_id  # Integer format
            ]
            
            for frame_key in frame_keys_to_try:
                if frame_key in self.det_by_frame:
                    # Get detected faces for this frame
                    det_faces = self.det_by_frame[frame_key]
                    found_match = True
                    matched_frames += 1
                    
                    # Convert GT bboxes to [x1, y1, x2, y2] format for IoU calculation
                    gt_bboxes = []
                    for face in gt_frame["faces"]:
                        bbox = face["bbox"]
                        # Check bbox format and convert if needed
                        if len(bbox) == 4:
                            if isinstance(bbox[0], (int, float)) and isinstance(bbox[2], (int, float)):
                                if bbox[2] < 100 and bbox[3] < 100:  # Likely width/height format
                                    gt_bboxes.append([
                                        bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                                    ])
                                else:  # Likely already x1,y1,x2,y2 format
                                    gt_bboxes.append(bbox)
                    
                    # Convert detection bboxes to the same format
                    det_bboxes = []
                    for face in det_faces:
                        bbox = face["bbox"]
                        
                        # Handle different detection bbox formats
                        if isinstance(bbox, tuple) and len(bbox) == 4:
                            # Already a tuple, convert to list for consistency
                            det_bbox = list(bbox)
                            # Check if it's in x1,y1,x2,y2 format or x,y,w,h format
                            if det_bbox[2] < det_bbox[0] or det_bbox[3] < det_bbox[1]:
                                # Invalid - width/height might be negative
                                print(f"Warning: Invalid bbox detected {det_bbox}")
                                continue
                            elif det_bbox[2] < 100 and det_bbox[3] < 100:
                                # Likely width/height format
                                det_bboxes.append([
                                    det_bbox[0], det_bbox[1], 
                                    det_bbox[0] + det_bbox[2], det_bbox[1] + det_bbox[3]
                                ])
                            else:
                                # Already in x1,y1,x2,y2 format
                                det_bboxes.append(det_bbox)
                        elif isinstance(bbox, list) and len(bbox) == 4:
                            # Already a list
                            # Check if it's in x1,y1,x2,y2 format or x,y,w,h format
                            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                                # Invalid - width/height might be negative
                                print(f"Warning: Invalid bbox detected {bbox}")
                                continue
                            elif bbox[2] < 100 and bbox[3] < 100:
                                # Likely width/height format
                                det_bboxes.append([
                                    bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                                ])
                            else:
                                # Already in x1,y1,x2,y2 format
                                det_bboxes.append(bbox)
                        else:
                            # Handle unexpected formats
                            print(f"Warning: Unexpected bbox format {bbox} in frame {frame_key}")
                            continue
                    
                    # Match detections to ground truth using IoU
                    matched_gt_indices = set()
                    
                    for det_idx, det_bbox in enumerate(det_bboxes):
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for gt_idx, gt_bbox in enumerate(gt_bboxes):
                            if gt_idx in matched_gt_indices:
                                continue  # Skip already matched GT bboxes
                            
                            iou = self._compute_iou(det_bbox, gt_bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                        
                        if best_iou >= iou_threshold:
                            true_positives += 1
                            matched_gt_indices.add(best_gt_idx)
                        else:
                            false_positives += 1
                    
                    # Count unmatched ground truth as false negatives
                    false_negatives += len(gt_bboxes) - len(matched_gt_indices)
                    break  # Exit the frame_key loop once a match is found
            
            if not found_match:
                if len(gt_frame["faces"]) > 0:
                    missing_det_frames.append(frame_id)
                    # If frame isn't in detection results, count all GT faces as false negatives
                    false_negatives += len(gt_frame["faces"])
        
        # Check for any detection frames not in ground truth (additional false positives)
        for frame_key in self.det_by_frame:
            # Convert to the same type as gt_by_frame keys for comparison
            # Try different conversions
            found_match = False
            for gt_key in [frame_key, str(frame_key), int(frame_key) if isinstance(frame_key, str) and frame_key.isdigit() else frame_key]:
                if gt_key in self.gt_by_frame:
                    found_match = True
                    break
            
            if not found_match:
                # If we found a detection frame that's not in ground truth
                missing_gt_frames.append(frame_key)
                false_positives += len(self.det_by_frame[frame_key])
        
        # Print debug information
        print(f"Processed {processed_frames} frames from ground truth")
        print(f"Matched {matched_frames} frames between ground truth and detection")
        print(f"Frames in detection but not in ground truth: {len(missing_gt_frames)}")
        print(f"Frames in ground truth but not in detection: {len(missing_det_frames)}")
        if len(missing_gt_frames) > 0:
            print(f"Sample missing GT frames: {missing_gt_frames[:5]}")
        if len(missing_det_frames) > 0:
            print(f"Sample missing detection frames: {missing_det_frames[:5]}")
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def evaluate_recognition(self, iou_threshold=0.5):
        """
        Evaluate face recognition performance
        
        Args:
            iou_threshold: Threshold for considering a detection as matching a ground truth face
            
        Returns:
            Dictionary with recognition accuracy and other metrics
        """
        total_matched_faces = 0
        correct_identifications = 0
        
        # Print more debugging info
        print("\nEvaluating face recognition performance...")
        if len(self.det_by_frame) > 0:
            sample_frame_key = next(iter(self.det_by_frame))
            print(f"Sample detection frame structure for frame {sample_frame_key}:")
            sample_faces = self.det_by_frame[sample_frame_key]
            if len(sample_faces) > 0:
                print(f"Sample face fields: {sample_faces[0].keys()}")
        
        # Keep track of face ID conflicts
        id_conflicts = 0
        
        for frame_id, gt_frame in self.gt_by_frame.items():
            # Try different frame key formats
            found_match = False
            
            for frame_key in [frame_id, str(frame_id), int(str(frame_id)) if isinstance(frame_id, str) and str(frame_id).isdigit() else frame_id]:
                if frame_key not in self.det_by_frame:
                    continue
                
                found_match = True
                # Get detected faces for this frame
                det_faces = self.det_by_frame[frame_key]
                
                # Convert GT data
                gt_bboxes = []
                gt_ids = []
                for face in gt_frame["faces"]:
                    bbox = face["bbox"]
                    # Check bbox format and convert if needed
                    if len(bbox) == 4:
                        if bbox[2] < 100 and bbox[3] < 100:  # Likely width/height format
                            gt_bboxes.append([
                                bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                            ])
                        else:  # Likely already x1,y1,x2,y2 format
                            gt_bboxes.append(bbox)
                    gt_ids.append(face["face_id"])
                
                # Convert detection data
                det_bboxes = []
                det_ids = []
                for face in det_faces:
                    bbox = face["bbox"]
                    
                    # Handle different detection bbox formats
                    if isinstance(bbox, tuple) and len(bbox) == 4:
                        # Already a tuple, convert to list for consistency
                        det_bbox = list(bbox)
                        if det_bbox[2] < 100 and det_bbox[3] < 100:
                            # Likely width/height format
                            det_bboxes.append([
                                det_bbox[0], det_bbox[1], 
                                det_bbox[0] + det_bbox[2], det_bbox[1] + det_bbox[3]
                            ])
                        else:
                            # Already in x1,y1,x2,y2 format
                            det_bboxes.append(det_bbox)
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        # Already a list
                        if bbox[2] < 100 and bbox[3] < 100:
                            # Likely width/height format
                            det_bboxes.append([
                                bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                            ])
                        else:
                            # Already in x1,y1,x2,y2 format
                            det_bboxes.append(bbox)
                    else:
                        # Handle unexpected formats
                        print(f"Warning: Unexpected bbox format {bbox}")
                        continue
                    
                    # Check which field contains the ID information
                    face_id = -1
                    if "match_idx" in face:
                        face_id = face["match_idx"]
                    elif "face_id" in face:
                        face_id = face["face_id"]
                    else:
                        # Default to -1 if no ID field found
                        print(f"Warning: No ID field found in detection result for frame {frame_key}")
                    
                    det_ids.append(face_id)
                
                # Match detections to ground truth
                for det_idx, det_bbox in enumerate(det_bboxes):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_bbox in enumerate(gt_bboxes):
                        iou = self._compute_iou(det_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold:
                        total_matched_faces += 1
                        # Check if IDs match
                        if best_gt_idx >= 0 and det_idx < len(det_ids):
                            gt_id = gt_ids[best_gt_idx]
                            det_id = det_ids[det_idx]
                            
                            if gt_id == det_id:
                                correct_identifications += 1
                            else:
                                # ID conflict - helpful for debugging
                                id_conflicts += 1
                                if id_conflicts <= 5:  # Limit the number of conflicts to report
                                    print(f"ID conflict: Ground truth ID {gt_id} vs Detection ID {det_id} in frame {frame_id}")
                break  # Exit frame_key loop once a match is found
        
        # Print debug information
        print(f"Total ID conflicts: {id_conflicts}")
        
        # Calculate recognition accuracy
        recognition_accuracy = correct_identifications / total_matched_faces if total_matched_faces > 0 else 0
        
        return {
            "recognition_accuracy": recognition_accuracy,
            "correct_identifications": correct_identifications,
            "total_matched_faces": total_matched_faces
        }
    
    def _compute_iou(self, box1, box2):
        """
        Compute Intersection over Union for two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Sanitize inputs
        try:
            box1 = [float(val) for val in box1]
            box2 = [float(val) for val in box2]
        except Exception as e:
            print(f"Error converting boxes to float: {e}")
            print(f"Box1: {box1}, Box2: {box2}")
            return 0.0
        
        # Compute intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Compute areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Check for invalid areas
        if box1_area <= 0 or box2_area <= 0:
            print(f"Warning: Invalid box area detected. Box1: {box1_area}, Box2: {box2_area}")
            return 0.0
        
        # Compute IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def convert_ground_truth_formats(self, output_path=None):
        """
        Convert between different ground truth formats
        
        Args:
            output_path: Path to save the converted ground truth file
            
        Returns:
            The converted ground truth data
        """
        # If ground truth is frame-based, convert to single-face format
        if "frames" in self.ground_truth:
            print("Converting from frame-based to single-face format...")
            single_face_data = {
                "video_name": self.ground_truth.get("video_name", "unknown"),
                "faces": []
            }
            
            for frame in self.ground_truth["frames"]:
                frame_id = frame["frame_id"]
                for i, face in enumerate(frame["faces"]):
                    # Skip faces without valid IDs
                    if face["face_id"] < 0:
                        continue
                        
                    single_face_data["faces"].append({
                        "face_path": face.get("face_path", f"frame_{frame_id}_face_{i}.jpg"),
                        "frame_id": frame_id,
                        "face_idx": i,
                        "face_id": face["face_id"],
                        "person_name": face.get("person_name", f"Person {face['face_id']}")
                    })
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(single_face_data, f, indent=2)
                print(f"Converted single-face format saved to {output_path}")
            
            return single_face_data
            
        # If ground truth is single-face format, convert to frame-based format
        else:
            print("Converting from single-face format to frame-based format...")
            frame_based_data = {
                "video_name": self.ground_truth.get("video_name", "unknown"),
                "frames": []
            }
            
            # Group faces by frame_id
            frames_dict = {}
            for face in self.ground_truth["faces"]:
                frame_id = face["frame_id"]
                if frame_id not in frames_dict:
                    frames_dict[frame_id] = {"frame_id": frame_id, "faces": []}
                
                # Try to get bbox from face_path - this is a placeholder
                # and might need to be adjusted based on your implementation
                face_path = face.get("face_path", "")
                
                frames_dict[frame_id]["faces"].append({
                    "face_id": face["face_id"],
                    "bbox": [0, 0, 0, 0],  # Placeholder
                    "person_name": face.get("person_name", ""),
                    "face_path": face_path
                })
            
            # Convert dictionary to list
            frame_based_data["frames"] = list(frames_dict.values())
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(frame_based_data, f, indent=2)
                print(f"Converted frame-based format saved to {output_path}")
            
            return frame_based_data

if __name__ == "__main__":
    ground_truth_path = "ground_truth.json"
    # Try to find detection results path
    detection_results_paths = [
        r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\enhanced_detection_results.pkl",
        r"result\enhanced_detection_results.pkl",
        "enhanced_detection_results.pkl"
    ]
    
    detection_results_path = None
    for path in detection_results_paths:
        if os.path.exists(path):
            detection_results_path = path
            break
    
    if detection_results_path is None:
        print("Could not find detection results file. Please specify the path manually.")
        exit(1)
    
    print(f"Using detection results from: {detection_results_path}")
    
    evaluator = FaceDetectionEvaluator(ground_truth_path, detection_results_path)
    
    # Try to convert ground truth format if needed
    if "frames" not in evaluator.ground_truth and "faces" in evaluator.ground_truth:
        print("Converting single-face ground truth to frame-based format for evaluation...")
        evaluator.convert_ground_truth_formats("ground_truth_frame_based.json")
    
    # Evaluate detection performance
    detection_metrics = evaluator.evaluate_detection()
    print("\nFace Detection Performance:")
    print(f"Precision: {detection_metrics['precision']:.4f}")
    print(f"Recall: {detection_metrics['recall']:.4f}")
    print(f"F1 Score: {detection_metrics['f1_score']:.4f}")
    print(f"True Positives: {detection_metrics['true_positives']}")
    print(f"False Positives: {detection_metrics['false_positives']}")
    print(f"False Negatives: {detection_metrics['false_negatives']}")
    
    # Evaluate recognition performance
    recognition_metrics = evaluator.evaluate_recognition()
    print("\nFace Recognition Performance:")
    print(f"Recognition Accuracy: {recognition_metrics['recognition_accuracy']:.4f}")
    print(f"Correct Identifications: {recognition_metrics['correct_identifications']}")
    print(f"Total Matched Faces: {recognition_metrics['total_matched_faces']}")