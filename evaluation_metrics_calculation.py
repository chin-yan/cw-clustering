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
        with open(detection_results_path, 'rb') as f:
            detection_data = pickle.load(f)
            
        # Handle different detection result formats
        if isinstance(detection_data, dict) and 'frame_detection_results' in detection_data:
            self.detection_results = detection_data['frame_detection_results']
            self.detection_coordinate_system = detection_data.get('coordinate_system')
        else:
            self.detection_results = detection_data
            self.detection_coordinate_system = None
        
        # Extract coordinate system from ground truth
        self.ground_truth_coordinate_system = self.ground_truth.get('coordinate_system')
        
        # Print coordinate system info
        if self.ground_truth_coordinate_system:
            print(f"Ground truth coordinate system: {self.ground_truth_coordinate_system['width']}x{self.ground_truth_coordinate_system['height']}")
        if self.detection_coordinate_system:
            print(f"Detection coordinate system: {self.detection_coordinate_system['width']}x{self.detection_coordinate_system['height']}")
        
        # Calculate scale factors if both coordinate systems are available
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        if self.ground_truth_coordinate_system and self.detection_coordinate_system:
            gt_width = self.ground_truth_coordinate_system['width']
            gt_height = self.ground_truth_coordinate_system['height']
            det_width = self.detection_coordinate_system['width']
            det_height = self.detection_coordinate_system['height']
            
            if gt_width > 0 and det_width > 0:
                self.scale_x = det_width / gt_width
                self.scale_y = det_height / gt_height
                print(f"Coordinate scale factors: x={self.scale_x:.4f}, y={self.scale_y:.4f}")
        
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
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox  # Assume [x1, y1, x2, y2] format
            elif "face_path" in face:
                # Try to extract from filename - might need adjustment based on your naming convention
                # This is a placeholder and might need to be customized
                pass
                
            frames_dict[frame_id]["faces"].append({
                "face_id": face["face_id"],
                "bbox": [x1, y1, x2, y2],  # Keep in [x1, y1, x2, y2] format
                "person_name": face.get("person_name", ""),
                "face_path": face.get("face_path", "")
            })
            
        print(f"Converted {len(self.ground_truth['faces'])} faces to {len(frames_dict)} frames")
        return frames_dict
    
    def _find_matching_frame_id(self, frame_id, target_dict):
        """Find a matching frame ID in the target dictionary, trying different formats"""
        # Try different frame ID formats
        formats_to_try = [
            frame_id,                               # Original format
            str(frame_id),                          # String format 
            int(str(frame_id)) if isinstance(frame_id, str) else frame_id,  # Integer format
        ]
            
        # Try exact match first
        for frame_key in formats_to_try:
            if frame_key in target_dict:
                return frame_key
                
        # If no exact match, try to find the closest frame
        if isinstance(frame_id, int) or (isinstance(frame_id, str) and frame_id.isdigit()):
            frame_num = int(frame_id)
            closest_diff = float('inf')
            closest_key = None
            
            for key in target_dict.keys():
                if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                    key_num = int(key)
                    diff = abs(key_num - frame_num)
                    if diff < closest_diff and diff <= 5:  # Accept frames within 5 frames difference
                        closest_diff = diff
                        closest_key = key
            
            if closest_key is not None:
                return closest_key
                
        return None
    
    def _standardize_bbox(self, bbox, source="gt"):
        """
        Standardize bounding box to [x1, y1, x2, y2] format and scale to detection coordinate system
        
        Args:
            bbox: The bounding box in various formats
            source: Source of the bbox ('gt' or 'det')
            
        Returns:
            Standardized bbox in [x1, y1, x2, y2] format
        """
        # Convert bbox to list format
        if isinstance(bbox, tuple):
            bbox = list(bbox)
        
        # Check if it's a valid bbox
        if not bbox or len(bbox) != 4:
            return None
            
        # Determine if bbox is in [x1, y1, x2, y2] or [x, y, width, height] format
        # Heuristic: If the third value (width or x2) is smaller than the first value (x)
        # or the fourth value (height or y2) is smaller than the second value (y),
        # then it's likely [x, y, width, height]
        is_xywh = False
        if bbox[2] < 100 and bbox[3] < 100:  # Small values, likely width/height
            is_xywh = True
        
        # Convert to [x1, y1, x2, y2] format
        if is_xywh:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
        else:
            x1, y1, x2, y2 = bbox
        
        # Scale coordinates if needed
        if source == "gt" and self.scale_x != 1.0:
            # Scale ground truth to match detection coordinates
            x1 = int(x1 * self.scale_x)
            y1 = int(y1 * self.scale_y)
            x2 = int(x2 * self.scale_x)
            y2 = int(y2 * self.scale_y)
        elif source == "det" and self.scale_x != 1.0:
            # Scale detection to match ground truth coordinates (inverse)
            x1 = int(x1 / self.scale_x)
            y1 = int(y1 / self.scale_y)
            x2 = int(x2 / self.scale_x)
            y2 = int(y2 / self.scale_y)
        
        return [x1, y1, x2, y2]
        
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
            
            # Find matching frame ID in detection results
            matched_frame_id = self._find_matching_frame_id(frame_id, self.det_by_frame)
            
            if matched_frame_id is not None:
                # Get detected faces for this frame
                det_faces = self.det_by_frame[matched_frame_id]
                matched_frames += 1
                
                # Convert GT bboxes to standardized format
                gt_bboxes_std = []
                for face in gt_frame["faces"]:
                    bbox = face["bbox"]
                    std_bbox = self._standardize_bbox(bbox, source="gt")
                    if std_bbox:
                        gt_bboxes_std.append(std_bbox)
                
                # Convert detection bboxes to standardized format
                det_bboxes_std = []
                for face in det_faces:
                    # Use standardized_bbox if available, otherwise use bbox
                    bbox = face.get("standardized_bbox", face["bbox"])
                    std_bbox = self._standardize_bbox(bbox, source="det")
                    if std_bbox:
                        det_bboxes_std.append(std_bbox)
                
                # Print some debug info for the first few frames
                if processed_frames <= 5:
                    print(f"Frame {frame_id}: GT bboxes: {gt_bboxes_std}")
                    print(f"Frame {matched_frame_id}: Det bboxes: {det_bboxes_std}")
                
                # Match detections to ground truth using IoU
                matched_gt_indices = set()
                
                for det_idx, det_bbox in enumerate(det_bboxes_std):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_bbox in enumerate(gt_bboxes_std):
                        if gt_idx in matched_gt_indices:
                            continue  # Skip already matched GT bboxes
                        
                        iou = self._compute_iou(det_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold:
                        true_positives += 1
                        matched_gt_indices.add(best_gt_idx)
                        if processed_frames <= 5:
                            print(f"Match found: Det {det_idx} with GT {best_gt_idx}, IoU: {best_iou:.3f}")
                    else:
                        false_positives += 1
                        if processed_frames <= 5:
                            print(f"No match for Det {det_idx}, best IoU: {best_iou:.3f}")
                
                # Count unmatched ground truth as false negatives
                false_negatives += len(gt_bboxes_std) - len(matched_gt_indices)
                if processed_frames <= 5 and len(gt_bboxes_std) > len(matched_gt_indices):
                    unmatched = [i for i in range(len(gt_bboxes_std)) if i not in matched_gt_indices]
                    print(f"Unmatched GT bboxes: {unmatched}")
            else:
                if len(gt_frame["faces"]) > 0:
                    missing_det_frames.append(frame_id)
                    # If frame isn't in detection results, count all GT faces as false negatives
                    false_negatives += len(gt_frame["faces"])
        
        # Check for any detection frames not in ground truth (additional false positives)
        for frame_key in self.det_by_frame:
            # Check if this detection frame matches any ground truth frame
            matched_gt_id = self._find_matching_frame_id(frame_key, self.gt_by_frame)
            
            if matched_gt_id is None:
                # If we found a detection frame that's not in ground truth
                missing_gt_frames.append(frame_key)
        
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
        
        # Create a mapping from match_idx to ground truth face_id
        center_to_gt_mapping = {}
        
        # Keep track of face ID conflicts
        id_conflicts = 0
        detailed_conflicts = []
        
        for frame_id, gt_frame in self.gt_by_frame.items():
            # Find matching frame ID in detection results
            matched_frame_id = self._find_matching_frame_id(frame_id, self.det_by_frame)
            
            if matched_frame_id is None:
                continue
                
            # Get detected faces for this frame
            det_faces = self.det_by_frame[matched_frame_id]
            
            # Convert GT bboxes to standardized format
            gt_bboxes_std = []
            gt_ids = []
            for face in gt_frame["faces"]:
                bbox = face["bbox"]
                std_bbox = self._standardize_bbox(bbox, source="gt")
                if std_bbox:
                    gt_bboxes_std.append(std_bbox)
                    gt_ids.append(face["face_id"])
            
            # Convert detection bboxes to standardized format
            det_bboxes_std = []
            det_ids = []
            for face in det_faces:
                # Use standardized_bbox if available, otherwise use bbox
                bbox = face.get("standardized_bbox", face["bbox"])
                std_bbox = self._standardize_bbox(bbox, source="det")
                if std_bbox:
                    det_bboxes_std.append(std_bbox)
                    # Check which field contains the ID information
                    face_id = -1
                    if "match_idx" in face:
                        face_id = face["match_idx"]
                    elif "face_id" in face:
                        face_id = face["face_id"]
                    det_ids.append(face_id)
            
            # Match detections to ground truth
            for det_idx, det_bbox in enumerate(det_bboxes_std):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_bbox in enumerate(gt_bboxes_std):
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
                        
                        # Update center to ground truth mapping
                        if det_id not in center_to_gt_mapping:
                            center_to_gt_mapping[det_id] = gt_id
                        
                        # If the mapping exists but differs, use the most common mapping
                        if center_to_gt_mapping[det_id] == gt_id:
                            correct_identifications += 1
                        else:
                            # ID conflict - helpful for debugging
                            id_conflicts += 1
                            detailed_conflicts.append((det_id, gt_id, center_to_gt_mapping[det_id]))
                            if id_conflicts <= 5:  # Limit the number of conflicts to report
                                print(f"ID conflict: Detection ID {det_id} matched with GT ID {gt_id}, but previous mapping was {center_to_gt_mapping[det_id]}")
        
        # Print debug information
        print(f"Total ID conflicts: {id_conflicts}")
        print(f"Center to GT mapping: {center_to_gt_mapping}")
        
        # Calculate recognition accuracy
        recognition_accuracy = correct_identifications / total_matched_faces if total_matched_faces > 0 else 0
        
        return {
            "recognition_accuracy": recognition_accuracy,
            "correct_identifications": correct_identifications,
            "total_matched_faces": total_matched_faces,
            "center_to_gt_mapping": center_to_gt_mapping
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
    
    # Evaluate detection performance
    detection_metrics = evaluator.evaluate_detection(iou_threshold=0.5)
    print("\nFace Detection Performance:")
    print(f"Precision: {detection_metrics['precision']:.4f}")
    print(f"Recall: {detection_metrics['recall']:.4f}")
    print(f"F1 Score: {detection_metrics['f1_score']:.4f}")
    print(f"True Positives: {detection_metrics['true_positives']}")
    print(f"False Positives: {detection_metrics['false_positives']}")
    print(f"False Negatives: {detection_metrics['false_negatives']}")
    
    # Evaluate recognition performance
    recognition_metrics = evaluator.evaluate_recognition(iou_threshold=0.5)
    print("\nFace Recognition Performance:")
    print(f"Recognition Accuracy: {recognition_metrics['recognition_accuracy']:.4f}")
    print(f"Correct Identifications: {recognition_metrics['correct_identifications']}")
    print(f"Total Matched Faces: {recognition_metrics['total_matched_faces']}")