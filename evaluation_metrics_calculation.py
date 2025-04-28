import json
import os
import cv2
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

class FaceDetectionEvaluator:
    def __init__(self, ground_truth_path, detection_results_path):
        # Load ground truth (JSON file)
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
            
        # Load detection results (pickle file)
        with open(detection_results_path, 'rb') as f:  # Changed 'r' to 'rb' for binary mode
            self.detection_results = pickle.load(f)    # Changed json.load to pickle.load
        
        # Convert to frame-indexed dictionaries for easier access
        self.gt_by_frame = {frame["frame_id"]: frame for frame in self.ground_truth["frames"]}
        self.det_by_frame = self.detection_results  # Assuming this is already keyed by frame_id
        
        print(f"Loaded {len(self.gt_by_frame)} frames from ground truth")
        print(f"Loaded {len(self.det_by_frame)} frames from detection results")
        
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
        
        for frame_id, gt_frame in self.gt_by_frame.items():
            # Convert frame_id to string if detection results use string keys
            frame_key = str(frame_id)
            if frame_key not in self.det_by_frame:
                # If frame isn't in detection results, count all GT faces as false negatives
                false_negatives += len(gt_frame["faces"])
                continue
            
            # Get detected faces for this frame
            det_faces = self.det_by_frame[frame_key]
            
            # Convert GT bboxes to [x1, y1, x2, y2] format for IoU calculation
            gt_bboxes = []
            for face in gt_frame["faces"]:
                bbox = face["bbox"]
                gt_bboxes.append([
                    bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                ])
            
            # Convert detection bboxes to the same format
            det_bboxes = []
            for face in det_faces:
                bbox = face["bbox"]
                # Handle different possible bbox formats
                if len(bbox) == 4:
                    # If already in [x1, y1, x2, y2] format
                    det_bboxes.append(bbox)
                else:
                    # If in [x, y, width, height] format
                    det_bboxes.append([
                        bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                    ])
            
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
        
        # Print structure of detection results for debugging
        if len(self.det_by_frame) > 0:
            sample_frame_key = next(iter(self.det_by_frame))
            print(f"Sample detection frame structure for frame {sample_frame_key}:")
            sample_faces = self.det_by_frame[sample_frame_key]
            if len(sample_faces) > 0:
                print(f"Sample face fields: {sample_faces[0].keys()}")
        
        for frame_id, gt_frame in self.gt_by_frame.items():
            frame_key = str(frame_id)
            if frame_key not in self.det_by_frame:
                continue
            
            # Get detected faces for this frame
            det_faces = self.det_by_frame[frame_key]
            
            # Convert GT data
            gt_bboxes = []
            gt_ids = []
            for face in gt_frame["faces"]:
                bbox = face["bbox"]
                gt_bboxes.append([
                    bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                ])
                gt_ids.append(face["face_id"])
            
            # Convert detection data
            det_bboxes = []
            det_ids = []
            for face in det_faces:
                bbox = face["bbox"]
                # Handle different bbox formats
                if len(bbox) == 4:
                    det_bboxes.append(bbox)
                else:
                    det_bboxes.append([
                        bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                    ])
                
                # Check which field contains the ID information
                if "match_idx" in face:
                    det_ids.append(face["match_idx"])
                elif "face_id" in face:
                    det_ids.append(face["face_id"])
                else:
                    # Default to -1 if no ID field found
                    print(f"Warning: No ID field found in detection result for frame {frame_key}")
                    det_ids.append(-1)
            
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
                        if gt_ids[best_gt_idx] == det_ids[det_idx]:
                            correct_identifications += 1
        
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
        
        # Compute IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

if __name__ == "__main__":
    ground_truth_path = "ground_truth.json"
    # Change this to match your detection results file
    detection_results_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\enhanced_detection_results.pkl"  # From enhanced_video_annotation.py
    
    evaluator = FaceDetectionEvaluator(ground_truth_path, detection_results_path)
    
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