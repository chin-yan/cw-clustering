# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import json
import pickle
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleEvaluator:
    """
    A simplified evaluator for face detection and recognition tasks
    with consistent coordinate system handling
    """
    
    def __init__(self, ground_truth_path, detection_results_path):
        """
        Initialize the evaluator
        
        Args:
            ground_truth_path: Path to ground truth JSON file
            detection_results_path: Path to detection results pickle file
        """
        print(f"Loading ground truth from: {ground_truth_path}")
        print(f"Loading detection results from: {detection_results_path}")
        
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.detection_results = self._load_detection_results(detection_results_path)
        
        # Extract and standardize coordinate system
        self.coordinate_system = self._get_unified_coordinate_system()
        
        print(f"Using unified coordinate system: {self.coordinate_system['width']}x{self.coordinate_system['height']}")
    
    def _load_ground_truth(self, path):
        """
        Load and standardize ground truth data
        
        Args:
            path: Path to ground truth JSON file
            
        Returns:
            Standardized ground truth data
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Extract coordinate system if available
        coordinate_system = data.get('coordinate_system', None)
        if not coordinate_system and 'video_name' in data:
            # Default coordinate system if not specified
            coordinate_system = {'width': 1920, 'height': 1080}
            
        # Convert to frame-based format if needed
        frames_dict = {}
        
        # Check if data is already in frame-based format
        if 'frames' in data:
            # Already in frame-based format
            for frame in data['frames']:
                frame_id = frame['frame_id']
                frames_dict[frame_id] = frame
        else:
            # Convert from face-based format
            for face in data.get('faces', []):
                frame_id = face.get('frame_id', -1)
                if frame_id == -1:
                    continue
                    
                if frame_id not in frames_dict:
                    frames_dict[frame_id] = {
                        'frame_id': frame_id,
                        'faces': []
                    }
                
                # Standardize bbox format
                if 'bbox' in face:
                    bbox = face['bbox']
                    std_bbox = self._standardize_bbox_format(bbox)
                else:
                    # If no bbox, use default (will be filtered out later)
                    std_bbox = [0, 0, 10, 10]
                
                # Add face to frame
                frames_dict[frame_id]['faces'].append({
                    'face_id': face.get('face_id', -1),
                    'bbox': std_bbox,
                    'person_name': face.get('person_name', ''),
                    'face_path': face.get('face_path', '')
                })
        
        return {
            'coordinate_system': coordinate_system,
            'frames': frames_dict
        }
    
    def _load_detection_results(self, path):
        """
        Load and standardize detection results
        
        Args:
            path: Path to detection results pickle file
            
        Returns:
            Standardized detection results
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Detection results file not found: {path}")
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract coordinate system and frame results
        coordinate_system = None
        frame_results = {}
        
        if isinstance(data, dict):
            # Check different possible formats
            if 'coordinate_system' in data:
                coordinate_system = data['coordinate_system']
            
            # Extract frame results
            if 'frame_detection_results' in data:
                frame_results = data['frame_detection_results']
            elif 'by_frame' in data:
                frame_results = data['by_frame']
            elif 'results' in data:
                # Try to convert center-based results to frame-based
                # This is a placeholder and might need customization
                pass
        
        return {
            'coordinate_system': coordinate_system,
            'frames': frame_results
        }
    
    def _get_unified_coordinate_system(self):
        """
        Determine the unified coordinate system to use
        
        Returns:
            Coordinate system dictionary with 'width' and 'height'
        """
        # Prioritize ground truth coordinate system
        gt_coord = self.ground_truth.get('coordinate_system')
        if gt_coord and gt_coord.get('width', 0) > 0:
            return gt_coord
        
        # Otherwise use detection coordinate system
        det_coord = self.detection_results.get('coordinate_system')
        if det_coord and det_coord.get('width', 0) > 0:
            return det_coord
        
        # Default coordinate system if neither is available
        return {'width': 1920, 'height': 1080}
    
    def _standardize_bbox_format(self, bbox):
        """
        Standardize bounding box to [x1, y1, x2, y2] format
        
        Args:
            bbox: Bounding box in any format
            
        Returns:
            Standardized bbox in [x1, y1, x2, y2] format
        """
        if not bbox or len(bbox) != 4:
            return [0, 0, 10, 10]  # Default invalid bbox
            
        # Convert to list if tuple
        if isinstance(bbox, tuple):
            bbox = list(bbox)
            
        # Check if it's [x, y, width, height] format
        # If the 3rd value is less than 100, it's likely width not x2
        if bbox[2] < 100 and bbox[3] < 100:
            # Convert [x, y, width, height] to [x1, y1, x2, y2]
            return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        
        # Already in [x1, y1, x2, y2] format
        return bbox
    
    def _convert_bbox_to_unified_coordinates(self, bbox, source_coord):
        """
        Convert bbox to unified coordinate system
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            source_coord: Source coordinate system
            
        Returns:
            Bbox in unified coordinate system
        """
        if not source_coord or not self.coordinate_system:
            return bbox
            
        # Calculate scale factors
        scale_x = self.coordinate_system['width'] / source_coord['width']
        scale_y = self.coordinate_system['height'] / source_coord['height']
        
        # Apply scaling
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)
        
        return [x1, y1, x2, y2]
    
    def _compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two boxes
        
        Args:
            box1, box2: Bounding boxes in [x1, y1, x2, y2] format
            
        Returns:
            IoU value (0.0 to 1.0)
        """
        # Ensure boxes are valid
        if not box1 or not box2:
            return 0.0
            
        # Convert to float for precision
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]
        
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate box areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Handle invalid boxes
        if area1 <= 0 or area2 <= 0:
            return 0.0
            
        # Calculate IoU
        iou = intersection / float(area1 + area2 - intersection)
        return max(0.0, min(1.0, iou))  # Ensure value is between 0 and 1
    
    def evaluate_detection(self, iou_threshold=0.5):
        """
        Evaluate face detection performance
        
        Args:
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating detection performance (IoU threshold: {iou_threshold})...")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Get frames from ground truth
        gt_frames = self.ground_truth['frames']
        det_frames = self.detection_results['frames']
        
        # Get coordinate systems
        gt_coord = self.ground_truth.get('coordinate_system', self.coordinate_system)
        det_coord = self.detection_results.get('coordinate_system', self.coordinate_system)
        
        processed_frames = 0
        matched_frames = 0
        
        # Debug info
        debug_info = {
            'matched_frames': 0,
            'unmatched_frames': 0,
            'total_gt_faces': 0,
            'total_det_faces': 0,
            'total_matches': 0
        }
        
        # Process each frame in ground truth
        for frame_id, gt_frame in gt_frames.items():
            processed_frames += 1
            
            # Skip frames without faces
            if not gt_frame.get('faces', []):
                continue
                
            # Find corresponding detection frame
            det_frame = None
            if frame_id in det_frames:
                det_frame = det_frames[frame_id]
                matched_frames += 1
                debug_info['matched_frames'] += 1
            else:
                # Try string/int conversion
                try:
                    # Convert between string and int frame_id
                    alt_frame_id = int(frame_id) if isinstance(frame_id, str) else str(frame_id)
                    if alt_frame_id in det_frames:
                        det_frame = det_frames[alt_frame_id]
                        matched_frames += 1
                        debug_info['matched_frames'] += 1
                except:
                    debug_info['unmatched_frames'] += 1
            
            # Count ground truth faces
            gt_faces = gt_frame.get('faces', [])
            debug_info['total_gt_faces'] += len(gt_faces)
            
            # If no matching detection frame, count all GT faces as false negatives
            if not det_frame:
                false_negatives += len(gt_faces)
                continue
                
            # Get detected faces
            if isinstance(det_frame, list):
                det_faces = det_frame  # Already a list of faces
            else:
                det_faces = det_frame.get('faces', [])
            
            debug_info['total_det_faces'] += len(det_faces)
            
            # Convert bbox coordinates to unified system
            unified_gt_boxes = []
            for face in gt_faces:
                if 'bbox' in face:
                    # Standardize and convert to unified coordinates
                    std_bbox = self._standardize_bbox_format(face['bbox'])
                    unified_bbox = self._convert_bbox_to_unified_coordinates(std_bbox, gt_coord)
                    unified_gt_boxes.append(unified_bbox)
            
            unified_det_boxes = []
            for face in det_faces:
                if 'bbox' in face:
                    # Standardize and convert to unified coordinates
                    std_bbox = self._standardize_bbox_format(face['bbox'])
                    unified_bbox = self._convert_bbox_to_unified_coordinates(std_bbox, det_coord)
                    unified_det_boxes.append(unified_bbox)
            
            # Match ground truth boxes with detection boxes
            matched_gt_indices = set()
            
            for det_idx, det_bbox in enumerate(unified_det_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_bbox in enumerate(unified_gt_boxes):
                    if gt_idx in matched_gt_indices:
                        continue  # Skip already matched GT boxes
                        
                    iou = self._compute_iou(det_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    # Match found
                    true_positives += 1
                    matched_gt_indices.add(best_gt_idx)
                    debug_info['total_matches'] += 1
                else:
                    # No match found
                    false_positives += 1
            
            # Count unmatched ground truth boxes as false negatives
            false_negatives += len(unified_gt_boxes) - len(matched_gt_indices)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Print detection results
        print(f"Detection Results:")
        print(f"  Processed frames: {processed_frames}")
        print(f"  Matched frames: {matched_frames}")
        print(f"  True positives: {true_positives}")
        print(f"  False positives: {false_positives}")
        print(f"  False negatives: {false_negatives}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Print debug info
        print(f"\nDebug Info:")
        print(f"  Matched frames: {debug_info['matched_frames']}")
        print(f"  Unmatched frames: {debug_info['unmatched_frames']}")
        print(f"  Total GT faces: {debug_info['total_gt_faces']}")
        print(f"  Total detected faces: {debug_info['total_det_faces']}")
        print(f"  Total matches: {debug_info['total_matches']}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'debug_info': debug_info
        }
    
    def evaluate_recognition(self, iou_threshold=0.5):
        """
        Evaluate face recognition performance
        
        Args:
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating recognition performance (IoU threshold: {iou_threshold})...")
        
        # Get frames from ground truth
        gt_frames = self.ground_truth['frames']
        det_frames = self.detection_results['frames']
        
        # Get coordinate systems
        gt_coord = self.ground_truth.get('coordinate_system', self.coordinate_system)
        det_coord = self.detection_results.get('coordinate_system', self.coordinate_system)
        
        # Statistics
        total_matched_faces = 0
        correct_identifications = 0
        
        # Confusion matrix data
        all_true_ids = []
        all_pred_ids = []
        
        # Process each frame in ground truth
        for frame_id, gt_frame in gt_frames.items():
            # Skip frames without faces
            if not gt_frame.get('faces', []):
                continue
                
            # Find corresponding detection frame
            det_frame = None
            if frame_id in det_frames:
                det_frame = det_frames[frame_id]
            else:
                # Try string/int conversion
                try:
                    alt_frame_id = int(frame_id) if isinstance(frame_id, str) else str(frame_id)
                    if alt_frame_id in det_frames:
                        det_frame = det_frames[alt_frame_id]
                except:
                    pass
            
            if not det_frame:
                continue
            
            # Get faces
            gt_faces = gt_frame.get('faces', [])
            
            if isinstance(det_frame, list):
                det_faces = det_frame
            else:
                det_faces = det_frame.get('faces', [])
            
            # Convert bbox coordinates to unified system
            unified_gt_boxes = []
            gt_face_ids = []
            
            for face in gt_faces:
                if 'bbox' in face and face.get('face_id', -1) >= 0:
                    std_bbox = self._standardize_bbox_format(face['bbox'])
                    unified_bbox = self._convert_bbox_to_unified_coordinates(std_bbox, gt_coord)
                    unified_gt_boxes.append(unified_bbox)
                    gt_face_ids.append(face.get('face_id', -1))
            
            unified_det_boxes = []
            det_face_ids = []
            
            for face in det_faces:
                if 'bbox' in face:
                    std_bbox = self._standardize_bbox_format(face['bbox'])
                    unified_bbox = self._convert_bbox_to_unified_coordinates(std_bbox, det_coord)
                    unified_det_boxes.append(unified_bbox)
                    
                    # Get face ID from detection (could be match_idx or face_id)
                    face_id = -1
                    if 'match_idx' in face:
                        face_id = face['match_idx']
                    elif 'face_id' in face:
                        face_id = face['face_id']
                    
                    det_face_ids.append(face_id)
            
            # Match ground truth boxes with detection boxes
            for gt_idx, gt_bbox in enumerate(unified_gt_boxes):
                best_iou = 0
                best_det_idx = -1
                
                for det_idx, det_bbox in enumerate(unified_det_boxes):
                    iou = self._compute_iou(gt_bbox, det_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = det_idx
                
                if best_iou >= iou_threshold and best_det_idx >= 0:
                    # Match found - check if IDs match
                    total_matched_faces += 1
                    
                    gt_id = gt_face_ids[gt_idx]
                    det_id = det_face_ids[best_det_idx]
                    
                    # Add to confusion matrix data
                    all_true_ids.append(gt_id)
                    all_pred_ids.append(det_id)
                    
                    if gt_id == det_id:
                        correct_identifications += 1
        
        # Calculate recognition accuracy
        recognition_accuracy = correct_identifications / total_matched_faces if total_matched_faces > 0 else 0
        
        # Print recognition results
        print(f"Recognition Results:")
        print(f"  Total matched faces: {total_matched_faces}")
        print(f"  Correct identifications: {correct_identifications}")
        print(f"  Recognition accuracy: {recognition_accuracy:.4f}")
        
        return {
            'recognition_accuracy': recognition_accuracy,
            'correct_identifications': correct_identifications,
            'total_matched_faces': total_matched_faces,
            'true_ids': all_true_ids,
            'pred_ids': all_pred_ids
        }
    
    def visualize_results(self, output_dir):
        """
        Visualize evaluation results
        
        Args:
            output_dir: Directory to save visualization results
        """
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Run evaluations if not already done
        detection_metrics = self.evaluate_detection()
        recognition_metrics = self.evaluate_recognition()
        
        # Create detection metrics bar chart
        plt.figure(figsize=(10, 6))
        metrics = ['Precision', 'Recall', 'F1 Score']
        values = [
            detection_metrics['precision'],
            detection_metrics['recall'],
            detection_metrics['f1_score']
        ]
        
        plt.bar(metrics, values, color=['blue', 'green', 'red'])
        plt.ylim(0, 1.1)
        plt.title('Face Detection Performance')
        plt.ylabel('Score')
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.05, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_metrics.png'), dpi=300)
        plt.close()
        
        # Create recognition accuracy chart
        plt.figure(figsize=(8, 6))
        plt.bar(['Recognition Accuracy'], [recognition_metrics['recognition_accuracy']], color='purple')
        plt.ylim(0, 1.1)
        plt.title('Face Recognition Performance')
        plt.ylabel('Accuracy')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value on top of bar
        plt.text(0, recognition_metrics['recognition_accuracy'] + 0.05, 
                f"{recognition_metrics['recognition_accuracy']:.4f}", 
                ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'recognition_accuracy.png'), dpi=300)
        plt.close()
        
        # Create confusion matrix if there's enough data
        true_ids = recognition_metrics['true_ids']
        pred_ids = recognition_metrics['pred_ids']
        
        if len(true_ids) > 0 and len(set(true_ids)) > 1:
            plt.figure(figsize=(10, 8))
            
            # Get unique IDs
            all_ids = sorted(list(set(true_ids + pred_ids)))
            
            # Create confusion matrix
            cm = np.zeros((len(all_ids), len(all_ids)), dtype=int)
            id_to_idx = {id: i for i, id in enumerate(all_ids)}
            
            for t, p in zip(true_ids, pred_ids):
                t_idx = id_to_idx[t]
                p_idx = id_to_idx[p]
                cm[t_idx, p_idx] += 1
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=all_ids, yticklabels=all_ids)
            plt.xlabel('Predicted ID')
            plt.ylabel('True ID')
            plt.title('Face Recognition Confusion Matrix')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
        
        print(f"Visualization results saved to {output_dir}")


def main():
    """Main function to run evaluation"""
    # Default paths - adjust as needed
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
        return
    
    print(f"Using detection results from: {detection_results_path}")
    
    # Create output directory
    output_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\evaluation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run evaluation
    evaluator = SimpleEvaluator(ground_truth_path, detection_results_path)
    
    # Evaluate and visualize
    evaluator.visualize_results(output_dir)


if __name__ == "__main__":
    main()