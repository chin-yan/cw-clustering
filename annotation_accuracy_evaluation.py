#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speaking Face Recognition Accuracy Evaluation System

This tool compares:
- Ground Truth (manual annotations)
- System Predictions (from enhanced_detection_results.pkl)

And calculates various accuracy metrics.
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class SpeakingAccuracyEvaluator:
    """Evaluate speaking face recognition accuracy"""
    
    def __init__(self, ground_truth_path, system_results_path, output_dir):
        """
        Initialize evaluator
        
        Args:
            ground_truth_path: Path to ground truth JSON file
            system_results_path: Path to system detection results (PKL)
            output_dir: Output directory for results
        """
        self.ground_truth_path = ground_truth_path
        self.system_results_path = system_results_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        self.ground_truth = self._load_ground_truth()
        self.system_results = self._load_system_results()
        
        # Results storage
        self.matched_results = []
        self.metrics = {}
        
    def _load_ground_truth(self):
        """Load ground truth annotations"""
        print(f"Loading ground truth: {self.ground_truth_path}")
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} ground truth annotations")
        return data
    
    def _load_system_results(self):
        """Load system detection results"""
        print(f"Loading system results: {self.system_results_path}")
        
        with open(self.system_results_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check format
        if isinstance(data, dict):
            # Check if it's frame-indexed
            if all(isinstance(k, int) for k in list(data.keys())[:10]):
                print(f"Loaded frame-indexed results: {len(data)} frames")
                return data
            else:
                print(f"Warning: Unexpected data format")
                return data
        else:
            print(f"Warning: System results is not a dictionary")
            return {}
    
    def match_ground_truth_to_system(self, time_tolerance=0.5, fps=30.0):
        """
        Match ground truth annotations to system predictions
        
        Args:
            time_tolerance: Time tolerance in seconds for matching
            fps: Video frame rate
            
        Returns:
            List of matched results
        """
        print("\nMatching ground truth to system predictions...")
        
        matched = []
        unmatched_gt = []
        
        for gt_key, gt_data in self.ground_truth.items():
            timestamp = gt_data['timestamp']
            frame_idx = int(timestamp * fps)
            
            # Search nearby frames
            frame_tolerance = int(time_tolerance * fps)
            
            found_match = False
            for offset in range(-frame_tolerance, frame_tolerance + 1):
                check_frame = frame_idx + offset
                
                if check_frame in self.system_results:
                    # Found matching frame
                    system_faces = self.system_results[check_frame]
                    
                    # Extract system prediction
                    system_prediction = self._extract_system_prediction(
                        system_faces, gt_data
                    )
                    
                    matched.append({
                        'gt_key': gt_key,
                        'timestamp': timestamp,
                        'frame_idx': check_frame,
                        'gt_speaker_id': gt_data.get('speaker_id', -1),
                        'gt_speaker_ids': gt_data.get('speaker_ids', []),
                        'gt_all_faces': gt_data.get('all_faces', []),
                        'gt_status': gt_data.get('status', 'unknown'),
                        'gt_text': gt_data.get('text', ''),
                        'system_faces': system_faces,
                        'system_prediction': system_prediction
                    })
                    
                    found_match = True
                    break
            
            if not found_match:
                unmatched_gt.append({
                    'gt_key': gt_key,
                    'timestamp': timestamp,
                    'reason': 'No system data at this frame'
                })
        
        print(f"Matched: {len(matched)}")
        print(f"Unmatched: {len(unmatched_gt)}")
        
        self.matched_results = matched
        return matched
    
    def _extract_system_prediction(self, system_faces, gt_data):
        """
        Extract system prediction from face detection results
        
        Args:
            system_faces: List of detected faces from system
            gt_data: Ground truth data
            
        Returns:
            Dictionary with system prediction
        """
        if not system_faces:
            return {
                'detected_faces': [],
                'predicted_speaker_id': -1,
                'predicted_speaker_ids': [],
                'confidence': 0.0
            }
        
        # Extract all detected face IDs
        detected_face_ids = []
        face_info = []
        
        for face in system_faces:
            char_id = face.get('match_idx', -1)
            similarity = face.get('similarity', 0.0)
            
            detected_face_ids.append(char_id)
            face_info.append({
                'char_id': char_id,
                'similarity': similarity,
                'bbox': face.get('bbox', None)
            })
        
        # Determine predicted speaker
        # Strategy: Highest similarity face is predicted speaker
        if face_info:
            best_face = max(face_info, key=lambda x: x['similarity'])
            predicted_speaker_id = best_face['char_id']
            confidence = best_face['similarity']
        else:
            predicted_speaker_id = -1
            confidence = 0.0
        
        return {
            'detected_faces': detected_face_ids,
            'predicted_speaker_id': predicted_speaker_id,
            'predicted_speaker_ids': detected_face_ids,  # All detected faces
            'confidence': confidence,
            'face_info': face_info
        }
    
    def calculate_metrics(self):
        """Calculate accuracy metrics"""
        print("\nCalculating accuracy metrics...")
        
        if not self.matched_results:
            print("No matched results to evaluate")
            return
        
        # Initialize counters
        total = len(self.matched_results)
        
        # Speaker identification metrics
        speaker_correct = 0
        speaker_wrong = 0
        speaker_unknown = 0
        
        # Chorus handling
        chorus_correct = 0
        chorus_partial = 0
        chorus_wrong = 0
        
        # Per-status breakdown
        status_metrics = defaultdict(lambda: {
            'total': 0, 'correct': 0, 'wrong': 0, 'unknown': 0
        })
        
        # Confusion matrix
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        # Detailed results
        detailed_results = []
        
        for result in self.matched_results:
            gt_status = result['gt_status']
            gt_speaker_id = result['gt_speaker_id']
            gt_speaker_ids = result['gt_speaker_ids']
            
            system_pred = result['system_prediction']
            pred_speaker_id = system_pred['predicted_speaker_id']
            pred_speaker_ids = system_pred['predicted_speaker_ids']
            
            status_metrics[gt_status]['total'] += 1
            
            # Evaluate based on status
            if gt_status == 'chorus':
                # Chorus mode: Check if predicted speaker is in the chorus
                if pred_speaker_id in gt_speaker_ids and pred_speaker_id >= 0:
                    speaker_correct += 1
                    chorus_correct += 1
                    status_metrics[gt_status]['correct'] += 1
                    result_label = 'correct'
                elif pred_speaker_id >= 0:
                    # Detected someone, but wrong
                    speaker_wrong += 1
                    chorus_wrong += 1
                    status_metrics[gt_status]['wrong'] += 1
                    result_label = 'wrong'
                else:
                    # No detection
                    speaker_unknown += 1
                    chorus_wrong += 1
                    status_metrics[gt_status]['unknown'] += 1
                    result_label = 'unknown'
                
                # Also check if we detected all chorus members (partial match)
                detected_chorus = set(pred_speaker_ids) & set(gt_speaker_ids)
                if len(detected_chorus) > 0:
                    chorus_partial += 1
            
            elif gt_status == 'speaker_not_visible':
                # Speaker not in frame - system should not detect them
                # This is tricky - we can't expect system to predict correctly
                # Mark as special case
                status_metrics[gt_status]['total'] += 1
                result_label = 'not_applicable'

            elif gt_status == 'narration':
                # Narration - system should not predict any specific speaker
                # This is expected to have no correct speaker match
                status_metrics[gt_status]['total'] += 1
                
                # For narration, we don't evaluate as correct/wrong
                # Just mark as special case
                result_label = 'narration_not_evaluated'
            
            elif gt_status in ['all_correct', 'partially_corrected']:
                # Normal single speaker case
                if pred_speaker_id == gt_speaker_id and pred_speaker_id >= 0:
                    speaker_correct += 1
                    status_metrics[gt_status]['correct'] += 1
                    result_label = 'correct'
                elif pred_speaker_id >= 0:
                    speaker_wrong += 1
                    status_metrics[gt_status]['wrong'] += 1
                    result_label = 'wrong'
                else:
                    speaker_unknown += 1
                    status_metrics[gt_status]['unknown'] += 1
                    result_label = 'unknown'
                
                # Confusion matrix
                confusion_matrix[gt_speaker_id][pred_speaker_id] += 1
            
            else:
                # Unknown or other status
                result_label = 'skipped'
            
            # Store detailed result
            detailed_results.append({
                **result,
                'evaluation': result_label,
                'is_correct': result_label == 'correct'
            })
        
        # Calculate overall accuracy
        evaluable = speaker_correct + speaker_wrong
        if evaluable > 0:
            accuracy = speaker_correct / evaluable
        else:
            accuracy = 0.0
        
        # Calculate recall (how many ground truth speakers we detected)
        total_with_speaker = evaluable + speaker_unknown
        if total_with_speaker > 0:
            recall = speaker_correct / total_with_speaker
        else:
            recall = 0.0
        
        # Calculate precision (of detected speakers, how many were correct)
        total_detected = speaker_correct + speaker_wrong
        if total_detected > 0:
            precision = speaker_correct / total_detected
        else:
            precision = 0.0
        
        # F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Store metrics
        self.metrics = {
            'total_annotations': total,
            'speaker_correct': speaker_correct,
            'speaker_wrong': speaker_wrong,
            'speaker_unknown': speaker_unknown,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'chorus_correct': chorus_correct,
            'chorus_partial': chorus_partial,
            'chorus_wrong': chorus_wrong,
            'status_metrics': dict(status_metrics),
            'confusion_matrix': dict(confusion_matrix),
            'detailed_results': detailed_results
        }
        
        return self.metrics
    
    def print_report(self):
        """Print evaluation report"""
        if not self.metrics:
            print("No metrics to report. Run calculate_metrics() first.")
            return
        
        print("\n" + "="*70)
        print("SPEAKING FACE RECOGNITION ACCURACY REPORT")
        print("="*70)
        
        m = self.metrics
        
        print(f"\nTotal Ground Truth Annotations: {m['total_annotations']}")
        print(f"Evaluable Samples: {m['speaker_correct'] + m['speaker_wrong']}")
        
        print("\n" + "-"*70)
        print("OVERALL METRICS")
        print("-"*70)
        print(f"Accuracy:  {m['accuracy']*100:.2f}%  ({m['speaker_correct']}/{m['speaker_correct'] + m['speaker_wrong']})")
        print(f"Precision: {m['precision']*100:.2f}%  (of detected, how many correct)")
        print(f"Recall:    {m['recall']*100:.2f}%  (of all speakers, how many detected)")
        print(f"F1-Score:  {m['f1_score']*100:.2f}%")
        
        print("\n" + "-"*70)
        print("BREAKDOWN")
        print("-"*70)
        print(f"Correct Predictions:   {m['speaker_correct']}")
        print(f"Wrong Predictions:     {m['speaker_wrong']}")
        print(f"No Detection (Unknown): {m['speaker_unknown']}")
        
        if m['chorus_correct'] + m['chorus_wrong'] > 0:
            print("\n" + "-"*70)
            print("CHORUS MODE PERFORMANCE")
            print("-"*70)
            total_chorus = m['chorus_correct'] + m['chorus_wrong']
            chorus_acc = m['chorus_correct'] / total_chorus if total_chorus > 0 else 0
            print(f"Chorus Accuracy: {chorus_acc*100:.2f}%  ({m['chorus_correct']}/{total_chorus})")
            print(f"  Fully Correct: {m['chorus_correct']}")
            print(f"  Partially Correct: {m['chorus_partial']}")
            print(f"  Wrong: {m['chorus_wrong']}")
        
        print("\n" + "-"*70)
        print("PER-STATUS METRICS")
        print("-"*70)
        for status, stats in m['status_metrics'].items():
            if stats['total'] > 0:
                correct_rate = stats['correct'] / stats['total'] * 100
                print(f"\n{status}:")
                print(f"  Total: {stats['total']}")
                print(f"  Correct: {stats['correct']} ({correct_rate:.1f}%)")
                print(f"  Wrong: {stats['wrong']}")
                print(f"  Unknown: {stats['unknown']}")
        
        print("\n" + "="*70)
    
    def save_detailed_results(self):
        """Save detailed results to files"""
        print("\nSaving detailed results...")
        
        # Save metrics JSON
        metrics_file = self.output_dir / 'accuracy_metrics.json'
        
        # Prepare JSON-serializable metrics
        metrics_json = {
            k: v for k, v in self.metrics.items() 
            if k not in ['detailed_results', 'confusion_matrix', 'status_metrics']
        }
        
        # Add serializable versions
        metrics_json['status_metrics'] = {
            status: dict(stats) 
            for status, stats in self.metrics['status_metrics'].items()
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Saved metrics: {metrics_file}")
        
        # Save detailed results CSV
        csv_file = self.output_dir / 'detailed_results.csv'
        
        with open(csv_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("timestamp,frame_idx,gt_speaker_id,pred_speaker_id,")
            f.write("gt_status,evaluation,confidence,subtitle_text\n")
            
            # Data
            for result in self.metrics['detailed_results']:
                f.write(f"{result['timestamp']:.2f},")
                f.write(f"{result['frame_idx']},")
                f.write(f"{result['gt_speaker_id']},")
                f.write(f"{result['system_prediction']['predicted_speaker_id']},")
                f.write(f"{result['gt_status']},")
                f.write(f"{result['evaluation']},")
                f.write(f"{result['system_prediction']['confidence']:.3f},")
                f.write(f'"{result.get("gt_text", "")[:50]}"\n')
        
        print(f"Saved detailed results: {csv_file}")
        
        # Save confusion matrix
        self._save_confusion_matrix()
    
    def _save_confusion_matrix(self):
        """Save and visualize confusion matrix"""
        cm = self.metrics['confusion_matrix']
        
        if not cm:
            print("No confusion matrix to save")
            return
        
        # Get all unique IDs
        all_ids = sorted(set(
            list(cm.keys()) + 
            [pred_id for preds in cm.values() for pred_id in preds.keys()]
        ))
        
        # Filter out -1 (unknown)
        all_ids = [id for id in all_ids if id >= 0]
        
        if not all_ids:
            return
        
        # Create matrix
        matrix = np.zeros((len(all_ids), len(all_ids)), dtype=int)
        
        for i, true_id in enumerate(all_ids):
            for j, pred_id in enumerate(all_ids):
                matrix[i, j] = cm.get(true_id, {}).get(pred_id, 0)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=all_ids, yticklabels=all_ids)
        plt.title('Speaker Identification Confusion Matrix', fontsize=16, pad=20)
        plt.xlabel('Predicted Speaker ID', fontsize=12)
        plt.ylabel('True Speaker ID', fontsize=12)
        plt.tight_layout()
        
        cm_file = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix: {cm_file}")
    
    def create_visualizations(self):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        
        if not self.metrics or not self.metrics['detailed_results']:
            print("No data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy by status
        ax1 = axes[0, 0]
        statuses = []
        accuracies = []
        
        for status, stats in self.metrics['status_metrics'].items():
            if stats['total'] > 0 and status != 'speaker_not_visible':
                statuses.append(status)
                acc = stats['correct'] / stats['total'] * 100
                accuracies.append(acc)
        
        if statuses:
            bars = ax1.bar(range(len(statuses)), accuracies, color='skyblue')
            ax1.set_xticks(range(len(statuses)))
            ax1.set_xticklabels(statuses, rotation=45, ha='right')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('Accuracy by Annotation Status')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Prediction distribution
        ax2 = axes[0, 1]
        labels = ['Correct', 'Wrong', 'No Detection']
        sizes = [
            self.metrics['speaker_correct'],
            self.metrics['speaker_wrong'],
            self.metrics['speaker_unknown']
        ]
        colors = ['#4CAF50', '#F44336', '#9E9E9E']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax2.set_title('Prediction Distribution')
        
        # 3. Confidence distribution
        ax3 = axes[1, 0]
        correct_confs = []
        wrong_confs = []
        
        for result in self.metrics['detailed_results']:
            conf = result['system_prediction']['confidence']
            if result['evaluation'] == 'correct':
                correct_confs.append(conf)
            elif result['evaluation'] == 'wrong':
                wrong_confs.append(conf)
        
        if correct_confs or wrong_confs:
            bins = np.linspace(0, 1, 20)
            if correct_confs:
                ax3.hist(correct_confs, bins=bins, alpha=0.6, label='Correct', color='green')
            if wrong_confs:
                ax3.hist(wrong_confs, bins=bins, alpha=0.6, label='Wrong', color='red')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Confidence Score Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Metrics comparison
        ax4 = axes[1, 1]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            self.metrics['accuracy'] * 100,
            self.metrics['precision'] * 100,
            self.metrics['recall'] * 100,
            self.metrics['f1_score'] * 100
        ]
        
        bars = ax4.bar(range(len(metric_names)), metric_values, 
                      color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
        ax4.set_xticks(range(len(metric_names)))
        ax4.set_xticklabels(metric_names)
        ax4.set_ylabel('Score (%)')
        ax4.set_title('Overall Performance Metrics')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        vis_file = self.output_dir / 'accuracy_visualization.png'
        plt.savefig(vis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualizations: {vis_file}")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*70)
        print("RUNNING FULL EVALUATION PIPELINE")
        print("="*70)
        
        # Step 1: Match
        self.match_ground_truth_to_system()
        
        # Step 2: Calculate metrics
        self.calculate_metrics()
        
        # Step 3: Print report
        self.print_report()
        
        # Step 4: Save results
        self.save_detailed_results()
        
        # Step 5: Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate speaking face recognition accuracy'
    )
    parser.add_argument('--ground-truth', required=True,
                       help='Path to ground truth JSON file')
    parser.add_argument('--system-results', required=True,
                       help='Path to system detection results (PKL file)')
    parser.add_argument('--output', default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Video frame rate')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                       help='Time tolerance for matching (seconds)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SpeakingAccuracyEvaluator(
        ground_truth_path=args.ground_truth,
        system_results_path=args.system_results,
        output_dir=args.output
    )
    
    # Run evaluation
    evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()