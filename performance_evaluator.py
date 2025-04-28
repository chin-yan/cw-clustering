import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

class RetrievalPerformanceEvaluator:
    def __init__(self, ground_truth_path, centers_data_path):
        """
        Initialize the retrieval performance evaluator
        
        Args:
            ground_truth_path: Path to ground truth JSON file
            centers_data_path: Path to cluster centers data
        """
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
        
        # Load centers data
        with open(centers_data_path, 'rb') as f:
            self.centers_data = pickle.load(f)
        
        # Create mapping from face path to face_id
        self.face_path_to_id = {}
        for face in self.ground_truth["faces"]:
            self.face_path_to_id[os.path.basename(face["face_path"])] = face["face_id"]
            
        # Extract centers
        _, self.center_paths = self.centers_data["cluster_centers"]
    
    def evaluate_retrieval_results(self, retrieval_results_path):
        """
        Evaluate face retrieval performance
        
        Args:
            retrieval_results_path: Path to retrieval results pickle file
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating retrieval results from {retrieval_results_path}...")
        
        # Load retrieval results
        with open(retrieval_results_path, 'rb') as f:
            retrieval_data = pickle.load(f)
        
        # Handle different retrieval result formats
        if isinstance(retrieval_data, dict) and 'by_center' in retrieval_data:
            retrieval_results = retrieval_data['by_center']
        else:
            retrieval_results = retrieval_data
        
        # Initialize metrics
        total_faces = 0
        correct_retrievals = 0
        rank1_correct = 0
        
        # For confusion matrix
        all_true_ids = []
        all_pred_ids = []
        
        # For per-center metrics
        per_center_metrics = {}
        
        # Evaluate center by center
        for center_idx, center_path in enumerate(self.center_paths):
            if center_idx not in retrieval_results:
                continue
            
            # Get center's ground truth ID if available
            center_gt_id = -1
            center_basename = os.path.basename(center_path)
            if center_basename in self.face_path_to_id:
                center_gt_id = self.face_path_to_id[center_basename]
            else:
                # Try to find by matching any part of the path
                for face_path, gt_id in self.face_path_to_id.items():
                    if face_path in center_path or center_path in face_path:
                        center_gt_id = gt_id
                        break
            
            if center_gt_id == -1:
                print(f"Warning: No ground truth ID found for center {center_idx} ({center_path})")
                continue
            
            # Initialize per-center metrics
            if center_gt_id not in per_center_metrics:
                per_center_metrics[center_gt_id] = {
                    "total": 0,
                    "correct": 0,
                    "precision": 0
                }
            
            # Evaluate retrieved faces for this center
            retrieved_faces = retrieval_results[center_idx]
            center_total = 0
            center_correct = 0
            
            for rank, face_info in enumerate(retrieved_faces):
                face_path = face_info["path"]
                face_basename = os.path.basename(face_path)
                
                # Get ground truth ID for this face
                if face_basename in self.face_path_to_id:
                    total_faces += 1
                    center_total += 1
                    
                    retrieved_id = center_idx  # The ID assigned by the retrieval system
                    true_id = self.face_path_to_id[face_basename]
                    
                    # For confusion matrix
                    all_true_ids.append(true_id)
                    all_pred_ids.append(center_gt_id)  # Use ground truth ID of the center
                    
                    if true_id == center_gt_id:
                        correct_retrievals += 1
                        center_correct += 1
                        if rank == 0:  # Rank 1 (first result)
                            rank1_correct += 1
            
            # Update per-center metrics
            if center_total > 0:
                per_center_metrics[center_gt_id]["total"] += center_total
                per_center_metrics[center_gt_id]["correct"] += center_correct
                per_center_metrics[center_gt_id]["precision"] = center_correct / center_total
        
        # Calculate overall metrics
        retrieval_accuracy = correct_retrievals / total_faces if total_faces > 0 else 0
        rank1_accuracy = rank1_correct / total_faces if total_faces > 0 else 0
        
        # Calculate per-class metrics if true and pred IDs exist
        if all_true_ids and all_pred_ids:
            # Convert to arrays
            all_true_ids = np.array(all_true_ids)
            all_pred_ids = np.array(all_pred_ids)
            
            # Get unique labels
            unique_ids = sorted(set(all_true_ids) | set(all_pred_ids))
            
            # Calculate precision, recall, F1-score per class
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_true_ids, all_pred_ids, labels=unique_ids, average=None, zero_division=0
            )
            
            # Overall metrics
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
                all_true_ids, all_pred_ids, average='weighted', zero_division=0
            )
            
            # Create confusion matrix
            cm = confusion_matrix(all_true_ids, all_pred_ids, labels=unique_ids)
            
            # Store class metrics
            class_metrics = {
                unique_ids[i]: {
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1": f1[i]
                } for i in range(len(unique_ids))
            }
        else:
            overall_precision = 0
            overall_recall = 0
            overall_f1 = 0
            class_metrics = {}
            cm = None
            unique_ids = []
        
        # Compile results
        metrics = {
            "retrieval_accuracy": retrieval_accuracy,
            "rank1_accuracy": rank1_accuracy,
            "correct_retrievals": correct_retrievals,
            "total_faces": total_faces,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "per_center_metrics": per_center_metrics,
            "class_metrics": class_metrics,
            "confusion_matrix": cm,
            "unique_ids": unique_ids
        }
        
        # Print metrics summary
        print("\nRetrieval Performance Metrics:")
        print(f"Overall Retrieval Accuracy: {retrieval_accuracy:.4f}")
        print(f"Rank-1 Accuracy: {rank1_accuracy:.4f}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Correct Retrievals: {correct_retrievals}")
        print(f"Total Evaluated Faces: {total_faces}")
        
        # Return compiled metrics
        return metrics
    
    def visualize_results(self, metrics, output_dir):
        """
        Visualize evaluation results with charts
        
        Args:
            metrics: Dictionary with evaluation metrics
            output_dir: Directory to save visualization results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot per-class precision, recall, F1
        if metrics["class_metrics"]:
            plt.figure(figsize=(12, 6))
            
            class_ids = sorted(metrics["class_metrics"].keys())
            precision_values = [metrics["class_metrics"][cid]["precision"] for cid in class_ids]
            recall_values = [metrics["class_metrics"][cid]["recall"] for cid in class_ids]
            f1_values = [metrics["class_metrics"][cid]["f1"] for cid in class_ids]
            
            x = np.arange(len(class_ids))
            width = 0.25
            
            plt.bar(x - width, precision_values, width, label='Precision')
            plt.bar(x, recall_values, width, label='Recall')
            plt.bar(x + width, f1_values, width, label='F1 Score')
            
            plt.xlabel('Person ID')
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F1 Score per Person')
            plt.xticks(x, class_ids)
            plt.legend()
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300)
            plt.close()
        
        # Plot confusion matrix
        if metrics["confusion_matrix"] is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', 
                        xticklabels=metrics["unique_ids"],
                        yticklabels=metrics["unique_ids"])
            plt.xlabel('Predicted ID')
            plt.ylabel('True ID')
            plt.title('Confusion Matrix')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
        
        # Plot retrieval accuracy
        plt.figure(figsize=(8, 6))
        metrics_names = ['Retrieval Accuracy', 'Rank-1 Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [
            metrics["retrieval_accuracy"], 
            metrics["rank1_accuracy"],
            metrics["overall_precision"],
            metrics["overall_recall"],
            metrics["overall_f1"]
        ]
        
        plt.bar(metrics_names, metrics_values)
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.title('Overall Performance Metrics')
        plt.grid(True, alpha=0.3)
        
        for i, v in enumerate(metrics_values):
            plt.text(i, v + 0.05, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=300)
        plt.close()
    
    def evaluate_and_visualize(self, retrieval_results_path, output_dir):
        """
        Evaluate retrieval results and visualize the performance
        
        Args:
            retrieval_results_path: Path to retrieval results pickle file
            output_dir: Directory to save visualization results
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = self.evaluate_retrieval_results(retrieval_results_path)
        self.visualize_results(metrics, output_dir)
        return metrics

def main():
    # Configure paths
    ground_truth_path = "ground_truth.json"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\centers\centers_data.pkl"
    
    # Create evaluator
    evaluator = RetrievalPerformanceEvaluator(ground_truth_path, centers_data_path)
    
    # Evaluate enhanced retrieval results
    enhanced_retrieval_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\retrieval\enhanced_retrieval_results.pkl"
    if os.path.exists(enhanced_retrieval_path):
        enhanced_metrics = evaluator.evaluate_and_visualize(
            enhanced_retrieval_path, 
            r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\evaluation\enhanced"
        )
    
    # Evaluate original retrieval results if available
    original_retrieval_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\retrieval\retrieval_results.pkl"
    if os.path.exists(original_retrieval_path):
        original_metrics = evaluator.evaluate_and_visualize(
            original_retrieval_path, 
            r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\evaluation\original"
        )
        
        # Compare results if both methods were evaluated
        if 'enhanced_metrics' in locals():
            print("\nComparison of Enhanced vs Original Methods:")
            comparison_items = [
                ("Retrieval Accuracy", "retrieval_accuracy"),
                ("Rank-1 Accuracy", "rank1_accuracy"),
                ("Overall Precision", "overall_precision"),
                ("Overall Recall", "overall_recall"),
                ("Overall F1 Score", "overall_f1")
            ]
            
            for label, metric_key in comparison_items:
                enhanced_value = enhanced_metrics[metric_key]
                original_value = original_metrics[metric_key]
                diff = enhanced_value - original_value
                diff_percent = (diff / original_value) * 100 if original_value != 0 else float('inf')
                
                print(f"{label}: Enhanced: {enhanced_value:.4f}, Original: {original_value:.4f}, "
                      f"Diff: {diff:.4f} ({diff_percent:+.2f}%)")

if __name__ == "__main__":
    main()
