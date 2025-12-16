import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set font support for better compatibility
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class ConfusionMatrixAnalyzer:
    def __init__(self, excel_file_path):
        """
        Initialize confusion matrix analyzer
        
        Parameters:
        excel_file_path (str): Path to Excel file
        """
        self.excel_file_path = excel_file_path
        
        self.output_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\InsightFace\result_s2ep10"
        self.confusion_matrix = None
        self.labels = None
        
        # Create output directory if it doesn't exist
        self.create_output_directory()
        
        # Create timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"File timestamp: {self.timestamp}")
        
        self.load_data()
    
    def create_output_directory(self):
        """Create output directory"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"Created output directory: {self.output_dir}")
            else:
                print(f"Using existing output directory: {self.output_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            # If unable to create specified directory, use current directory
            self.output_dir = "."
            print("Will use current directory as output directory")
    
    def get_output_path(self, filename):
        """Get full output file path"""
        full_path = os.path.join(self.output_dir, f"{self.timestamp}_{filename}")
        print(f"Will save file to: {full_path}")
        return full_path
    
    def load_data(self):
        """Load confusion matrix from Excel file"""
        try:
            # Read Excel file
            df = pd.read_excel(self.excel_file_path, header=0, index_col=0)
            
            # Extract confusion matrix and labels
            self.confusion_matrix = df.values
            self.labels = df.columns.tolist()
            
            print(f"Successfully loaded confusion matrix: {self.confusion_matrix.shape}")
            print(f"Number of labels: {len(self.labels)}")
            print(f"Labels: {self.labels[:10]}...")  # Show first 10 labels
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def calculate_metrics(self):
        """Calculate various evaluation metrics"""
        # Calculate basic metrics
        total_samples = np.sum(self.confusion_matrix)
        correct_predictions = np.trace(self.confusion_matrix)  # Sum of diagonal elements
        overall_accuracy = correct_predictions / total_samples
        
        # Calculate metrics for each class
        n_classes = len(self.labels)
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_score = np.zeros(n_classes)
        support = np.zeros(n_classes)
        
        for i in range(n_classes):
            # True Positive
            tp = self.confusion_matrix[i, i]
            
            # False Positive (predicted as i but actually not i)
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            
            # False Negative (actually i but predicted as not i)
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            # Support (number of samples actually belonging to class i)
            support[i] = np.sum(self.confusion_matrix[i, :])
            
            # Calculate precision, recall, f1
            if tp + fp > 0:
                precision[i] = tp / (tp + fp)
            else:
                precision[i] = 0
                
            if tp + fn > 0:
                recall[i] = tp / (tp + fn)
            else:
                recall[i] = 0
                
            if precision[i] + recall[i] > 0:
                f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1_score[i] = 0
        
        # Calculate macro and weighted averages
        macro_precision = np.mean(precision[support > 0])  # Only calculate for classes with samples
        macro_recall = np.mean(recall[support > 0])
        macro_f1 = np.mean(f1_score[support > 0])
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1_score, weights=support)
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'total_samples': int(total_samples),
            'correct_predictions': int(correct_predictions),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support.astype(int),
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            }
        }
        
        return metrics
    
    def print_metrics_summary(self):
        """Print evaluation metrics summary and automatically save to file"""
        metrics = self.calculate_metrics()
        
        summary_text = []
        summary_text.append("="*60)
        summary_text.append("CONFUSION MATRIX EVALUATION REPORT")
        summary_text.append("="*60)
        summary_text.append(f"Total samples: {metrics['total_samples']:,}")
        summary_text.append(f"Correct predictions: {metrics['correct_predictions']:,}")
        summary_text.append(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        summary_text.append("")
        
        summary_text.append("Average metrics:")
        summary_text.append(f"  Macro Average:")
        summary_text.append(f"    Precision: {metrics['macro_avg']['precision']:.4f}")
        summary_text.append(f"    Recall: {metrics['macro_avg']['recall']:.4f}")
        summary_text.append(f"    F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
        summary_text.append("")
        summary_text.append(f"  Weighted Average:")
        summary_text.append(f"    Precision: {metrics['weighted_avg']['precision']:.4f}")
        summary_text.append(f"    Recall: {metrics['weighted_avg']['recall']:.4f}")
        summary_text.append(f"    F1-Score: {metrics['weighted_avg']['f1_score']:.4f}")
        summary_text.append("")
        
        # Find best and worst performing classes
        non_zero_mask = metrics['support'] > 0
        if np.any(non_zero_mask):
            valid_f1 = metrics['f1_score'][non_zero_mask]
            valid_labels = [self.labels[i] for i in range(len(self.labels)) if non_zero_mask[i]]
            
            if len(valid_f1) > 0:
                best_idx = np.argmax(valid_f1)
                worst_idx = np.argmin(valid_f1)
                
                summary_text.append(f"Best performing class: {valid_labels[best_idx]} (F1: {valid_f1[best_idx]:.4f})")
                summary_text.append(f"Worst performing class: {valid_labels[worst_idx]} (F1: {valid_f1[worst_idx]:.4f})")
        
        # Print to console
        for line in summary_text:
            print(line)
        
        # Automatically save to file
        summary_file = self.get_output_path("metrics_summary.txt")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_text))
            print(f"\nEvaluation metrics summary saved to: {summary_file}")
        except Exception as e:
            print(f"Error saving evaluation metrics summary: {e}")
    
    def plot_confusion_matrix_heatmap(self, figsize=(15, 12)):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=figsize)
        
        # Use log scale to better display differences
        matrix_log = np.log10(self.confusion_matrix + 1)  # +1 to avoid log(0)
        
        sns.heatmap(matrix_log, 
                   xticklabels=self.labels, 
                   yticklabels=self.labels,
                   annot=False,  # Don't show values for large matrices
                   cmap='Blues',
                   cbar_kws={'label': 'Log10(Count + 1)'})
        
        plt.title('Confusion Matrix Heatmap (Log Scale)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.get_output_path("confusion_matrix_heatmap.png")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion Matrix heatmap saved to: {save_path}")
        except Exception as e:
            print(f"Error saving Confusion Matrix heatmap: {e}")
        
        plt.show()
    
    def plot_class_performance(self, figsize=(15, 10)):
        """Plot performance metrics for each class"""
        metrics = self.calculate_metrics()
        
        # Only show classes with samples
        non_zero_mask = metrics['support'] > 0
        valid_labels = [self.labels[i] for i in range(len(self.labels)) if non_zero_mask[i]]
        valid_precision = metrics['precision'][non_zero_mask]
        valid_recall = metrics['recall'][non_zero_mask]
        valid_f1 = metrics['f1_score'][non_zero_mask]
        valid_support = metrics['support'][non_zero_mask]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Precision per class
        bars1 = ax1.bar(range(len(valid_labels)), valid_precision, alpha=0.7, color='skyblue')
        ax1.set_title('Precision per Class', fontweight='bold')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(range(len(valid_labels)))
        ax1.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        
        # 2. Recall per class
        bars2 = ax2.bar(range(len(valid_labels)), valid_recall, alpha=0.7, color='lightcoral')
        ax2.set_title('Recall per Class', fontweight='bold')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(range(len(valid_labels)))
        ax2.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        
        # 3. F1-Score per class
        bars3 = ax3.bar(range(len(valid_labels)), valid_f1, alpha=0.7, color='lightgreen')
        ax3.set_title('F1-Score per Class', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(range(len(valid_labels)))
        ax3.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)
        
        # 4. Support per class
        bars4 = ax4.bar(range(len(valid_labels)), valid_support, alpha=0.7, color='gold')
        ax4.set_title('Support (Sample Count) per Class', fontweight='bold')
        ax4.set_ylabel('Number of Samples')
        ax4.set_xticks(range(len(valid_labels)))
        ax4.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.get_output_path("class_performance.png")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class performance plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving class performance plot: {e}")
        
        plt.show()
    
    def plot_error_analysis(self, figsize=(12, 8)):
        """Analyze classification error patterns"""
        # Calculate error distribution for each class
        errors_per_class = []
        true_labels = []
        
        for i in range(len(self.labels)):
            total_true = np.sum(self.confusion_matrix[i, :])
            correct = self.confusion_matrix[i, i]
            errors = total_true - correct
            
            if total_true > 0:  # Only consider classes with samples
                errors_per_class.append(errors)
                true_labels.append(self.labels[i])
        
        # Plot error analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Number of errors per class
        ax1.bar(range(len(true_labels)), errors_per_class, alpha=0.7, color='salmon')
        ax1.set_title('Classification Errors per Class', fontweight='bold')
        ax1.set_ylabel('Number of Errors')
        ax1.set_xticks(range(len(true_labels)))
        ax1.set_xticklabels(true_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Error rate distribution
        error_rates = []
        for i in range(len(self.labels)):
            total_true = np.sum(self.confusion_matrix[i, :])
            if total_true > 0:
                correct = self.confusion_matrix[i, i]
                error_rate = (total_true - correct) / total_true
                error_rates.append(error_rate)
        
        ax2.hist(error_rates, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Distribution of Error Rates', fontweight='bold')
        ax2.set_xlabel('Error Rate')
        ax2.set_ylabel('Number of Classes')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.get_output_path("error_analysis.png")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error analysis plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving error analysis plot: {e}")
        
        plt.show()
    
    def generate_detailed_report(self):
        """Generate detailed classification report and automatically save"""
        metrics = self.calculate_metrics()
        
        # Create detailed report DataFrame
        report_data = []
        
        for i, label in enumerate(self.labels):
            if metrics['support'][i] > 0:  # Only include classes with samples
                report_data.append({
                    'Class': label,
                    'Precision': f"{metrics['precision'][i]:.4f}",
                    'Recall': f"{metrics['recall'][i]:.4f}",
                    'F1-Score': f"{metrics['f1_score'][i]:.4f}",
                    'Support': metrics['support'][i]
                })
        
        # Add average values
        report_data.append({
            'Class': 'Macro Avg',
            'Precision': f"{metrics['macro_avg']['precision']:.4f}",
            'Recall': f"{metrics['macro_avg']['recall']:.4f}",
            'F1-Score': f"{metrics['macro_avg']['f1_score']:.4f}",
            'Support': int(np.sum(metrics['support'][metrics['support'] > 0]))
        })
        
        report_data.append({
            'Class': 'Weighted Avg',
            'Precision': f"{metrics['weighted_avg']['precision']:.4f}",
            'Recall': f"{metrics['weighted_avg']['recall']:.4f}",
            'F1-Score': f"{metrics['weighted_avg']['f1_score']:.4f}",
            'Support': int(metrics['total_samples'])
        })
        
        report_df = pd.DataFrame(report_data)
        
        csv_path = self.get_output_path("detailed_classification_report.csv")
        try:
            report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Detailed classification report saved to: {csv_path}")
        except Exception as e:
            print(f"Error saving detailed classification report: {e}")
        
        return report_df

# Usage example
if __name__ == "__main__":
    analyzer = ConfusionMatrixAnalyzer(r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\InsightFace\result_s2ep10\confusion_matrix_s2ep10.xlsx")
    
    print("="*80)
    print("Starting Confusion Matrix analysis...")
    print("="*80)
    
    # Print evaluation metrics summary
    analyzer.print_metrics_summary()
    
    print("\n" + "="*60)
    print("Generating visualization charts...")
    print("="*60)
    
    # 1. Confusion Matrix heatmap
    print("\n1. Generating Confusion Matrix heatmap...")
    analyzer.plot_confusion_matrix_heatmap(figsize=(15, 12))
    
    # 2. Class performance plots
    print("\n2. Generating class performance plots...")
    analyzer.plot_class_performance(figsize=(15, 10))
    
    # 3. Error analysis plots
    print("\n3. Generating error analysis plots...")
    analyzer.plot_error_analysis(figsize=(12, 8))
    
    # 4. Generate detailed report and save to CSV
    print("\n4. Generating detailed classification report...")
    detailed_report = analyzer.generate_detailed_report()
    print("\nDetailed classification report:")
    print(detailed_report.to_string(index=False))
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)
    print(f"All files saved to directory: {analyzer.output_dir}")
    print("Generated files include:")
    print(f"- {analyzer.timestamp}_metrics_summary.txt (Evaluation metrics summary)")
    print(f"- {analyzer.timestamp}_confusion_matrix_heatmap.png (Confusion matrix heatmap)")
    print(f"- {analyzer.timestamp}_class_performance.png (Class performance plots)")
    print(f"- {analyzer.timestamp}_error_analysis.png (Error analysis plots)")
    print(f"- {analyzer.timestamp}_detailed_classification_report.csv (Detailed classification report)")
    print("="*80)