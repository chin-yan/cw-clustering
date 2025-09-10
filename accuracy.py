#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Excel Confusion Matrix Analysis Tool
Handles dimension mismatches and provides comprehensive visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_excel_confusion_matrix(excel_path):
    """
    Load confusion matrix from Excel file with robust dimension handling
    """
    print(f"📊 Loading Excel file: {excel_path}")
    
    try:
        # Read Excel file with first column as index
        df = pd.read_excel(excel_path, index_col=0)
        print(f"   Original shape: {df.shape}")
        
        # Transpose the matrix
        df = df.T
        print(f"   After transpose - Shape: {df.shape}")
        
        # Get labels
        true_labels = [str(label) for label in df.index.tolist()]
        pred_labels = [str(label) for label in df.columns.tolist()]
        
        # Convert to numeric
        confusion_matrix = pd.DataFrame(df.values).apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        # CRITICAL FIX: Ensure square matrix
        rows, cols = confusion_matrix.shape
        min_dim = min(rows, cols, len(true_labels), len(pred_labels))
        
        print(f"   Matrix dimensions: {rows}x{cols}")
        print(f"   Label counts: {len(true_labels)} true, {len(pred_labels)} pred")
        
        if rows != cols or rows != len(true_labels) or cols != len(pred_labels):
            print(f"   ⚠️ Dimension mismatch detected!")
            print(f"   🔧 Trimming all to {min_dim}x{min_dim}")
            
            confusion_matrix = confusion_matrix[:min_dim, :min_dim]
            true_labels = true_labels[:min_dim]
            pred_labels = pred_labels[:min_dim]
        
        print(f"✅ Final matrix shape: {confusion_matrix.shape}")
        print(f"✅ Final label counts: {len(true_labels)} each")
        print(f"✅ Total samples: {np.sum(confusion_matrix)}")
        
        return confusion_matrix, true_labels, pred_labels
        
    except Exception as e:
        print(f"❌ Failed to load Excel file: {e}")
        raise

def calculate_accuracy_metrics(confusion_matrix, true_labels, pred_labels):
    """
    Calculate accuracy metrics with dimension safety
    """
    total_samples = np.sum(confusion_matrix)
    
    if total_samples == 0:
        print("❌ Error: No samples found in confusion matrix")
        return None
    
    print(f"🧮 Calculating metrics for {confusion_matrix.shape[0]}x{confusion_matrix.shape[1]} matrix")
    
    # Overall Accuracy
    correct_predictions = np.trace(confusion_matrix)
    overall_accuracy = correct_predictions / total_samples
    
    # Per-class metrics - use matrix dimensions, not label length
    n_classes = confusion_matrix.shape[0]  # Use actual matrix dimension
    per_class_metrics = {}
    
    for i in range(n_classes):
        # Use available label or create one
        true_label = true_labels[i] if i < len(true_labels) else str(i)
        
        # Calculate metrics safely
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(confusion_matrix[i, :])
        
        per_class_metrics[true_label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Macro averages (only for classes with support > 0)
    active_metrics = [v for v in per_class_metrics.values() if v['support'] > 0]
    
    if active_metrics:
        macro_precision = np.mean([v['precision'] for v in active_metrics])
        macro_recall = np.mean([v['recall'] for v in active_metrics])
        macro_f1 = np.mean([v['f1_score'] for v in active_metrics])
    else:
        macro_precision = macro_recall = macro_f1 = 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct_predictions,
        'total_samples': total_samples,
        'per_class_metrics': per_class_metrics,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

def create_safe_confusion_matrix_analysis(confusion_matrix, true_labels, pred_labels, output_dir):
    """
    Create confusion matrix analysis with complete dimension safety
    """
    print("🔥 Creating SAFE confusion matrix analysis...")
    
    # Ensure all dimensions match
    n_classes = confusion_matrix.shape[0]
    
    print(f"   Matrix: {confusion_matrix.shape}")
    print(f"   Using {n_classes} classes for analysis")
    
    # Create safe labels array
    safe_labels = []
    for i in range(n_classes):
        if i < len(true_labels):
            safe_labels.append(str(true_labels[i]))
        else:
            safe_labels.append(f"Class_{i}")
    
    # Find non-zero classes SAFELY
    row_sums = np.sum(confusion_matrix, axis=1)  # Shape: (n_classes,)
    col_sums = np.sum(confusion_matrix, axis=0)  # Shape: (n_classes,)
    
    # Now both arrays have the same shape
    non_zero_indices = np.where((row_sums > 0) | (col_sums > 0))[0]
    
    if len(non_zero_indices) == 0:
        print("❌ No non-zero classes found")
        return
    
    print(f"   Found {len(non_zero_indices)} non-zero classes")
    
    # Limit display size
    if len(non_zero_indices) > 25:
        class_totals = row_sums + col_sums
        top_indices = np.argsort(class_totals)[-25:]
        display_matrix = confusion_matrix[np.ix_(top_indices, top_indices)]
        display_labels = [safe_labels[i] for i in top_indices]
        title_suffix = " (Top 25 Most Active Classes)"
    else:
        display_matrix = confusion_matrix[np.ix_(non_zero_indices, non_zero_indices)]
        display_labels = [safe_labels[i] for i in non_zero_indices]
        title_suffix = ""
    
    print(f"   Display matrix: {display_matrix.shape}")
    print(f"   Display labels: {len(display_labels)}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Confusion Matrix Analysis{title_suffix}', fontsize=16, fontweight='bold')
    
    # 1. Raw Confusion Matrix
    ax1 = axes[0, 0]
    im1 = ax1.imshow(display_matrix, interpolation='nearest', cmap='Blues')
    ax1.set_title('Raw Confusion Matrix', fontweight='bold')
    
    tick_marks = np.arange(len(display_labels))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(display_labels, rotation=45, ha='right')
    ax1.set_yticklabels(display_labels)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Add annotations for smaller matrices
    if len(display_labels) <= 15:
        thresh = display_matrix.max() / 2.
        for i in range(len(display_labels)):
            for j in range(len(display_labels)):
                ax1.text(j, i, format(int(display_matrix[i, j]), 'd'),
                        ha="center", va="center", fontsize=8,
                        color="white" if display_matrix[i, j] > thresh else "black")
    
    plt.colorbar(im1, ax=ax1)
    
    # 2. Normalized Confusion Matrix
    ax2 = axes[0, 1]
    row_sums_display = np.maximum(display_matrix.sum(axis=1), 1)
    normalized_matrix = display_matrix / row_sums_display[:, np.newaxis]
    
    im2 = ax2.imshow(normalized_matrix, interpolation='nearest', cmap='Blues')
    ax2.set_title('Normalized Confusion Matrix', fontweight='bold')
    
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(display_labels, rotation=45, ha='right')
    ax2.set_yticklabels(display_labels)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    if len(display_labels) <= 15:
        for i in range(len(display_labels)):
            for j in range(len(display_labels)):
                ax2.text(j, i, format(normalized_matrix[i, j], '.2f'),
                        ha="center", va="center", fontsize=8,
                        color="white" if normalized_matrix[i, j] > 0.5 else "black")
    
    plt.colorbar(im2, ax=ax2)
    
    # 3. Error Rate by Class
    ax3 = axes[1, 0]
    diagonal = np.diag(display_matrix)
    row_sums_display = np.maximum(display_matrix.sum(axis=1), 1)
    error_rates = 1 - (diagonal / row_sums_display)
    
    bars = ax3.bar(range(len(display_labels)), error_rates, color='lightcoral', alpha=0.7)
    ax3.set_title('Error Rate by Class', fontweight='bold')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Error Rate')
    ax3.set_xticks(range(len(display_labels)))
    ax3.set_xticklabels(display_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Most Confused Class Pairs
    ax4 = axes[1, 1]
    
    confusion_pairs = []
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            if i != j and display_matrix[i, j] > 0:
                confusion_pairs.append({
                    'true_class': display_labels[i],
                    'pred_class': display_labels[j],
                    'count': display_matrix[i, j]
                })
    
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)[:10]
    
    if confusion_pairs:
        pair_labels = [f"{p['true_class']}→{p['pred_class']}" for p in confusion_pairs]
        pair_counts = [p['count'] for p in confusion_pairs]
        
        bars = ax4.barh(range(len(pair_labels)), pair_counts, color='orange', alpha=0.7)
        ax4.set_title('Top 10 Confusion Pairs', fontweight='bold')
        ax4.set_xlabel('Number of Misclassifications')
        ax4.set_yticks(range(len(pair_labels)))
        ax4.set_yticklabels(pair_labels)
        ax4.grid(True, alpha=0.3, axis='x')
        
        for bar, count in zip(bars, pair_counts):
            width = bar.get_width()
            ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{int(count)}', ha='left', va='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No significant confusion pairs found', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Confusion Pairs Analysis', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    confusion_path = output_dir / 'safe_confusion_matrix_analysis.png'
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Safe confusion matrix analysis saved to: {confusion_path}")

def create_performance_dashboard(metrics, output_dir):
    """
    Create main performance dashboard
    """
    print("📊 Creating performance dashboard...")
    
    active_metrics = {k: v for k, v in metrics['per_class_metrics'].items() if v['support'] > 0}
    
    if not active_metrics:
        print("❌ No active classes to visualize")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Face Clustering Performance Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Overall metrics
    ax1 = axes[0, 0]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metrics_values = [
        metrics['overall_accuracy'],
        metrics['macro_precision'],
        metrics['macro_recall'],
        metrics['macro_f1']
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_title('Overall Performance Metrics', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Support distribution
    ax2 = axes[0, 1]
    supports = [v['support'] for v in active_metrics.values()]
    ax2.hist(supports, bins=min(20, max(len(supports)//3, 5)), alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Class Support Distribution', fontweight='bold')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Number of Classes')
    ax2.grid(True, alpha=0.3)
    
    # 3. F1 vs Support scatter
    ax3 = axes[0, 2]
    f1_scores = [v['f1_score'] for v in active_metrics.values()]
    precisions = [v['precision'] for v in active_metrics.values()]
    
    scatter = ax3.scatter(supports, f1_scores, c=precisions, s=60, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Support')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score vs Support (Color = Precision)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Precision')
    
    # 4. Top performing classes
    ax4 = axes[1, 0]
    sorted_by_f1 = sorted(active_metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:10]
    
    class_names = [f'Class {k}' for k, v in sorted_by_f1]
    f1_values = [v['f1_score'] for k, v in sorted_by_f1]
    
    bars = ax4.barh(range(len(class_names)), f1_values, color='lightgreen', alpha=0.8)
    ax4.set_yticks(range(len(class_names)))
    ax4.set_yticklabels(class_names)
    ax4.set_xlabel('F1 Score')
    ax4.set_title('Top 10 Performing Classes', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Performance distribution
    ax5 = axes[1, 1]
    precisions = [v['precision'] for v in active_metrics.values()]
    recalls = [v['recall'] for v in active_metrics.values()]
    
    box_data = [precisions, recalls, f1_scores]
    box_labels = ['Precision', 'Recall', 'F1 Score']
    
    box_plot = ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('Score')
    ax5.set_title('Performance Distribution', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    perfect_precision = sum(1 for v in active_metrics.values() if v['precision'] == 1.0)
    perfect_recall = sum(1 for v in active_metrics.values() if v['recall'] == 1.0)
    perfect_f1 = sum(1 for v in active_metrics.values() if v['f1_score'] == 1.0)
    
    summary_text = f"""
PERFORMANCE SUMMARY
==================

Total Samples: {int(metrics['total_samples']):,}
Overall Accuracy: {metrics['overall_accuracy']:.3f}

Active Classes: {len(active_metrics)}
Total Classes: {len(metrics['per_class_metrics'])}

Macro Averages:
• Precision: {metrics['macro_precision']:.3f}
• Recall: {metrics['macro_recall']:.3f}
• F1 Score: {metrics['macro_f1']:.3f}

Class Statistics:
• Largest: {max(supports)} samples
• Smallest: {min(supports)} samples
• Median: {np.median(supports):.0f} samples

Perfect Scores:
• Perfect Precision: {perfect_precision}
• Perfect Recall: {perfect_recall}
• Perfect F1: {perfect_f1}

Low Performance:
• F1 < 0.5: {sum(1 for v in active_metrics.values() if v['f1_score'] < 0.5)}
• Support < 10: {sum(1 for v in active_metrics.values() if v['support'] < 10)}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    dashboard_path = output_dir / 'performance_dashboard.png'
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance dashboard saved to: {dashboard_path}")

def print_detailed_report(metrics):
    """
    Print detailed evaluation report
    """
    print("\n" + "="*60)
    print("📊 ACCURACY ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Overall Accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"   Correct Predictions: {int(metrics['correct_predictions']):,}")
    print(f"   Total Samples: {int(metrics['total_samples']):,}")
    
    print(f"\n📈 Average Metrics:")
    print(f"   Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"   Macro Recall: {metrics['macro_recall']:.3f}")
    print(f"   Macro F1-Score: {metrics['macro_f1']:.3f}")
    
    # Performance evaluation
    accuracy = metrics['overall_accuracy']
    if accuracy >= 0.95:
        print("   ✅ Excellent (≥95%) - Ready for production")
    elif accuracy >= 0.90:
        print("   ✅ Good (≥90%) - Suitable for most applications")
    elif accuracy >= 0.80:
        print("   ⚠️ Average (≥80%) - Needs improvement")
    elif accuracy >= 0.70:
        print("   ⚠️ Below Average (≥70%) - Recommend parameter tuning")
    else:
        print("   ❌ Poor (<70%) - System redesign needed")
    
    # Count classes with data
    active_classes = [k for k, v in metrics['per_class_metrics'].items() if v['support'] > 0]
    
    print(f"\n📊 Class Statistics:")
    print(f"   Total Classes: {len(metrics['per_class_metrics'])}")
    print(f"   Active Classes: {len(active_classes)}")
    print(f"   Empty Classes: {len(metrics['per_class_metrics']) - len(active_classes)}")
    
    # Top classes by support
    if active_classes:
        print(f"\n📋 Top 10 Classes by Sample Count:")
        sorted_classes = sorted([(k, v) for k, v in metrics['per_class_metrics'].items() if v['support'] > 0], 
                              key=lambda x: x[1]['support'], reverse=True)[:10]
        
        print(f"{'Class':<8} {'Support':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<9}")
        print("-" * 50)
        
        for class_name, class_metrics in sorted_classes:
            print(f"{str(class_name):<8} "
                  f"{int(class_metrics['support']):<8} "
                  f"{class_metrics['precision']:<10.3f} "
                  f"{class_metrics['recall']:<8.3f} "
                  f"{class_metrics['f1_score']:<9.3f}")

def save_results(metrics, output_dir):
    """
    Save results in multiple formats
    """
    print("\n💾 Saving results...")
    
    try:
        # Create detailed CSV
        detailed_data = []
        for class_id, class_metrics in metrics['per_class_metrics'].items():
            detailed_data.append({
                'class_id': class_id,
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1_score': class_metrics['f1_score'],
                'support': class_metrics['support']
            })
        
        df = pd.DataFrame(detailed_data)
        csv_path = output_dir / 'detailed_class_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV saved: {csv_path}")
        
        # Create accuracy results in original format
        accuracy_results_path = output_dir / 'accuracy_results.txt'
        with open(accuracy_results_path, 'w', encoding='utf-8') as f:
            f.write("Face Clustering Accuracy Analysis Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Correct Predictions: {metrics['correct_predictions']}\n")
            f.write(f"Total Samples: {metrics['total_samples']}\n\n")
            
            f.write("Per-class metrics:\n")
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                f.write(f"{class_name}: Precision={class_metrics['precision']:.3f}, "
                       f"Recall={class_metrics['recall']:.3f}, "
                       f"F1={class_metrics['f1_score']:.3f}, "
                       f"Support={class_metrics['support']}\n")
        
        print(f"✅ Accuracy results saved: {accuracy_results_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """
    Fixed main function with comprehensive error handling
    """
    print("🔧 FIXED Face Clustering Accuracy Calculator")
    print("="*60)
    print("This version handles all dimension mismatch issues")
    print()
    
    # Get file path
    file_path = input("Please enter Excel file path: ").strip('"')
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Create output directory
    output_dir = Path(file_path).parent / "fixed_analysis_results"
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()
    
    try:
        # 1. Load confusion matrix with dimension fixes
        confusion_matrix, true_labels, pred_labels = load_excel_confusion_matrix(file_path)
        
        # 2. Calculate metrics
        metrics = calculate_accuracy_metrics(confusion_matrix, true_labels, pred_labels)
        
        if metrics is None:
            print("❌ Failed to calculate metrics")
            return
        
        # 3. Print basic report
        print_detailed_report(metrics)
        
        # 4. Create visualizations safely
        print(f"\n🎨 Creating visualizations...")
        
        # Performance Dashboard
        create_performance_dashboard(metrics, output_dir)
        
        # Safe Confusion Matrix Analysis
        create_safe_confusion_matrix_analysis(confusion_matrix, true_labels, pred_labels, output_dir)
        
        # 5. Save results
        save_results(metrics, output_dir)
        
        # 6. Final summary
        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Overall Accuracy: {metrics['overall_accuracy']:.1%}")
        print(f"📁 All results saved to: {output_dir}")
        print(f"\n📄 Generated Files:")
        print(f"   • performance_dashboard.png - Main performance dashboard")
        print(f"   • safe_confusion_matrix_analysis.png - Fixed confusion matrix analysis")
        print(f"   • detailed_class_metrics.csv - Metrics in CSV format")
        print(f"   • accuracy_results.txt - Original format results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()