#!/usr/bin/env python3
"""
Season 1 Face Clustering Experiment Analysis Tool
Analyzes 13 episodes of Season 1 with comprehensive statistics and visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better visualization
plt.style.use('default')
sns.set_palette("husl")

class S1EpisodeAnalyzer:
    """Season 1 Episode Analysis Tool"""
    
    def __init__(self, results_dir):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing episode result folders
        """
        self.results_dir = Path(results_dir)
        self.episode_data = {}
        self.summary_df = None
        self.output_dir = self.results_dir / "season1_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_episode_results(self):
        """Load all episode results from the directory structure"""
        print("Loading episode results...")
        
        # Look for episode folders
        episode_folders = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and 'ep' in item.name.lower():
                episode_folders.append(item)
        
        # Sort folders to ensure correct order
        episode_folders.sort(key=lambda x: self._extract_episode_number(x.name))
        
        for folder in episode_folders:
            ep_num = self._extract_episode_number(folder.name)
            if ep_num:
                metrics_file = folder / "metrics_summary.txt"
                if metrics_file.exists():
                    episode_result = self._parse_metrics_file(metrics_file, ep_num)
                    if episode_result:
                        self.episode_data[ep_num] = episode_result
                        print(f"Loaded Episode {ep_num}")
        
        print(f"Successfully loaded {len(self.episode_data)} episodes")
        return len(self.episode_data) > 0
    
    def _extract_episode_number(self, folder_name):
        """Extract episode number from folder name"""
        patterns = [
            r'ep(\d+)',
            r'episode(\d+)',
            r'ep_(\d+)',
            r'episode_(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, folder_name.lower())
            if match:
                return int(match.group(1))
        return None
    
    def _parse_metrics_file(self, file_path, episode_num):
        """Parse metrics summary file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = {'episode': episode_num}
            
            # Extract total samples
            total_match = re.search(r'Total samples:\s*(\d+)', content)
            if total_match:
                result['total_samples'] = int(total_match.group(1))
            
            # Extract correct predictions
            correct_match = re.search(r'Correct predictions:\s*(\d+)', content)
            if correct_match:
                result['correct_predictions'] = int(correct_match.group(1))
            
            # Extract overall accuracy
            accuracy_match = re.search(r'Overall Accuracy:\s*([\d.]+)', content)
            if accuracy_match:
                result['overall_accuracy'] = float(accuracy_match.group(1))
            
            # Extract Weighted Average metrics
            weighted_section = re.search(r'Weighted Average:(.*?)(?=Best performing|$)', content, re.DOTALL)
            if weighted_section:
                weighted_text = weighted_section.group(1)
                
                precision_match = re.search(r'Precision:\s*([\d.]+)', weighted_text)
                if precision_match:
                    result['weighted_precision'] = float(precision_match.group(1))
                
                recall_match = re.search(r'Recall:\s*([\d.]+)', weighted_text)
                if recall_match:
                    result['weighted_recall'] = float(recall_match.group(1))
                
                f1_match = re.search(r'F1-Score:\s*([\d.]+)', weighted_text)
                if f1_match:
                    result['weighted_f1'] = float(f1_match.group(1))
            
            # Extract Macro Average metrics
            macro_section = re.search(r'Macro Average:(.*?)Weighted Average:', content, re.DOTALL)
            if macro_section:
                macro_text = macro_section.group(1)
                
                precision_match = re.search(r'Precision:\s*([\d.]+)', macro_text)
                if precision_match:
                    result['macro_precision'] = float(precision_match.group(1))
                
                recall_match = re.search(r'Recall:\s*([\d.]+)', macro_text)
                if recall_match:
                    result['macro_recall'] = float(recall_match.group(1))
                
                f1_match = re.search(r'F1-Score:\s*([\d.]+)', macro_text)
                if f1_match:
                    result['macro_f1'] = float(f1_match.group(1))
            
            # Extract best and worst performing classes
            best_match = re.search(r'Best performing class:\s*([\d.]+)', content)
            if best_match:
                result['best_class'] = float(best_match.group(1))
            
            worst_match = re.search(r'Worst performing class:\s*([\d.]+)', content)
            if worst_match:
                result['worst_class'] = float(worst_match.group(1))
            
            return result
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def create_summary_dataframe(self):
        """Create summary dataframe from episode data"""
        if not self.episode_data:
            print("No episode data loaded")
            return None
        
        # Prepare data for DataFrame
        summary_data = []
        for ep_num in sorted(self.episode_data.keys()):
            data = self.episode_data[ep_num]
            summary_data.append({
                'Episode': ep_num,
                'Total_Samples': data.get('total_samples', 0),
                'Correct_Predictions': data.get('correct_predictions', 0),
                'Overall_Accuracy': data.get('overall_accuracy', 0),
                'Weighted_Precision': data.get('weighted_precision', 0),
                'Weighted_Recall': data.get('weighted_recall', 0),
                'Weighted_F1': data.get('weighted_f1', 0),
                'Macro_Precision': data.get('macro_precision', 0),
                'Macro_Recall': data.get('macro_recall', 0),
                'Macro_F1': data.get('macro_f1', 0),
                'Best_Class': data.get('best_class', 0),
                'Worst_Class': data.get('worst_class', 0)
            })
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Calculate additional metrics
        self.summary_df['Error_Rate'] = 1 - self.summary_df['Overall_Accuracy']
        self.summary_df['Error_Count'] = self.summary_df['Total_Samples'] - self.summary_df['Correct_Predictions']
        
        return self.summary_df
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        print("\n" + "="*70)
        print("SEASON 1 COMPREHENSIVE ANALYSIS REPORT")
        print("="*70)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Episodes: {len(self.episode_data)}")
        print(f"Total Face Samples: {self.summary_df['Total_Samples'].sum():,}")
        print(f"Total Correct Predictions: {self.summary_df['Correct_Predictions'].sum():,}")
        print(f"Average Samples per Episode: {self.summary_df['Total_Samples'].mean():.0f}")
        
        print("\n" + "-"*50)
        print("WEIGHTED AVERAGE PERFORMANCE METRICS")
        print("-"*50)
        
        weighted_metrics = ['Overall_Accuracy', 'Weighted_Precision', 'Weighted_Recall', 'Weighted_F1']
        
        for metric in weighted_metrics:
            values = self.summary_df[metric] * 100  # Convert to percentage
            print(f"{metric.replace('_', ' ')}:")
            print(f"  Mean: {values.mean():.2f}% ± {values.std():.2f}%")
            print(f"  Range: {values.min():.2f}% - {values.max():.2f}%")
            print(f"  Best Episode: Ep{self.summary_df.loc[values.idxmax(), 'Episode']} ({values.max():.2f}%)")
            print(f"  Worst Episode: Ep{self.summary_df.loc[values.idxmin(), 'Episode']} ({values.min():.2f}%)")
            print()
        
        print("-"*50)
        print("EPISODE RANKINGS (by Weighted F1-Score)")
        print("-"*50)
        
        ranked_episodes = self.summary_df.sort_values('Weighted_F1', ascending=False)
        for i, (_, row) in enumerate(ranked_episodes.iterrows(), 1):
            print(f"Rank {i:2d}: Episode {int(row['Episode']):2d} - {row['Weighted_F1']*100:6.2f}%")
        
        print("-"*50)
        print("PERFORMANCE CONSISTENCY ANALYSIS")
        print("-"*50)
        
        f1_std = self.summary_df['Weighted_F1'].std() * 100
        accuracy_std = self.summary_df['Overall_Accuracy'].std() * 100
        
        print(f"Weighted F1-Score Std Dev: {f1_std:.2f}%")
        print(f"Overall Accuracy Std Dev: {accuracy_std:.2f}%")
        
        if f1_std < 5:
            consistency = "High"
        elif f1_std < 10:
            consistency = "Medium"
        else:
            consistency = "Low"
        
        print(f"Performance Consistency: {consistency}")
        
        # Correlation analysis
        corr_samples_f1 = np.corrcoef(self.summary_df['Total_Samples'], self.summary_df['Weighted_F1'])[0, 1]
        print(f"Correlation (Samples vs F1-Score): {corr_samples_f1:.4f}")
        
        return self.summary_df
    
    def plot_performance_overview(self, save_plots=True):
        """Create comprehensive performance overview plots"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        episodes = self.summary_df['Episode']
        
        # 1. Overall Accuracy Trend
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, self.summary_df['Overall_Accuracy'] * 100, 
                marker='o', linewidth=3, markersize=8, color='#2E86AB')
        ax1.fill_between(episodes, self.summary_df['Overall_Accuracy'] * 100, 
                        alpha=0.3, color='#2E86AB')
        ax1.set_title('Overall Accuracy Trend', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(70, 100)
        
        # Add mean line
        mean_acc = self.summary_df['Overall_Accuracy'].mean() * 100
        ax1.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_acc:.2f}%')
        ax1.legend()
        
        # 2. Weighted Metrics Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(episodes, self.summary_df['Weighted_Precision'] * 100, 
                marker='s', label='Precision', linewidth=2, markersize=6)
        ax2.plot(episodes, self.summary_df['Weighted_Recall'] * 100, 
                marker='^', label='Recall', linewidth=2, markersize=6)
        ax2.plot(episodes, self.summary_df['Weighted_F1'] * 100, 
                marker='o', label='F1-Score', linewidth=2, markersize=6)
        
        ax2.set_title('Weighted Average Metrics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(60, 100)
        
        # 3. Sample Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(episodes, self.summary_df['Total_Samples'], 
                      alpha=0.7, color='#A23B72')
        ax3.set_title('Face Samples per Episode', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Sample Count')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, self.summary_df['Total_Samples']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                    f'{value}', ha='center', va='bottom', fontsize=9)
        
        # 4. Performance vs Sample Count Scatter
        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(self.summary_df['Total_Samples'], 
                            self.summary_df['Weighted_F1'] * 100,
                            c=episodes, cmap='viridis', s=100, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(self.summary_df['Total_Samples'], 
                      self.summary_df['Weighted_F1'] * 100, 1)
        p = np.poly1d(z)
        ax4.plot(self.summary_df['Total_Samples'], p(self.summary_df['Total_Samples']), 
                "r--", alpha=0.8, linewidth=2)
        
        ax4.set_title('F1-Score vs Sample Count', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sample Count')
        ax4.set_ylabel('Weighted F1-Score (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add episode labels
        for i, ep in enumerate(episodes):
            ax4.annotate(f'Ep{ep}', 
                        (self.summary_df.iloc[i]['Total_Samples'], 
                         self.summary_df.iloc[i]['Weighted_F1'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Episode')
        
        # 5. Error Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(episodes, self.summary_df['Error_Count'], 
               alpha=0.7, color='#E74C3C')
        ax5.set_title('Error Count per Episode', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Error Count')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Weighted vs Macro F1 Comparison
        ax6 = fig.add_subplot(gs[1, 2])
        x = np.arange(len(episodes))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, self.summary_df['Weighted_F1'] * 100, 
                       width, label='Weighted F1', alpha=0.8)
        bars2 = ax6.bar(x + width/2, self.summary_df['Macro_F1'] * 100, 
                       width, label='Macro F1', alpha=0.8)
        
        ax6.set_title('Weighted vs Macro F1-Score', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('F1-Score (%)')
        ax6.set_xticks(x)
        ax6.set_xticklabels([f'Ep{ep}' for ep in episodes])
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Performance Distribution Box Plot
        ax7 = fig.add_subplot(gs[2, 0])
        performance_data = [
            self.summary_df['Overall_Accuracy'] * 100,
            self.summary_df['Weighted_Precision'] * 100,
            self.summary_df['Weighted_Recall'] * 100,
            self.summary_df['Weighted_F1'] * 100
        ]
        
        box_plot = ax7.boxplot(performance_data, 
                              labels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                              patch_artist=True)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax7.set_title('Performance Metrics Distribution', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Score (%)')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Episode Ranking
        ax8 = fig.add_subplot(gs[2, 1])
        ranked_data = self.summary_df.sort_values('Weighted_F1', ascending=True)
        
        bars = ax8.barh(range(len(episodes)), ranked_data['Weighted_F1'] * 100,
                       color=plt.cm.RdYlGn(ranked_data['Weighted_F1'] / ranked_data['Weighted_F1'].max()))
        
        ax8.set_yticks(range(len(episodes)))
        ax8.set_yticklabels([f'Episode {ep}' for ep in ranked_data['Episode']])
        ax8.set_xlabel('Weighted F1-Score (%)')
        ax8.set_title('Episode Performance Ranking', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, ranked_data['Weighted_F1'] * 100)):
            ax8.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{score:.1f}%', va='center', fontweight='bold', fontsize=9)
        
        # 9. Correlation Matrix
        ax9 = fig.add_subplot(gs[2, 2])
        corr_data = self.summary_df[['Total_Samples', 'Overall_Accuracy', 
                                   'Weighted_Precision', 'Weighted_Recall', 'Weighted_F1']].corr()
        
        im = ax9.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
        ax9.set_xticks(range(len(corr_data.columns)))
        ax9.set_yticks(range(len(corr_data.columns)))
        ax9.set_xticklabels([col.replace('_', '\n') for col in corr_data.columns], rotation=45)
        ax9.set_yticklabels([col.replace('_', '\n') for col in corr_data.columns])
        ax9.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                        ha='center', va='center', fontweight='bold',
                        color='white' if abs(corr_data.iloc[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax9, shrink=0.8)
        
        plt.suptitle('Season 1 Face Clustering Performance Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        if save_plots:
            save_path = self.output_dir / 'season1_performance_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance analysis plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_episode_comparison(self, save_plots=True):
        """Create detailed episode comparison plots"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        episodes = self.summary_df['Episode']
        
        # 1. Heatmap of all metrics
        metrics_for_heatmap = self.summary_df[['Episode', 'Overall_Accuracy', 'Weighted_Precision', 
                                             'Weighted_Recall', 'Weighted_F1']].set_index('Episode')
        
        im1 = ax1.imshow(metrics_for_heatmap.T * 100, cmap='RdYlGn', aspect='auto', vmin=60, vmax=100)
        ax1.set_xticks(range(len(episodes)))
        ax1.set_xticklabels([f'Ep{int(ep)}' for ep in episodes])
        ax1.set_yticks(range(len(metrics_for_heatmap.columns)))
        ax1.set_yticklabels([col.replace('_', ' ') for col in metrics_for_heatmap.columns])
        ax1.set_title('Episode Performance Heatmap', fontsize=14, fontweight='bold')
        
        # Add values to heatmap - fix indexing
        n_metrics = len(metrics_for_heatmap.columns)  # number of rows in the transposed matrix
        n_episodes = len(episodes)  # number of columns in the transposed matrix
        
        for i in range(n_metrics):
            for j in range(n_episodes):
                # Use .values to get numpy array and avoid pandas indexing issues
                value = metrics_for_heatmap.iloc[j, i]  # Note: j, i because we're accessing original (non-transposed) data
                text = ax1.text(j, i, f'{value*100:.1f}%',
                               ha='center', va='center', fontweight='bold', fontsize=9)
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. Performance Consistency Analysis
        rolling_mean = self.summary_df['Weighted_F1'].rolling(window=3, center=True).mean()
        
        ax2.plot(episodes, self.summary_df['Weighted_F1'] * 100, 
                marker='o', label='F1-Score', linewidth=2, alpha=0.7)
        ax2.plot(episodes, rolling_mean * 100, 
                label='3-Episode Moving Average', linewidth=3, color='red')
        
        # Add confidence interval
        f1_mean = self.summary_df['Weighted_F1'].mean() * 100
        f1_std = self.summary_df['Weighted_F1'].std() * 100
        ax2.axhline(y=f1_mean, color='green', linestyle='--', alpha=0.7, 
                   label=f'Mean: {f1_mean:.2f}%')
        ax2.fill_between(episodes, f1_mean - f1_std, f1_mean + f1_std, 
                        alpha=0.2, color='green', label=f'±1σ: {f1_std:.2f}%')
        
        ax2.set_title('F1-Score Consistency Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Weighted F1-Score (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample Efficiency Analysis
        efficiency = (self.summary_df['Weighted_F1'] * 100) / (self.summary_df['Total_Samples'] / 100)
        
        bars = ax3.bar(episodes, efficiency, alpha=0.7, 
                      color=plt.cm.viridis(efficiency / efficiency.max()))
        ax3.set_title('Sample Efficiency (F1-Score per 100 samples)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Efficiency Score')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, eff in zip(bars, efficiency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{eff:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Weighted vs Macro Performance Gap
        performance_gap = (self.summary_df['Weighted_F1'] - self.summary_df['Macro_F1']) * 100
        
        colors = ['green' if gap > 0 else 'red' for gap in performance_gap]
        bars = ax4.bar(episodes, performance_gap, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('Weighted vs Macro F1-Score Gap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Performance Gap (%)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, gap in zip(bars, performance_gap):
            y_pos = bar.get_height() + 0.5 if gap > 0 else bar.get_height() - 1
            ax4.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{gap:.1f}%', ha='center', va='bottom' if gap > 0 else 'top', fontsize=9)
        
        plt.suptitle('Detailed Episode Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            save_path = self.output_dir / 'detailed_episode_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed comparison plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_report=True):
        """Generate comprehensive text report"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SEASON 1 FACE CLUSTERING EXPERIMENT ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-"*40)
        report_lines.append(f"Episodes Analyzed: {len(self.episode_data)}")
        report_lines.append(f"Total Face Samples: {self.summary_df['Total_Samples'].sum():,}")
        report_lines.append(f"Overall Success Rate: {(self.summary_df['Correct_Predictions'].sum() / self.summary_df['Total_Samples'].sum())*100:.2f}%")
        report_lines.append(f"Average Weighted F1-Score: {self.summary_df['Weighted_F1'].mean()*100:.2f}%")
        report_lines.append(f"Performance Range: {self.summary_df['Weighted_F1'].min()*100:.2f}% - {self.summary_df['Weighted_F1'].max()*100:.2f}%")
        report_lines.append("")
        
        # Performance Statistics
        report_lines.append("DETAILED PERFORMANCE STATISTICS")
        report_lines.append("-"*40)
        
        weighted_metrics = ['Overall_Accuracy', 'Weighted_Precision', 'Weighted_Recall', 'Weighted_F1']
        for metric in weighted_metrics:
            values = self.summary_df[metric] * 100
            report_lines.append(f"{metric.replace('_', ' ')}:")
            report_lines.append(f"  Mean ± Std: {values.mean():.2f}% ± {values.std():.2f}%")
            report_lines.append(f"  Best: Episode {int(self.summary_df.loc[values.idxmax(), 'Episode'])} ({values.max():.2f}%)")
            report_lines.append(f"  Worst: Episode {int(self.summary_df.loc[values.idxmin(), 'Episode'])} ({values.min():.2f}%)")
            report_lines.append("")
        
        # Episode Rankings
        report_lines.append("EPISODE PERFORMANCE RANKINGS")
        report_lines.append("-"*40)
        ranked = self.summary_df.sort_values('Weighted_F1', ascending=False)
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            report_lines.append(f"Rank {i:2d}: Episode {int(row['Episode']):2d} - F1: {row['Weighted_F1']*100:6.2f}%")
        report_lines.append("")
        
        # Consistency Analysis
        report_lines.append("PERFORMANCE CONSISTENCY ANALYSIS")
        report_lines.append("-"*40)
        f1_std = self.summary_df['Weighted_F1'].std() * 100
        cv = f1_std / (self.summary_df['Weighted_F1'].mean() * 100) * 100
        report_lines.append(f"Weighted F1-Score Standard Deviation: {f1_std:.2f}%")
        report_lines.append(f"Coefficient of Variation: {cv:.2f}%")
        
        if cv < 10:
            consistency = "High"
        elif cv < 20:
            consistency = "Medium"
        else:
            consistency = "Low"
        
        report_lines.append(f"Performance Consistency Level: {consistency}")
        report_lines.append("")
        
        # Correlation Analysis
        report_lines.append("CORRELATION ANALYSIS")
        report_lines.append("-"*40)
        corr_samples_f1 = np.corrcoef(self.summary_df['Total_Samples'], 
                                     self.summary_df['Weighted_F1'])[0, 1]
        report_lines.append(f"Sample Count vs F1-Score: {corr_samples_f1:.4f}")
        
        if abs(corr_samples_f1) > 0.7:
            strength = "Strong"
        elif abs(corr_samples_f1) > 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        direction = "positive" if corr_samples_f1 > 0 else "negative"
        report_lines.append(f"Correlation Strength: {strength} {direction}")
        report_lines.append("")
        
        # Outlier Detection
        report_lines.append("OUTLIER DETECTION")
        report_lines.append("-"*40)
        f1_mean = self.summary_df['Weighted_F1'].mean()
        f1_std_val = self.summary_df['Weighted_F1'].std()
        
        outliers = self.summary_df[
            (self.summary_df['Weighted_F1'] < f1_mean - 2*f1_std_val) | 
            (self.summary_df['Weighted_F1'] > f1_mean + 2*f1_std_val)
        ]
        
        if len(outliers) > 0:
            report_lines.append("Episodes with exceptional performance (>2σ from mean):")
            for _, row in outliers.iterrows():
                status = "Exceptionally High" if row['Weighted_F1'] > f1_mean else "Needs Attention"
                report_lines.append(f"  Episode {int(row['Episode']):2d}: {status} ({row['Weighted_F1']*100:.2f}%)")
        else:
            report_lines.append("No significant performance outliers detected.")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-"*40)
        
        worst_episodes = ranked.tail(3)['Episode'].tolist()
        best_episodes = ranked.head(3)['Episode'].tolist()
        
        report_lines.append(f"1. Focus on underperforming episodes: {', '.join([f'Episode {int(ep)}' for ep in worst_episodes])}")
        report_lines.append(f"2. Analyze best practices from: {', '.join([f'Episode {int(ep)}' for ep in best_episodes])}")
        
        if corr_samples_f1 < -0.3:
            report_lines.append("3. Sample quality optimization recommended (negative correlation detected)")
        elif corr_samples_f1 > 0.3:
            report_lines.append("3. Consider increasing sample size for lower-performing episodes")
        
        if f1_std > 10:
            report_lines.append("4. Standardize processing parameters to improve consistency")
        
        # Check for performance gaps
        avg_gap = (self.summary_df['Weighted_F1'] - self.summary_df['Macro_F1']).mean() * 100
        if avg_gap > 10:
            report_lines.append("5. Class imbalance issues detected - consider rebalancing strategies")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        if save_report:
            report_path = self.output_dir / 'season1_analysis_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"Analysis report saved to: {report_path}")
        
        # Also save CSV data
        csv_path = self.output_dir / 'season1_summary_data.csv'
        self.summary_df.to_csv(csv_path, index=False)
        print(f"Summary data saved to: {csv_path}")
        
        # Print to console
        for line in report_lines:
            print(line)
        
        return report_lines
    
    def export_to_excel(self, filename=None):
        """Export analysis results to Excel file"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        if filename is None:
            filename = self.output_dir / 'season1_analysis.xlsx'
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main summary
                self.summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Statistics sheet
                stats_data = []
                weighted_metrics = ['Overall_Accuracy', 'Weighted_Precision', 'Weighted_Recall', 'Weighted_F1']
                
                for metric in weighted_metrics:
                    values = self.summary_df[metric] * 100
                    stats_data.append({
                        'Metric': metric.replace('_', ' '),
                        'Mean': values.mean(),
                        'Std_Dev': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'Best_Episode': self.summary_df.loc[values.idxmax(), 'Episode'],
                        'Worst_Episode': self.summary_df.loc[values.idxmin(), 'Episode']
                    })
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Rankings sheet
                ranked_df = self.summary_df.sort_values('Weighted_F1', ascending=False)[['Episode', 'Weighted_F1']]
                ranked_df['Rank'] = range(1, len(ranked_df) + 1)
                ranked_df['Weighted_F1_Percent'] = ranked_df['Weighted_F1'] * 100
                ranked_df = ranked_df[['Rank', 'Episode', 'Weighted_F1_Percent']].copy()
                ranked_df.to_excel(writer, sheet_name='Rankings', index=False)
            
            print(f"Excel file exported to: {filename}")
            
        except ImportError:
            print("Warning: openpyxl not installed. Cannot export to Excel.")
            print("Install with: pip install openpyxl")
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
    
    def run_complete_analysis(self, save_outputs=True):
        """Run complete analysis pipeline"""
        print("Starting Season 1 Complete Analysis...")
        print("="*60)
        
        # Load data
        if not self.load_episode_results():
            print("Failed to load episode data. Check directory structure.")
            return False
        
        # Create summary
        self.create_summary_dataframe()
        
        # Print statistics
        print("\nGenerating summary statistics...")
        self.print_summary_statistics()
        
        # Generate visualizations
        print("\nCreating performance overview plots...")
        self.plot_performance_overview(save_outputs)
        
        print("\nCreating detailed comparison plots...")
        self.plot_detailed_episode_comparison(save_outputs)
        
        # Generate report
        print("\nGenerating comprehensive report...")
        self.generate_report(save_outputs)
        
        # Export to Excel
        if save_outputs:
            print("\nExporting to Excel...")
            self.export_to_excel()
        
        print(f"\nAnalysis complete! All outputs saved to: {self.output_dir}")
        return True

# Utility functions
def quick_analysis(results_dir):
    """Quick analysis for immediate insights"""
    analyzer = S1EpisodeAnalyzer(results_dir)
    
    if analyzer.load_episode_results():
        summary = analyzer.create_summary_dataframe()
        
        print("\nQUICK ANALYSIS RESULTS")
        print("="*40)
        print(f"Episodes: {len(analyzer.episode_data)}")
        print(f"Avg F1-Score: {summary['Weighted_F1'].mean()*100:.2f}%")
        print(f"Best: Episode {summary.loc[summary['Weighted_F1'].idxmax(), 'Episode']} ({summary['Weighted_F1'].max()*100:.2f}%)")
        print(f"Worst: Episode {summary.loc[summary['Weighted_F1'].idxmin(), 'Episode']} ({summary['Weighted_F1'].min()*100:.2f}%)")
        print(f"Consistency: {summary['Weighted_F1'].std()*100:.2f}% std dev")
        
        return analyzer
    else:
        print("Failed to load data.")
        return None

def validate_directory_structure(results_dir):
    """Validate that the directory structure is correct"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Directory does not exist: {results_dir}")
        return False
    
    episode_folders = []
    for item in results_path.iterdir():
        if item.is_dir() and 'ep' in item.name.lower():
            episode_folders.append(item)
    
    print(f"Found {len(episode_folders)} episode folders:")
    
    valid_episodes = 0
    for folder in sorted(episode_folders):
        metrics_file = folder / "metrics_summary.txt"
        if metrics_file.exists():
            print(f"  {folder.name}: Valid")
            valid_episodes += 1
        else:
            print(f"  {folder.name}: Missing metrics_summary.txt")
    
    print(f"\nValid episodes: {valid_episodes}")
    return valid_episodes > 0

# Example usage
def main():
    """Main function demonstrating usage"""
    
    # Set your results directory path here
    results_directory = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_s1_overall"  # Adjust path as needed
    
    print("Season 1 Face Clustering Analysis Tool")
    print("="*50)
    
    # Validate directory structure
    print("Validating directory structure...")
    if not validate_directory_structure(results_directory):
        print("Please check your directory structure and try again.")
        return
    
    # Create analyzer and run complete analysis
    analyzer = S1EpisodeAnalyzer(results_directory)
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\nAnalysis completed successfully!")
        print("Check the 'season1_analysis' folder for all outputs.")
    else:
        print("Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    # You can run the main function or use individual components
    main()
    
    # Alternative: Quick analysis only
    # results_dir = r"result_s1_overall"
    # quick_analyzer = quick_analysis(results_dir)