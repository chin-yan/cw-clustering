import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 設置中文字體支援
plt.rcParams['font.sans-serif'] = ['Microsoft YheiUI', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ConfusionMatrixAnalyzer:
    def __init__(self, excel_file_path):
        """
        初始化confusion matrix分析器
        
        Parameters:
        excel_file_path (str): Excel檔案路徑
        """
        self.excel_file_path = excel_file_path
        
        self.output_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering"
            
        self.confusion_matrix = None
        self.labels = None
        
        # 創建輸出目錄（如果不存在）
        self.create_output_directory()
        
        # 創建時間戳記用於檔案命名
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"檔案時間戳記: {self.timestamp}")
        
        self.load_data()
    
    def create_output_directory(self):
        """創建輸出目錄"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"創建輸出目錄: {self.output_dir}")
            else:
                print(f"使用現有輸出目錄: {self.output_dir}")
        except Exception as e:
            print(f"創建目錄時發生錯誤: {e}")
            # 如果無法創建指定目錄，使用當前目錄
            self.output_dir = "."
            print("將使用當前目錄作為輸出目錄")
    
    def get_output_path(self, filename):
        """獲取完整的輸出檔案路徑"""
        full_path = os.path.join(self.output_dir, f"{self.timestamp}_{filename}")
        print(f"將保存檔案至: {full_path}")
        return full_path
    
    def load_data(self):
        """從Excel檔案載入confusion matrix"""
        try:
            # 讀取Excel檔案
            df = pd.read_excel(self.excel_file_path, header=0, index_col=0)
            
            # 提取confusion matrix和標籤
            self.confusion_matrix = df.values
            self.labels = df.columns.tolist()
            
            print(f"成功載入confusion matrix: {self.confusion_matrix.shape}")
            print(f"標籤數量: {len(self.labels)}")
            print(f"標籤: {self.labels[:10]}...")  # 顯示前10個標籤
            
        except Exception as e:
            print(f"載入資料時發生錯誤: {e}")
    
    def calculate_metrics(self):
        """計算各種評估指標"""
        # 計算基本指標
        total_samples = np.sum(self.confusion_matrix)
        correct_predictions = np.trace(self.confusion_matrix)  # 對角線元素總和
        overall_accuracy = correct_predictions / total_samples
        
        # 計算每個類別的指標
        n_classes = len(self.labels)
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_score = np.zeros(n_classes)
        support = np.zeros(n_classes)
        
        for i in range(n_classes):
            # True Positive
            tp = self.confusion_matrix[i, i]
            
            # False Positive (預測為i但實際不是i的總數)
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            
            # False Negative (實際為i但預測不是i的總數)
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            # Support (實際為類別i的樣本數)
            support[i] = np.sum(self.confusion_matrix[i, :])
            
            # 計算precision, recall, f1
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
        
        # 計算macro和weighted平均值
        macro_precision = np.mean(precision[support > 0])  # 只計算有樣本的類別
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
        """打印評估指標摘要並自動保存到檔案"""
        metrics = self.calculate_metrics()
        
        summary_text = []
        summary_text.append("="*60)
        summary_text.append("CONFUSION MATRIX 評估報告")
        summary_text.append("="*60)
        summary_text.append(f"總樣本數: {metrics['total_samples']:,}")
        summary_text.append(f"正確預測數: {metrics['correct_predictions']:,}")
        summary_text.append(f"整體準確率 (Overall Accuracy): {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        summary_text.append("")
        
        summary_text.append("平均指標:")
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
        
        # 找出表現最好和最差的類別
        non_zero_mask = metrics['support'] > 0
        if np.any(non_zero_mask):
            valid_f1 = metrics['f1_score'][non_zero_mask]
            valid_labels = [self.labels[i] for i in range(len(self.labels)) if non_zero_mask[i]]
            
            if len(valid_f1) > 0:
                best_idx = np.argmax(valid_f1)
                worst_idx = np.argmin(valid_f1)
                
                summary_text.append(f"表現最好的類別: {valid_labels[best_idx]} (F1: {valid_f1[best_idx]:.4f})")
                summary_text.append(f"表現最差的類別: {valid_labels[worst_idx]} (F1: {valid_f1[worst_idx]:.4f})")
        
        # 打印到控制台
        for line in summary_text:
            print(line)
        
        # 自動保存到檔案
        summary_file = self.get_output_path("metrics_summary.txt")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_text))
            print(f"\n評估指標摘要已保存至: {summary_file}")
        except Exception as e:
            print(f"保存評估指標摘要時發生錯誤: {e}")
    
    def plot_confusion_matrix_heatmap(self, figsize=(15, 12)):
        """繪製confusion matrix熱力圖"""
        plt.figure(figsize=figsize)
        
        # 使用log scale來更好地顯示差異
        matrix_log = np.log10(self.confusion_matrix + 1)  # +1避免log(0)
        
        sns.heatmap(matrix_log, 
                   xticklabels=self.labels, 
                   yticklabels=self.labels,
                   annot=False,  # 對於大矩陣不顯示數值
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
            print(f"Confusion Matrix 熱力圖已保存至: {save_path}")
        except Exception as e:
            print(f"保存Confusion Matrix熱力圖時發生錯誤: {e}")
        
        plt.show()
    
    def plot_class_performance(self, figsize=(15, 10)):
        """繪製各類別的performance指標"""
        metrics = self.calculate_metrics()
        
        # 只顯示有樣本的類別
        non_zero_mask = metrics['support'] > 0
        valid_labels = [self.labels[i] for i in range(len(self.labels)) if non_zero_mask[i]]
        valid_precision = metrics['precision'][non_zero_mask]
        valid_recall = metrics['recall'][non_zero_mask]
        valid_f1 = metrics['f1_score'][non_zero_mask]
        valid_support = metrics['support'][non_zero_mask]
        
        # 創建subplot
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
            print(f"類別性能圖已保存至: {save_path}")
        except Exception as e:
            print(f"保存類別性能圖時發生錯誤: {e}")
        
        plt.show()
    
    def plot_error_analysis(self, figsize=(12, 8)):
        """分析分類錯誤模式"""
        # 計算每個類別的錯誤分布
        errors_per_class = []
        true_labels = []
        
        for i in range(len(self.labels)):
            total_true = np.sum(self.confusion_matrix[i, :])
            correct = self.confusion_matrix[i, i]
            errors = total_true - correct
            
            if total_true > 0:  # 只考慮有樣本的類別
                errors_per_class.append(errors)
                true_labels.append(self.labels[i])
        
        # 繪製錯誤分析圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. 每個類別的錯誤數量
        ax1.bar(range(len(true_labels)), errors_per_class, alpha=0.7, color='salmon')
        ax1.set_title('Classification Errors per Class', fontweight='bold')
        ax1.set_ylabel('Number of Errors')
        ax1.set_xticks(range(len(true_labels)))
        ax1.set_xticklabels(true_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 錯誤率分布
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
            print(f"錯誤分析圖已保存至: {save_path}")
        except Exception as e:
            print(f"保存錯誤分析圖時發生錯誤: {e}")
        
        plt.show()
    
    def generate_detailed_report(self):
        """生成詳細的分類報告並自動保存"""
        metrics = self.calculate_metrics()
        
        # 創建詳細報告的DataFrame
        report_data = []
        
        for i, label in enumerate(self.labels):
            if metrics['support'][i] > 0:  # 只包含有樣本的類別
                report_data.append({
                    'Class': label,
                    'Precision': f"{metrics['precision'][i]:.4f}",
                    'Recall': f"{metrics['recall'][i]:.4f}",
                    'F1-Score': f"{metrics['f1_score'][i]:.4f}",
                    'Support': metrics['support'][i]
                })
        
        # 添加平均值
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
            print(f"詳細分類報告已保存至: {csv_path}")
        except Exception as e:
            print(f"保存詳細分類報告時發生錯誤: {e}")
        
        return report_df

# 使用範例
if __name__ == "__main__":
    analyzer = ConfusionMatrixAnalyzer(r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\confusion_matrix_s1ep1_2.0.xlsx")
    
    print("="*80)
    print("開始進行 Confusion Matrix 分析...")
    print("="*80)
    
    # 打印評估指標摘要
    analyzer.print_metrics_summary()
    
    print("\n" + "="*60)
    print("生成可視化圖表...")
    print("="*60)
    
    # 1. Confusion Matrix 熱力圖
    print("\n1. 生成 Confusion Matrix 熱力圖...")
    analyzer.plot_confusion_matrix_heatmap(figsize=(15, 12))
    
    # 2. 各類別性能圖
    print("\n2. 生成各類別性能圖...")
    analyzer.plot_class_performance(figsize=(15, 10))
    
    # 3. 錯誤分析圖
    print("\n3. 生成錯誤分析圖...")
    analyzer.plot_error_analysis(figsize=(12, 8))
    
    # 4. 生成詳細報告並保存到CSV
    print("\n4. 生成詳細分類報告...")
    detailed_report = analyzer.generate_detailed_report()
    print("\n詳細分類報告:")
    print(detailed_report.to_string(index=False))
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"所有檔案已保存至目錄: {analyzer.output_dir}")
    print("生成的檔案包括:")
    print(f"- {analyzer.timestamp}_metrics_summary.txt (評估指標摘要)")
    print(f"- {analyzer.timestamp}_confusion_matrix_heatmap.png (混淆矩陣熱力圖)")
    print(f"- {analyzer.timestamp}_class_performance.png (各類別性能圖)")
    print(f"- {analyzer.timestamp}_error_analysis.png (錯誤分析圖)")
    print(f"- {analyzer.timestamp}_detailed_classification_report.csv (詳細分類報告)")
    print("="*80)