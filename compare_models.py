import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# ================= 設定區 =================
# 資料夾根目錄
BASE_DIR = 'comparison'

# 方法名稱 (需與資料夾名稱一致)
METHODS = ['MTCNN+FaceNet', 'SCRFD+ArcFace']

# Epoch 對應表
EPOCH_MAP = {
    'result_ep1': 1,
    'result_ep2': 2,
    'result_ep3': 3,
    'result_ep4': 4,
    'result_ep6': 6,
    'result_ep7': 7,
    'result_ep8': 8,
    'result_ep9': 9,
    'result_ep10': 10
}

# 檔案名稱
METRICS_FILE = 'metrics_summary.txt'

# 正規表達式 (只抓取 Accuracy)
PATTERN_ACC = r"Overall Accuracy: ([\d\.]+)"
# =========================================

def parse_accuracy(file_path):
    """讀取檔案並抓取 Accuracy"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(PATTERN_ACC, content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def main():
    records = []
    print(f"Starting analysis in directory: {BASE_DIR}...")

    # 1. 遍歷資料夾讀取數據
    for method in METHODS:
        method_path = os.path.join(BASE_DIR, method)
        if not os.path.exists(method_path):
            print(f"Warning: Directory not found -> {method_path}")
            continue

        for ep_folder, ep_num in EPOCH_MAP.items():
            file_path = os.path.join(method_path, ep_folder, METRICS_FILE)
            
            if os.path.exists(file_path):
                acc = parse_accuracy(file_path)
                if acc is not None:
                    records.append({
                        'Method': method,
                        'Epoch': ep_num,
                        'Accuracy': acc
                    })
            else:
                pass # 忽略找不到的檔案

    if not records:
        print("No data found. Please check your directory structure.")
        return

    # 轉為 DataFrame 並排序
    df = pd.DataFrame(records)
    df = df.sort_values(by='Epoch')

    # 2. 繪製 Accuracy 折線圖 (含數值標示)
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    for method in METHODS:
        subset = df[df['Method'] == method]
        # 繪製折線
        plt.plot(subset['Epoch'], subset['Accuracy'], marker='o', label=method, linewidth=2)
        
        # --- 新增功能：在點上標示數據 ---
        for x, y in zip(subset['Epoch'], subset['Accuracy']):
            # xytext設定標籤偏移量，避免蓋住點
            plt.annotate(f'{y:.4f}', 
                         (x, y), 
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center', 
                         fontsize=9,
                         fontweight='bold')

    plt.title('Accuracy Comparison: MTCNN+FaceNet vs SCRFD+ArcFace', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Overall Accuracy')
    plt.xticks(sorted(EPOCH_MAP.values())) # 確保 X 軸只有這些 Epoch 點
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 儲存圖片
    plot_filename = 'accuracy_comparison_plot.png'
    plt.savefig(plot_filename)
    print(f"\nPlot saved as: {plot_filename}")
    # plt.show() # 如果在支援視窗的環境執行，可以取消註解

    # 3. 計算進步幅度
    print("\n" + "="*60)
    print(f" Improvement Analysis (Accuracy): {METHODS[1]} vs {METHODS[0]}")
    print("="*60)

    try:
        # 整理表格計算差異
        df_pivot = df.pivot(index='Epoch', columns='Method', values='Accuracy')
        
        target = METHODS[1] # SCRFD+ArcFace
        base = METHODS[0]   # MTCNN+FaceNet
        
        # 計算差值
        df_pivot['Improvement'] = df_pivot[target] - df_pivot[base]
        df_pivot['Improvement (%)'] = (df_pivot['Improvement'] / df_pivot[base]) * 100

        # 整理輸出
        print(df_pivot[[base, target, 'Improvement', 'Improvement (%)']].round(4).to_string())

        # 存成 CSV
        csv_filename = 'accuracy_improvement.csv'
        df_pivot.reset_index().to_csv(csv_filename, index=False)
        print(f"\nDetailed data saved to: {csv_filename}")

    except KeyError as e:
        print(f"Error calculating improvement: {e} (Check if both methods have data)")

if __name__ == "__main__":
    main()