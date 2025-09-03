# 影片人臉聚類系統

這個系統可以從影片中擷取人臉，使用FaceNet計算特徵向量，然後通過Chinese Whispers算法進行聚類，以找出影片中相同人物的不同人臉圖像。

## 功能概述

1. **人臉擷取**：使用MTCNN從影片中擷取人臉
2. **特徵提取**：使用FaceNet將人臉轉換為特徵向量
3. **人臉聚類**：使用Chinese Whispers算法進行聚類
4. **中心計算**：計算每個聚類的中心作為人臉庫
5. **結果可視化**：創建各種可視化來展示聚類結果

## 系統需求

- Python 3.7+
- TensorFlow 1.x (由於使用的是FaceNet原始模型，需要使用TensorFlow 1.x)
- OpenCV
- NetworkX
- Matplotlib
- NumPy
- tqdm

## 安裝指南

1. 克隆存儲庫：
   ```bash
   git clone https://github.com/yourusername/video-face-clustering.git
   cd video-face-clustering
   ```

2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

3. 下載FaceNet預訓練模型：
   ```bash
   # 可以從GitHub的FaceNet存儲庫下載
   # https://github.com/davidsandberg/facenet
   ```

## 使用方法

### 基本用法

```bash
python main.py --input_video 影片路徑.mp4 --output_dir 輸出目錄 --model_dir FaceNet模型目錄
```

### 完整參數

- `--input_video`：輸入影片路徑
- `--output_dir`：輸出目錄
- `--model_dir`：FaceNet模型目錄
- `--batch_size`：批量大小（默認100）
- `--face_size`：人臉圖像大小（默認160）
- `--cluster_threshold`：聚類閾值（默認0.7）
- `--frames_interval`：擷取幀的間隔（默認30）
- `--visualize`：是否可視化結果（默認否）

## 輸出目錄結構

處理完成後，輸出目錄將包含以下內容：

```
輸出目錄/
├── faces/                # 從影片中擷取的所有人臉
├── clusters/             # 聚類結果
│   ├── cluster_0/        # 聚類0的人臉
│   ├── cluster_1/        # 聚類1的人臉
│   └── ...
├── centers/              # 聚類中心信息
│   └── centers_data.pkl  # 中心數據（包括編碼）
└── visualization/        # 可視化結果
    ├── cluster_sizes.png           # 聚類大小分佈圖
    ├── cluster_0_network.png       # 聚類0的網絡圖
    ├── cluster_0_thumbnails.png    # 聚類0的縮略圖集
    ├── ...
    └── cluster_centers.png         # 所有聚類中心
```

## 模組說明

- `main.py`：主程序，整合所有步驟
- `face_detection.py`：人臉檢測模組
- `feature_extraction.py`：特徵提取模組
- `clustering.py`：聚類模組
- `visualization.py`：可視化模組

## 範例

```bash
python main.py --input_video 電影.mp4 --output_dir ./results --model_dir ./models/facenet --visualize
```

## 技術細節

- **MTCNN**：多任務級聯卷積網絡，用於人臉檢測和對齊
- **FaceNet**：用於將人臉轉換為特徵向量（嵌入）
- **Chinese Whispers**：一種基於圖的聚類算法，特別適用於人臉聚類
- **聚類中心計算**：可以使用平均值或最小平均距離來計算中心

## 故障排除

- 如果出現內存錯誤，可以減少批量大小（`--batch_size`）
- 如果聚類結果不理想，可以調整聚類閾值（`--cluster_threshold`）
- 如果處理速度太慢，可以增加幀間隔（`--frames_interval`）

## 後續改進方向

1. 添加人臉識別功能，將聚類結果與已知人物數據庫匹配
2. 改進聚類算法，提高精度
3. 優化處理速度，支持實時處理
4. 添加GUI界面
5. 支持多個影片的批量處理
