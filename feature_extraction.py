# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

def load_model(sess, model_dir):
    """
    Loading the InsightFace model (ArcFace)
    """
    print(f'Loading InsightFace model (ArcFace)...')
    
    # 初始化 FaceAnalysis，同時載入 detection 和 recognition 以避免報錯
    # providers 列表讓它優先嘗試 GPU，沒有的話會用 CPU
    app = FaceAnalysis(
        name='buffalo_l', 
        allowed_modules=['detection', 'recognition'], 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # === 關鍵的自動降級機制 ===
    try:
        # 嘗試使用 GPU (ctx_id=0)
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace initialized on GPU")
    except Exception as e:
        # 如果失敗（例如 CUDA 版本不符），自動切換回 CPU (ctx_id=-1)
        print(f"⚠️  GPU initialization failed, falling back to CPU. Error: {e}")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ InsightFace initialized on CPU")
    
    print("InsightFace model loaded successfully")
    return app

def compute_facial_encodings(sess, images_placeholder, embeddings, phase_train_placeholder, 
                             image_size, embedding_size, nrof_images, nrof_batches, 
                             emb_array, batch_size, paths, model_handler=None):
    """
    Calculate facial feature encoding using InsightFace
    """
    print("Calculating facial feature encoding with InsightFace...")
    
    # 取得識別模型
    rec_model = model_handler.models['recognition']
    
    facial_encodings = {}
    
    for i in tqdm(range(nrof_images)):
        path = paths[i]
        
        try:
            img = cv2.imread(path)
            if img is None:
                continue
                
            # InsightFace 識別模型標準輸入是 112x112
            input_face = cv2.resize(img, (112, 112))
            
            # 提取特徵 (512維)
            embedding = rec_model.get_feat(input_face)
            embedding = embedding.flatten()

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            emb_array[i, :] = embedding
            facial_encodings[path] = embedding
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
            
    print("Facial feature coding calculation completed")
    return facial_encodings