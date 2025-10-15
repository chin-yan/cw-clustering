# -*- coding: utf-8 -*-
"""
修正版的 enhanced_face_retrieval.py
確保保存 speaking_ground_truth_tool.py 所需的所有資訊
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import re

def extract_frame_info(image_path):
    """從圖片路徑提取 frame 編號和 face 索引"""
    basename = os.path.basename(image_path)
    match = re.match(r'retrieval_frame_(\d+)_mainface_(\d+)\.jpg', basename)
    
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    
    return -1, -1

def save_retrieval_results_for_ground_truth(retrieval_results, frame_results, 
                                            facial_encodings, output_dir, fps=30.0):
    """
    保存適用於 speaking_ground_truth_tool.py 的結果格式
    
    預期格式:
    {
        frame_idx: {
            'character_id': int,
            'similarity': float,
            'bbox': tuple,
            'path': str  # 可選,用於除錯
        }
    }
    """
    print("\n💾 保存 Ground Truth 工具適用的格式...")
    
    # 創建 frame_idx -> result 的映射
    ground_truth_format = {}
    
    # 方法1: 從 frame_results 轉換
    if frame_results:
        print("   使用 frame_results 資料...")
        for frame_num, matches in frame_results.items():
            if not matches:
                continue
            
            # 使用最佳匹配 (第一個)
            best_match = matches[0]
            
            ground_truth_format[frame_num] = {
                'character_id': best_match.get('center_id', -1),
                'similarity': best_match.get('similarity', 0.0),
                'bbox': None,  # frame_results 中可能沒有 bbox
                'path': best_match.get('path', ''),
                'quality': best_match.get('quality', 0.0)
            }
    
    # 方法2: 從 retrieval_results 補充
    if retrieval_results:
        print("   補充 retrieval_results 資料...")
        for center_id, matches in retrieval_results.items():
            for match in matches:
                path = match.get('path', '')
                frame_num, _ = extract_frame_info(path)
                
                if frame_num <= 0:
                    continue
                
                # 如果這個 frame 還沒有記錄,或者這個匹配更好
                current_sim = match.get('original_similarity', 0.0)
                
                if frame_num not in ground_truth_format:
                    ground_truth_format[frame_num] = {
                        'character_id': center_id,
                        'similarity': current_sim,
                        'bbox': None,
                        'path': path,
                        'quality': match.get('quality', 0.0)
                    }
                elif current_sim > ground_truth_format[frame_num]['similarity']:
                    # 更新為更好的匹配
                    ground_truth_format[frame_num].update({
                        'character_id': center_id,
                        'similarity': current_sim,
                        'path': path
                    })
    
    # 保存結果
    output_path = os.path.join(output_dir, 'retrieval_results_ground_truth_format.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(ground_truth_format, f)
    
    print(f"✅ 已保存 {len(ground_truth_format)} 個 frame 的結果")
    print(f"   檔案: {output_path}")
    
    # 也保存 JSON 版本供檢查
    import json
    json_path = output_path.replace('.pkl', '.json')
    
    # 轉換為 JSON 可序列化格式
    json_data = {}
    for frame_idx, data in list(ground_truth_format.items())[:10]:  # 只保存前10個樣本
        json_data[str(frame_idx)] = {
            'character_id': int(data['character_id']),
            'similarity': float(data['similarity']),
            'bbox': data['bbox'],
            'path': data['path'],
            'quality': float(data['quality'])
        }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"   JSON 樣本: {json_path}")
    
    # 統計資訊
    character_counts = {}
    for data in ground_truth_format.values():
        char_id = data['character_id']
        character_counts[char_id] = character_counts.get(char_id, 0) + 1
    
    print(f"\n📊 角色出現統計:")
    for char_id in sorted(character_counts.keys()):
        if char_id >= 0:
            count = character_counts[char_id]
            print(f"   角色 ID {char_id}: {count} frames")
    
    if -1 in character_counts:
        print(f"   未識別: {character_counts[-1]} frames")
    
    return ground_truth_format


def fix_existing_retrieval_results(pkl_path, output_path=None):
    """
    修復現有的 retrieval_results.pkl 檔案
    將其轉換為 speaking_ground_truth_tool.py 可用的格式
    """
    print(f"🔧 修復檔案: {pkl_path}")
    
    # 載入原始檔案
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"   原始格式: {type(data)}")
    
    # 檢查格式
    if isinstance(data, dict):
        if 'by_center' in data or 'by_frame' in data:
            print("   偵測到新格式 (by_center/by_frame)")
            retrieval_results = data.get('by_center', {})
            frame_results = data.get('by_frame', {})
        else:
            print("   偵測到舊格式")
            retrieval_results = data
            frame_results = {}
    else:
        print("   ❌ 無法識別的格式")
        return False
    
    # 轉換格式
    ground_truth_format = {}
    
    # 從 by_center 轉換
    if retrieval_results:
        print(f"   處理 {len(retrieval_results)} 個角色的資料...")
        for center_id, matches in retrieval_results.items():
            for match in matches:
                path = match.get('path', '')
                frame_num = match.get('frame', -1)
                
                # 如果 frame 是 -1,嘗試從路徑提取
                if frame_num == -1:
                    frame_num, _ = extract_frame_info(path)
                
                if frame_num <= 0:
                    continue
                
                similarity = match.get('original_similarity', 
                                     match.get('similarity', 0.0))
                
                # 記錄最佳匹配
                if frame_num not in ground_truth_format:
                    ground_truth_format[frame_num] = {
                        'character_id': center_id,
                        'similarity': similarity,
                        'bbox': None,
                        'path': path
                    }
                elif similarity > ground_truth_format[frame_num]['similarity']:
                    ground_truth_format[frame_num].update({
                        'character_id': center_id,
                        'similarity': similarity,
                        'path': path
                    })
    
    # 從 by_frame 轉換 (如果有的話會更準確)
    if frame_results:
        print(f"   處理 {len(frame_results)} 個 frame 的資料...")
        for frame_num, matches in frame_results.items():
            if not matches:
                continue
            
            best_match = matches[0]
            ground_truth_format[frame_num] = {
                'character_id': best_match.get('center_id', -1),
                'similarity': best_match.get('similarity', 0.0),
                'bbox': None,
                'path': best_match.get('path', '')
            }
    
    # 保存修復後的檔案
    if output_path is None:
        output_path = pkl_path.replace('.pkl', '_fixed.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(ground_truth_format, f)
    
    print(f"✅ 已保存修復後的檔案: {output_path}")
    print(f"   包含 {len(ground_truth_format)} 個 frame 的資料")
    
    # 保存 JSON 樣本
    json_path = output_path.replace('.pkl', '_sample.json')
    import json
    
    sample_data = {}
    for frame_idx in sorted(ground_truth_format.keys())[:10]:
        data = ground_truth_format[frame_idx]
        sample_data[str(frame_idx)] = {
            'character_id': int(data['character_id']),
            'similarity': float(data['similarity']),
            'bbox': data['bbox'],
            'path': data['path']
        }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"   JSON 樣本: {json_path}")
    
    return True


# 使用範例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        fix_existing_retrieval_results(pkl_path, output_path)
    else:
        print("使用方法:")
        print("  python fix_retrieval_results.py <input.pkl> [output.pkl]")
        print("\n範例:")
        print("  python fix_retrieval_results.py enhanced_retrieval_results.pkl")