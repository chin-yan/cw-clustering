#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
診斷 retrieval results 檔案
"""

import pickle
import json

def diagnose_retrieval_file(file_path):
    """診斷 retrieval results 檔案"""
    
    print("="*70)
    print(f"🔍 診斷檔案: {file_path}")
    print("="*70)
    
    try:
        # 嘗試載入
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✅ 檔案載入成功")
        print(f"資料類型: {type(data)}")
        
        # 分析結構
        if isinstance(data, dict):
            print(f"\n📊 字典結構分析:")
            print(f"   Key 數量: {len(data)}")
            
            if len(data) > 0:
                # 顯示前 5 個 keys
                print(f"\n   前 5 個 Keys:")
                for i, key in enumerate(list(data.keys())[:5]):
                    print(f"     {i+1}. {key} (類型: {type(key)})")
                
                # 檢查第一個 value 的結構
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                
                print(f"\n   第一個 Value 結構:")
                print(f"     類型: {type(first_value)}")
                
                if isinstance(first_value, dict):
                    print(f"     Keys: {list(first_value.keys())}")
                    print(f"     內容範例:")
                    for k, v in list(first_value.items())[:3]:
                        print(f"       {k}: {v}")
                elif isinstance(first_value, list):
                    print(f"     長度: {len(first_value)}")
                    if len(first_value) > 0:
                        print(f"     第一個元素: {first_value[0]}")
                else:
                    print(f"     值: {first_value}")
                
                # 檢查是否有 'by_center' 或 'by_frame' 結構
                print(f"\n   檢查常見結構:")
                if 'by_center' in data:
                    print(f"     ✓ 發現 'by_center'")
                    print(f"       Centers 數量: {len(data['by_center'])}")
                
                if 'by_frame' in data:
                    print(f"     ✓ 發現 'by_frame'")
                    print(f"       Frames 數量: {len(data['by_frame'])}")
                
                # 嘗試找出可能的幀索引
                print(f"\n   尋找幀資料...")
                possible_frame_keys = []
                for key in list(data.keys())[:20]:
                    # 檢查 key 是否像幀索引
                    if isinstance(key, (int, str)):
                        try:
                            frame_idx = int(key)
                            possible_frame_keys.append((key, frame_idx))
                        except:
                            pass
                
                if possible_frame_keys:
                    print(f"     ✓ 可能的幀索引 (前 5 個):")
                    for original_key, frame_idx in possible_frame_keys[:5]:
                        print(f"       {original_key} → Frame {frame_idx}")
            else:
                print(f"\n❌ 字典是空的！")
        
        elif isinstance(data, list):
            print(f"\n📊 列表結構分析:")
            print(f"   元素數量: {len(data)}")
            if len(data) > 0:
                print(f"   第一個元素類型: {type(data[0])}")
                print(f"   第一個元素: {data[0]}")
        
        else:
            print(f"\n⚠️  未預期的資料類型: {type(data)}")
        
        # 儲存 JSON 版本以便檢視
        json_path = file_path.replace('.pkl', '_debug.json')
        try:
            # 嘗試轉換並儲存為 JSON
            def convert_to_json_serializable(obj):
                """轉換為 JSON 可序列化格式"""
                if isinstance(obj, dict):
                    return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_json_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            json_data = convert_to_json_serializable(data)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 已儲存 JSON 版本: {json_path}")
            print(f"   你可以用文字編輯器開啟檢視完整結構")
        
        except Exception as e:
            print(f"\n⚠️  無法儲存 JSON: {e}")
        
    except Exception as e:
        print(f"\n❌ 載入失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("請輸入 pkl 檔案路徑: ").strip('"')
    
    diagnose_retrieval_file(file_path)