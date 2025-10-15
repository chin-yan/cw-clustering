#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度診斷 PKL 檔案 - 找出所有可用資訊
"""

import pickle
import json
import os
from pathlib import Path

def deep_diagnose_pkl(pkl_path):
    """深度診斷 PKL 檔案,找出所有可用資訊"""
    
    print("="*70)
    print(f"🔍 深度診斷: {pkl_path}")
    print("="*70)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✅ 成功載入檔案")
        print(f"📦 資料型態: {type(data)}")
        print(f"📊 資料大小: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        # 分析結構
        if isinstance(data, dict):
            print(f"\n📋 字典結構分析:")
            print(f"   Key 總數: {len(data)}")
            print(f"   Key 類型: {type(list(data.keys())[0]) if data else 'empty'}")
            
            # 顯示前 5 個 keys
            print(f"\n   前 5 個 Keys:")
            for i, key in enumerate(list(data.keys())[:5]):
                print(f"     {i+1}. {key} (類型: {type(key)})")
            
            # 深度分析第一個 value
            if data:
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                
                print(f"\n   第一個 Value 深度分析:")
                print(f"     型態: {type(first_value)}")
                
                if isinstance(first_value, dict):
                    print(f"     字典 Keys: {list(first_value.keys())}")
                    print(f"\n     完整內容:")
                    for k, v in first_value.items():
                        print(f"       {k}: {v} (型態: {type(v)})")
                
                elif isinstance(first_value, list):
                    print(f"     列表長度: {len(first_value)}")
                    if first_value:
                        print(f"     第一個元素型態: {type(first_value[0])}")
                        print(f"     第一個元素內容: {first_value[0]}")
                        
                        if isinstance(first_value[0], dict):
                            print(f"\n     第一個元素的 Keys:")
                            for k, v in first_value[0].items():
                                print(f"       {k}: {v}")
                
                else:
                    print(f"     值: {first_value}")
            
            # 檢查特殊結構
            print(f"\n🔍 檢查常見結構:")
            special_keys = ['by_center', 'by_frame', 'retrieval_results', 
                          'frame_results', 'centers', 'character_id', 'match_idx']
            
            for key in special_keys:
                if key in data:
                    print(f"     ✓ 發現 '{key}'")
                    value = data[key]
                    if isinstance(value, (dict, list)):
                        print(f"       型態: {type(value)}, 大小: {len(value)}")
        
        # 嘗試找出可能的 character_id / center_id
        print(f"\n🎯 尋找角色識別資訊...")
        character_info = find_character_info(data)
        
        if character_info:
            print(f"   ✅ 找到可能的角色資訊:")
            for info in character_info[:5]:
                print(f"     - {info}")
        else:
            print(f"   ❌ 未找到明確的角色識別資訊")
        
        # 嘗試找出 frame 資訊
        print(f"\n🎬 尋找 Frame 資訊...")
        frame_info = find_frame_info(data)
        
        if frame_info:
            print(f"   ✅ 找到可能的 frame 資訊:")
            for info in frame_info[:5]:
                print(f"     - {info}")
        else:
            print(f"   ❌ 未找到 frame 資訊")
        
        # 保存詳細診斷結果
        save_diagnosis(data, pkl_path)
        
        return data
        
    except Exception as e:
        print(f"\n❌ 診斷失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_character_info(data, max_depth=5, current_depth=0):
    """遞迴尋找可能的角色識別資訊"""
    character_keys = ['character_id', 'match_idx', 'center_id', 'cluster_id', 
                     'person_id', 'face_id', 'label', 'class_id']
    
    found = []
    
    if current_depth >= max_depth:
        return found
    
    if isinstance(data, dict):
        for key, value in list(data.items())[:20]:  # 限制檢查數量
            # 直接匹配
            if any(ck in str(key).lower() for ck in character_keys):
                found.append(f"Key '{key}' -> {value}")
            
            # 如果 value 是字典，檢查裡面
            if isinstance(value, dict):
                for vk, vv in value.items():
                    if any(ck in str(vk).lower() for ck in character_keys):
                        found.append(f"Path {key}.{vk} -> {vv}")
                
                # 遞迴搜尋
                if current_depth < max_depth - 1:
                    found.extend(find_character_info(value, max_depth, current_depth + 1))
            
            # 如果 value 是列表，檢查第一個元素
            elif isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    for vk, vv in value[0].items():
                        if any(ck in str(vk).lower() for ck in character_keys):
                            found.append(f"Path {key}[0].{vk} -> {vv}")
    
    elif isinstance(data, list) and data:
        if isinstance(data[0], dict):
            for key, value in data[0].items():
                if any(ck in str(key).lower() for ck in character_keys):
                    found.append(f"List[0].{key} -> {value}")
    
    return found

def find_frame_info(data, max_depth=5, current_depth=0):
    """遞迴尋找 frame 資訊"""
    frame_keys = ['frame', 'frame_num', 'frame_idx', 'frame_number', 'timestamp']
    
    found = []
    
    if current_depth >= max_depth:
        return found
    
    if isinstance(data, dict):
        for key, value in list(data.items())[:20]:
            # 檢查 key 是否為數字 (可能是 frame_idx)
            try:
                frame_idx = int(key)
                if 0 <= frame_idx < 1000000:  # 合理的 frame 範圍
                    found.append(f"Key as frame_idx: {frame_idx}")
            except:
                pass
            
            # 直接匹配
            if any(fk in str(key).lower() for fk in frame_keys):
                found.append(f"Key '{key}' -> {value}")
            
            # 如果 value 是字典，檢查裡面
            if isinstance(value, dict):
                for vk, vv in value.items():
                    if any(fk in str(vk).lower() for fk in frame_keys):
                        found.append(f"Path {key}.{vk} -> {vv}")
            
            # 如果 value 是列表
            elif isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    for vk, vv in value[0].items():
                        if any(fk in str(vk).lower() for fk in frame_keys):
                            found.append(f"Path {key}[0].{vk} -> {vv}")
    
    return found

def save_diagnosis(data, pkl_path):
    """保存詳細診斷結果"""
    output_dir = os.path.dirname(pkl_path)
    base_name = os.path.basename(pkl_path).replace('.pkl', '')
    
    # 保存 JSON 樣本
    json_path = os.path.join(output_dir, f'{base_name}_full_diagnosis.json')
    
    try:
        def convert_to_serializable(obj, max_depth=3, current_depth=0):
            """轉換為 JSON 可序列化格式"""
            if current_depth >= max_depth:
                return str(type(obj))
            
            if isinstance(obj, dict):
                result = {}
                for k, v in list(obj.items())[:10]:  # 限制數量
                    result[str(k)] = convert_to_serializable(v, max_depth, current_depth + 1)
                if len(obj) > 10:
                    result['...'] = f'還有 {len(obj) - 10} 個項目'
                return result
            elif isinstance(obj, list):
                if len(obj) <= 5:
                    return [convert_to_serializable(item, max_depth, current_depth + 1) for item in obj]
                else:
                    return [convert_to_serializable(obj[0], max_depth, current_depth + 1), 
                           '...', 
                           f'共 {len(obj)} 個項目']
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        json_data = convert_to_serializable(data)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 詳細診斷已保存: {json_path}")
        
    except Exception as e:
        print(f"\n⚠️  無法保存 JSON: {e}")

def suggest_fix_strategy(pkl_path):
    """根據診斷結果建議修復策略"""
    data = deep_diagnose_pkl(pkl_path)
    
    if data is None:
        return
    
    print("\n" + "="*70)
    print("💡 修復建議")
    print("="*70)
    
    has_character_info = False
    has_frame_info = False
    
    # 檢查是否有角色資訊
    character_info = find_character_info(data)
    if character_info:
        has_character_info = True
        print("✓ 發現角色識別資訊")
    else:
        print("❌ 缺少角色識別資訊")
    
    # 檢查是否有 frame 資訊
    frame_info = find_frame_info(data)
    if frame_info:
        has_frame_info = True
        print("✓ 發現 frame 資訊")
    else:
        print("❌ 缺少 frame 資訊")
    
    print("\n建議:")
    
    if has_character_info and has_frame_info:
        print("1. ✅ 可以使用 fix_retrieval_results.py 修復")
        print("   python fix_retrieval_results.py", pkl_path)
    
    elif has_character_info and not has_frame_info:
        print("1. ⚠️  有角色資訊但缺少 frame 編號")
        print("   需要從檔案路徑提取 frame 編號")
        print("   檢查是否有 'path' 欄位包含檔名如: 'frame_XXXXXX_...'")
    
    elif not has_character_info and has_frame_info:
        print("1. ⚠️  有 frame 資訊但缺少角色識別")
        print("   需要從其他來源獲取角色識別結果")
        print("   選項 A: 檢查是否有 centers_data.pkl")
        print("   選項 B: 重新執行 enhanced_video_annotation.py")
    
    else:
        print("1. ❌ 此 PKL 檔案無法直接用於 Ground Truth 標註")
        print("   需要:")
        print("   - 角色識別結果 (character_id)")
        print("   - Frame 編號 (frame_idx)")
        print("\n   建議重新執行 enhanced_video_annotation.py")
        print("   或使用 enhanced_detection_results.pkl (如果有的話)")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        pkl_path = input("請輸入 PKL 檔案路徑: ").strip('"')
    
    suggest_fix_strategy(pkl_path)