#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
評估說話時刻的角色識別準確率
"""

import json
import argparse
from collections import defaultdict


def evaluate_speaking_accuracy(ground_truth_file):
    """評估說話時刻準確率"""
    
    print("="*70)
    print("📊 說話時刻準確率評估")
    print("="*70)
    
    # 載入 Ground Truth
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    
    # 統計
    total = 0
    correct = 0
    wrong = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for key, annotation in gt.items():
        # 跳過 unknown 和 skipped
        if annotation['status'] in ['unknown', 'skipped']:
            continue
        
        total += 1
        gt_id = annotation['ground_truth_id']
        pred_id = annotation['system_predicted_id']
        
        if gt_id == pred_id:
            correct += 1
        else:
            wrong += 1
        
        # 混淆矩陣
        confusion_matrix[gt_id][pred_id] += 1
    
    # 計算準確率
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n✅ 評估結果:")
    print(f"   總說話時刻: {total}")
    print(f"   正確識別: {correct}")
    print(f"   錯誤識別: {wrong}")
    print(f"   準確率: {accuracy*100:.2f}%")
    
    # 每個角色的準確率
    print(f"\n📊 各角色準確率:")
    for gt_id in sorted(confusion_matrix.keys()):
        char_total = sum(confusion_matrix[gt_id].values())
        char_correct = confusion_matrix[gt_id][gt_id]
        char_accuracy = char_correct / char_total if char_total > 0 else 0
        
        print(f"   角色 ID {gt_id}: {char_correct}/{char_total} = {char_accuracy*100:.1f}%")
    
    # 混淆矩陣
    print(f"\n📋 混淆矩陣:")
    print(f"   (行=真實, 列=預測)")
    
    all_ids = sorted(set(list(confusion_matrix.keys()) + 
                        [pred for preds in confusion_matrix.values() for pred in preds.keys()]))
    
    # 表頭
    print("     ", end="")
    for pred_id in all_ids:
        print(f"P{pred_id:2d} ", end="")
    print()
    
    # 內容
    for gt_id in all_ids:
        print(f"  T{gt_id:2d} ", end="")
        for pred_id in all_ids:
            count = confusion_matrix[gt_id][pred_id]
            print(f"{count:3d} ", end="")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='評估說話時刻準確率')
    parser.add_argument('--ground-truth', required=True, 
                       help='Ground Truth 檔案路徑')
    
    args = parser.parse_args()
    
    evaluate_speaking_accuracy(args.ground_truth)