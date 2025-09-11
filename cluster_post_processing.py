# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from collections import Counter

def merge_similar_large_clusters(clusters, facial_encodings, min_cluster_size=10, 
                                similarity_threshold=0.55, max_cross_sample_check=20):
    """
    Phase 1: 檢查並合併相似的大clusters
    
    Args:
        clusters: 原始clusters
        facial_encodings: 人臉編碼字典
        min_cluster_size: 被視為"大cluster"的最小size
        similarity_threshold: 合併閾值（大cluster需要更高相似度）
        max_cross_sample_check: 每個cluster最多取樣本數進行比對
        
    Returns:
        合併後的clusters和merge actions
    """
    print(f"🔍 Phase 1: 分析大clusters（size >= {min_cluster_size}）...")
    
    #進度顯示改進
    from tqdm import tqdm
    
    large_clusters = [(i, cluster) for i, cluster in enumerate(clusters) if len(cluster) >= min_cluster_size]
    print(f"   找到 {len(large_clusters)} 個大clusters")
    
    if len(large_clusters) < 2:
        print("   大clusters數量不足，跳過Phase 1")
        return clusters, []
    
    # 使用tqdm顯示比較進度
    total_comparisons = len(large_clusters) * (len(large_clusters) - 1) // 2
    merge_candidates = []
    
    pbar = tqdm(total=total_comparisons, desc="比較大clusters")
    for i in range(len(large_clusters)):
        for j in range(i+1, len(large_clusters)):
            # 比較邏輯...
            pbar.update(1)
    pbar.close()

    # 識別大clusters
    large_clusters = []
    small_clusters = []
    
    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_cluster_size:
            large_clusters.append((i, cluster))
        else:
            small_clusters.append((i, cluster))
    
    print(f"   找到 {len(large_clusters)} 個大clusters，{len(small_clusters)} 個小clusters")
    
    if len(large_clusters) < 2:
        print("   大clusters數量不足，跳過Phase 1")
        return clusters, []
    
    # 計算大clusters之間的相似度
    merge_candidates = []
    
    for i in range(len(large_clusters)):
        for j in range(i+1, len(large_clusters)):
            idx_i, cluster_i = large_clusters[i]
            idx_j, cluster_j = large_clusters[j]
            
            print(f"   比較 Cluster {idx_i} ({len(cluster_i)} faces) vs Cluster {idx_j} ({len(cluster_j)} faces)")
            
            # 計算cross-cluster相似度
            similarities = calculate_cross_cluster_similarity(
                cluster_i, cluster_j, facial_encodings, max_cross_sample_check
            )
            
            # 多重判斷條件
            centroid_sim = similarities['centroid']
            max_pairwise = similarities['max_pairwise']
            avg_pairwise = similarities['avg_pairwise']
            high_sim_ratio = similarities['high_similarity_ratio']  # 高相似度pairs的比例
            
            print(f"      質心相似度: {centroid_sim:.3f}")
            print(f"      最高配對相似度: {max_pairwise:.3f}")
            print(f"      平均配對相似度: {avg_pairwise:.3f}")
            print(f"      高相似度比例: {high_sim_ratio:.3f}")
            
            # 決定是否合併（使用多重條件）
            should_merge = (
                centroid_sim > similarity_threshold and
                (max_pairwise > similarity_threshold + 0.1 or  # 有非常相似的faces
                 (avg_pairwise > similarity_threshold - 0.05 and high_sim_ratio > 0.3))  # 整體相似且有足夠高相似度pairs
            )
            
            if should_merge:
                merge_candidates.append({
                    'cluster_i': idx_i,
                    'cluster_j': idx_j,
                    'centroid_sim': centroid_sim,
                    'avg_pairwise': avg_pairwise,
                    'confidence': (centroid_sim + avg_pairwise + high_sim_ratio) / 3
                })
                print(f"      ✅ 標記為合併候選")
            else:
                print(f"      ❌ 不符合合併條件")
    
    # 執行合併（按confidence排序）
    merge_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 使用Union-Find來處理多路合併
    cluster_groups = {}
    for i, cluster in enumerate(clusters):
        cluster_groups[i] = i
    
    def find_root(x):
        if cluster_groups[x] != x:
            cluster_groups[x] = find_root(cluster_groups[x])
        return cluster_groups[x]
    
    def union(x, y):
        root_x, root_y = find_root(x), find_root(y)
        if root_x != root_y:
            cluster_groups[root_y] = root_x
    
    merge_actions = []
    
    for candidate in merge_candidates:
        i, j = candidate['cluster_i'], candidate['cluster_j']
        
        # 檢查是否已經在同一組
        if find_root(i) != find_root(j):
            union(i, j)
            merge_actions.append({
                'type': 'large_cluster_merge',
                'cluster_i': i,
                'cluster_j': j,
                'confidence': candidate['confidence'],
                'centroid_sim': candidate['centroid_sim']
            })
            print(f"   ✅ 合併 Cluster {i} 和 Cluster {j} (confidence: {candidate['confidence']:.3f})")
    
    # 重建clusters
    group_to_cluster = {}
    for i, cluster in enumerate(clusters):
        root = find_root(i)
        if root not in group_to_cluster:
            group_to_cluster[root] = []
        group_to_cluster[root].extend(cluster)
    
    new_clusters = list(group_to_cluster.values())
    
    print(f"✅ Phase 1 完成: {len(clusters)} → {len(new_clusters)} clusters")
    return new_clusters, merge_actions


def calculate_cross_cluster_similarity(cluster1, cluster2, facial_encodings, max_samples=20):
    """
    計算兩個clusters之間的詳細相似度指標
    """
    # 採樣以避免計算量過大
    sample1 = cluster1[:max_samples] if len(cluster1) <= max_samples else \
              np.random.choice(cluster1, max_samples, replace=False).tolist()
    sample2 = cluster2[:max_samples] if len(cluster2) <= max_samples else \
              np.random.choice(cluster2, max_samples, replace=False).tolist()
    
    # 計算質心相似度
    center1 = calculate_cluster_center(sample1, facial_encodings)
    center2 = calculate_cluster_center(sample2, facial_encodings)
    centroid_sim = np.dot(center1, center2) if center1 is not None and center2 is not None else 0
    
    # 計算pairwise相似度
    similarities = []
    for path1 in sample1:
        for path2 in sample2:
            sim = np.dot(facial_encodings[path1], facial_encodings[path2])
            similarities.append(sim)
    
    if not similarities:
        return {'centroid': 0, 'max_pairwise': 0, 'avg_pairwise': 0, 'high_similarity_ratio': 0}
    
    similarities = np.array(similarities)
    
    return {
        'centroid': centroid_sim,
        'max_pairwise': np.max(similarities),
        'avg_pairwise': np.mean(similarities),
        'high_similarity_ratio': np.sum(similarities > 0.7) / len(similarities)  # 高相似度比例
    }


def merge_small_clusters_intelligently(clusters, facial_encodings, small_cluster_threshold=50,
                                     merge_threshold=0.45, safety_checks=True):
    """
    Phase 2: 智能合併小clusters，帶有安全檢查
    
    Args:
        clusters: Phase 1處理後的clusters
        facial_encodings: 人臉編碼字典
        small_cluster_threshold: 小cluster的閾值
        merge_threshold: 合併閾值
        safety_checks: 是否啟用安全檢查
        
    Returns:
        最終clusters和merge actions
    """
    print(f"🔍 Phase 2: 處理小clusters（size < {small_cluster_threshold}）...")
    
    # 識別大小clusters
    large_clusters = []
    small_clusters = []
    
    for i, cluster in enumerate(clusters):
        if len(cluster) >= small_cluster_threshold:
            large_clusters.append((i, cluster))
        else:
            small_clusters.append((i, cluster))
    
    print(f"   大clusters: {len(large_clusters)}，小clusters: {len(small_clusters)}")
    
    if not small_clusters:
        print("   沒有小clusters需要處理")
        return clusters, []
    
    # 為每個小cluster尋找最佳合併目標
    merge_proposals = []
    
    for small_idx, small_cluster in small_clusters:
        best_match = None
        best_similarity = 0
        
        print(f"   分析小cluster {small_idx} ({len(small_cluster)} faces)...")
        
        # 與所有大clusters比較
        for large_idx, large_cluster in large_clusters:
            similarities = calculate_inter_cluster_similarity(
                small_cluster, large_cluster, facial_encodings
            )
            
            combined_score = (
                similarities['centroid'] * 0.4 +
                similarities['max_pairwise'] * 0.3 +
                similarities['avg_pairwise'] * 0.3
            )
            
            if combined_score > best_similarity:
                best_similarity = combined_score
                best_match = large_idx
                
        if best_match is not None and best_similarity > merge_threshold:
            # 安全檢查
            if safety_checks:
                is_safe = perform_safety_checks(
                    small_cluster, clusters[best_match], facial_encodings, merge_threshold
                )
                if not is_safe:
                    print(f"      ❌ 安全檢查失敗，跳過合併")
                    continue
            
            merge_proposals.append({
                'small_cluster': small_idx,
                'target_cluster': best_match,
                'similarity': best_similarity,
                'small_size': len(small_cluster)
            })
            print(f"      ✅ 建議合併到 Cluster {best_match} (相似度: {best_similarity:.3f})")
        else:
            print(f"      ❌ 沒找到合適的合併目標 (最高相似度: {best_similarity:.3f})")
    
    # 執行合併
    new_clusters = [cluster.copy() for cluster in clusters]
    merge_actions = []
    
    # 按similarity排序，優先處理高相似度的合併
    merge_proposals.sort(key=lambda x: x['similarity'], reverse=True)
    
    merged_indices = set()
    
    for proposal in merge_proposals:
        small_idx = proposal['small_cluster']
        target_idx = proposal['target_cluster']
        
        if small_idx not in merged_indices:
            # 執行合併
            new_clusters[target_idx].extend(new_clusters[small_idx])
            new_clusters[small_idx] = []  # 清空被合併的cluster
            merged_indices.add(small_idx)
            
            merge_actions.append({
                'type': 'small_cluster_merge',
                'source': small_idx,
                'target': target_idx,
                'faces_added': proposal['small_size'],
                'similarity': proposal['similarity']
            })
            print(f"   ✅ 已合併 Cluster {small_idx} → Cluster {target_idx}")
    
    # 移除空clusters
    final_clusters = [cluster for cluster in new_clusters if len(cluster) > 0]
    
    print(f"✅ Phase 2 完成: 合併了 {len(merge_actions)} 個小clusters")
    print(f"   最終clusters數量: {len(final_clusters)}")
    
    return final_clusters, merge_actions


def perform_safety_checks(small_cluster, target_cluster, facial_encodings, base_threshold):
    """
    執行安全檢查以防止錯誤合併
    
    Args:
        small_cluster: 要合併的小cluster
        target_cluster: 目標cluster
        facial_encodings: 人臉編碼字典
        base_threshold: 基礎閾值
        
    Returns:
        bool: True表示安全，False表示不安全
    """
    # 安全檢查1: 避免極端尺寸差異（防止noise被合併到主要cluster）
    size_ratio = len(small_cluster) / len(target_cluster)
    if size_ratio < 0.02 and len(small_cluster) < 5:  # 太小且比例極低
        print(f"        安全檢查1失敗: cluster太小且比例過低 ({len(small_cluster)}/{len(target_cluster)})")
        return False
    
    # 安全檢查2: 檢查小cluster內部一致性
    small_internal_similarity = calculate_internal_cluster_consistency(small_cluster, facial_encodings)
    if small_internal_similarity < base_threshold - 0.1:  # 小cluster內部都不一致，可能是noise
        print(f"        安全檢查2失敗: 小cluster內部一致性太低 ({small_internal_similarity:.3f})")
        return False
    
    # 安全檢查3: 檢查合併後不會顯著降低目標cluster的一致性
    target_internal = calculate_internal_cluster_consistency(target_cluster[:50], facial_encodings)  # 採樣50個
    
    # 模擬合併後的一致性
    combined_sample = target_cluster[:25] + small_cluster[:25]  # 各取25個樣本
    combined_internal = calculate_internal_cluster_consistency(combined_sample, facial_encodings)
    
    consistency_drop = target_internal - combined_internal
    if consistency_drop > 0.15:  # 一致性下降太多
        print(f"        安全檢查3失敗: 合併會顯著降低目標cluster一致性 (下降 {consistency_drop:.3f})")
        return False
    
    print(f"        ✅ 通過所有安全檢查")
    return True


def calculate_internal_cluster_consistency(cluster, facial_encodings, max_samples=20):
    """
    計算cluster內部的一致性（平均pairwise相似度）
    """
    if len(cluster) < 2:
        return 1.0
    
    # 採樣以避免計算量過大
    sample = cluster[:max_samples] if len(cluster) <= max_samples else \
             np.random.choice(cluster, max_samples, replace=False).tolist()
    
    similarities = []
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            sim = np.dot(facial_encodings[sample[i]], facial_encodings[sample[j]])
            similarities.append(sim)
    
    return np.mean(similarities) if similarities else 0.0

def calculate_cluster_center(cluster, facial_encodings):
    """
    Calculate the centroid of a cluster
    
    Args:
        cluster: List of face paths in the cluster
        facial_encodings: Dictionary of facial encodings
        
    Returns:
        Normalized centroid vector
    """
    if not cluster:
        return None
    
    # Get all encodings for the cluster
    cluster_encodings = [facial_encodings[path] for path in cluster]
    
    # Calculate centroid
    centroid = np.mean(cluster_encodings, axis=0)
    
    # Normalize
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    
    return centroid

def calculate_inter_cluster_similarity(cluster1, cluster2, facial_encodings):
    """
    Calculate similarity between two clusters using multiple methods
    
    Args:
        cluster1, cluster2: Lists of face paths
        facial_encodings: Dictionary of facial encodings
        
    Returns:
        Dictionary with different similarity measures
    """
    # Method 1: Centroid similarity
    center1 = calculate_cluster_center(cluster1, facial_encodings)
    center2 = calculate_cluster_center(cluster2, facial_encodings)
    
    if center1 is None or center2 is None:
        return {'centroid': 0, 'max_pairwise': 0, 'avg_pairwise': 0}
    
    centroid_sim = np.dot(center1, center2)
    
    # Method 2: Maximum pairwise similarity
    max_sim = 0
    total_sim = 0
    count = 0
    
    sample_size = min(10, len(cluster1), len(cluster2))  # Sample to avoid too much computation
    
    for i, path1 in enumerate(cluster1[:sample_size]):
        for j, path2 in enumerate(cluster2[:sample_size]):
            sim = np.dot(facial_encodings[path1], facial_encodings[path2])
            max_sim = max(max_sim, sim)
            total_sim += sim
            count += 1
    
    avg_sim = total_sim / count if count > 0 else 0
    
    return {
        'centroid': centroid_sim,
        'max_pairwise': max_sim,
        'avg_pairwise': avg_sim
    }

def identify_oversplit_clusters(clusters, facial_encodings, target_large_clusters=None):
    """
    Identify clusters that might be oversplit from large clusters
    
    Args:
        clusters: List of clusters
        facial_encodings: Dictionary of facial encodings
        target_large_clusters: List of cluster indices to focus on (e.g., [2] for cluster 2)
        
    Returns:
        Dictionary of potential merges
    """
    print("🔍 Identifying potentially oversplit clusters...")
    
    merge_candidates = {}
    
    # Identify large clusters (>50 faces) as potential "parent" clusters
    large_clusters = []
    for i, cluster in enumerate(clusters):
        if len(cluster) > 50:
            large_clusters.append(i)
            print(f"   Large cluster {i}: {len(cluster)} faces")
    
    # If specific target clusters provided, use those
    if target_large_clusters:
        large_clusters = [i for i in target_large_clusters if i < len(clusters)]
    
    # For each large cluster, find similar small clusters
    for large_idx in large_clusters:
        large_cluster = clusters[large_idx]
        large_center = calculate_cluster_center(large_cluster, facial_encodings)
        
        if large_center is None:
            continue
        
        print(f"\n   Analyzing cluster {large_idx} ({len(large_cluster)} faces)...")
        
        similar_clusters = []
        
        # Check similarity with all other clusters
        for small_idx, small_cluster in enumerate(clusters):
            if small_idx == large_idx:
                continue
            
            # Focus on smaller clusters that might be fragments
            if len(small_cluster) > len(large_cluster) * 0.5:  # Skip clusters that are too large
                continue
            
            small_center = calculate_cluster_center(small_cluster, facial_encodings)
            if small_center is None:
                continue
            
            # Calculate similarity
            centroid_sim = np.dot(large_center, small_center)
            
            # Also calculate cross-cluster similarity
            similarities = calculate_inter_cluster_similarity(large_cluster, small_cluster, facial_encodings)
            
            # Decision criteria for potential merge
            if (centroid_sim > 0.4 or  # Centroid similarity > 0.4
                similarities['max_pairwise'] > 0.6 or  # Some faces very similar
                similarities['avg_pairwise'] > 0.45):  # Average similarity decent
                
                similar_clusters.append({
                    'cluster_idx': small_idx,
                    'size': len(small_cluster),
                    'centroid_sim': centroid_sim,
                    'max_pairwise': similarities['max_pairwise'],
                    'avg_pairwise': similarities['avg_pairwise']
                })
                
                print(f"      Potential merge: Cluster {small_idx} ({len(small_cluster)} faces)")
                print(f"         Centroid sim: {centroid_sim:.3f}")
                print(f"         Max pairwise: {similarities['max_pairwise']:.3f}")
                print(f"         Avg pairwise: {similarities['avg_pairwise']:.3f}")
        
        if similar_clusters:
            # Sort by similarity
            similar_clusters.sort(key=lambda x: x['centroid_sim'], reverse=True)
            merge_candidates[large_idx] = similar_clusters
    
    return merge_candidates

def merge_clusters_intelligently(clusters, facial_encodings, merge_candidates, 
                                merge_threshold=0.45, max_merges_per_cluster=5):
    """
    Intelligently merge clusters based on candidates
    
    Args:
        clusters: Original clusters
        facial_encodings: Dictionary of facial encodings
        merge_candidates: Output from identify_oversplit_clusters
        merge_threshold: Minimum similarity for merging
        max_merges_per_cluster: Maximum number of clusters to merge into one
        
    Returns:
        New merged clusters
    """
    print(f"\n🔧 Performing intelligent cluster merging...")
    
    # Create a copy of clusters
    new_clusters = [cluster.copy() for cluster in clusters]
    merged_indices = set()  # Track which clusters have been merged
    
    merge_actions = []
    
    for parent_idx, candidates in merge_candidates.items():
        if parent_idx in merged_indices:
            continue
        
        print(f"\n   Processing merges for cluster {parent_idx}...")
        
        merges_count = 0
        
        for candidate in candidates:
            if merges_count >= max_merges_per_cluster:
                break
            
            child_idx = candidate['cluster_idx']
            
            # Skip if already merged
            if child_idx in merged_indices:
                continue
            
            # Check if similarity meets threshold
            if (candidate['centroid_sim'] > merge_threshold or
                candidate['max_pairwise'] > merge_threshold + 0.1):
                
                # Merge child cluster into parent
                print(f"      ✅ Merging cluster {child_idx} → {parent_idx}")
                print(f"         Adding {len(new_clusters[child_idx])} faces")
                
                new_clusters[parent_idx].extend(new_clusters[child_idx])
                new_clusters[child_idx] = []  # Empty the merged cluster
                
                merged_indices.add(child_idx)
                merges_count += 1
                
                merge_actions.append({
                    'parent': parent_idx,
                    'child': child_idx,
                    'faces_added': len(clusters[child_idx]),
                    'similarity': candidate['centroid_sim']
                })
    
    # Remove empty clusters
    final_clusters = [cluster for cluster in new_clusters if len(cluster) > 0]
    
    print(f"\n📊 Merge Summary:")
    print(f"   Original clusters: {len(clusters)}")
    print(f"   Final clusters: {len(final_clusters)}")
    print(f"   Clusters merged: {len(merged_indices)}")
    print(f"   Merge actions performed: {len(merge_actions)}")
    
    for action in merge_actions:
        print(f"   Cluster {action['child']} → {action['parent']}: "
              f"+{action['faces_added']} faces (sim: {action['similarity']:.3f})")
    
    return final_clusters, merge_actions

def post_process_clusters_legacy(clusters, facial_encodings, target_clusters=None, 
                         merge_threshold=0.45, max_merges_per_cluster=5):
    """
    Main post-processing function
    
    Args:
        clusters: Original clusters from clustering algorithm
        facial_encodings: Dictionary of facial encodings
        target_clusters: Specific clusters to focus on (e.g., [2] for cluster 2)
        merge_threshold: Minimum similarity for merging
        max_merges_per_cluster: Maximum merges per parent cluster
        
    Returns:
        Processed clusters and merge report
    """
    print("🚀 Starting cluster post-processing...")
    
    # Step 1: Identify merge candidates
    merge_candidates = identify_oversplit_clusters(
        clusters, facial_encodings, target_clusters
    )
    
    if not merge_candidates:
        print("   No merge candidates found.")
        return clusters, []
    
    # Step 2: Perform merging
    final_clusters, merge_actions = merge_clusters_intelligently(
        clusters, facial_encodings, merge_candidates, 
        merge_threshold, max_merges_per_cluster
    )
    
    return final_clusters, merge_actions

def post_process_clusters(clusters, facial_encodings, strategy='intelligent', **kwargs):
    """
    新的統一post-processing入口，支持多種策略
    """
    if strategy == 'legacy':
        return post_process_clusters_legacy(clusters, facial_encodings, **kwargs)
    elif strategy == 'intelligent':
        # 使用新的智能兩階段策略
        total_faces = sum(len(cluster) for cluster in clusters)
        
        # Phase 1
        phase1_clusters, phase1_actions = merge_similar_large_clusters(
            clusters, facial_encodings,
            min_cluster_size=kwargs.get('min_large_cluster_size', max(10, total_faces * 0.02)),
            similarity_threshold=kwargs.get('large_cluster_threshold', 0.55),
            max_cross_sample_check=kwargs.get('max_cross_sample_check', 20)  # 添加這個參數
        )
        
        # Phase 2
        final_clusters, phase2_actions = merge_small_clusters_intelligently(
            phase1_clusters, facial_encodings,
            small_cluster_threshold=kwargs.get('small_cluster_threshold', total_faces * 0.05),
            merge_threshold=kwargs.get('small_cluster_merge_threshold', 0.45),
            safety_checks=kwargs.get('safety_checks', True)  # 添加這個參數
        )
        
        return final_clusters, phase1_actions + phase2_actions
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def save_post_processed_results(clusters, merge_actions, facial_encodings, output_dir):
    """
    Save post-processed results
    """
    print("\n💾 Saving post-processed results...")
    
    # Calculate new cluster centers
    from clustering import find_cluster_centers_adjusted
    cluster_centers = find_cluster_centers_adjusted(
        clusters, facial_encodings, method='min_distance'
    )
    
    # Save data
    post_processed_data = {
        'clusters': clusters,
        'facial_encodings': facial_encodings,
        'cluster_centers': cluster_centers,
        'merge_actions': merge_actions,
        'original_cluster_count': len(clusters) + len(merge_actions)
    }
    
    output_path = os.path.join(output_dir, 'post_processed_centers_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(post_processed_data, f)
    
    print(f"✅ Post-processed data saved to: {output_path}")
    
    return output_path

# Example usage function
def apply_post_processing_to_existing_results(centers_data_path, output_dir):
    """
    Apply post-processing to existing clustering results
    
    Args:
        centers_data_path: Path to existing centers_data.pkl
        output_dir: Directory to save results
    """
    print("📂 Loading existing clustering results...")
    
    # Load existing results
    with open(centers_data_path, 'rb') as f:
        centers_data = pickle.load(f)
    
    clusters = centers_data['clusters']
    facial_encodings = centers_data['facial_encodings']
    
    print(f"   Loaded {len(clusters)} clusters")
    
    # Apply post-processing with focus on cluster 2
    final_clusters, merge_actions = post_process_clusters(
        clusters, facial_encodings, 
        target_clusters=[2],  # Focus on cluster 2
        merge_threshold=0.45,
        max_merges_per_cluster=6  # Allow more merges for cluster 2
    )
    
    # Save results
    output_path = save_post_processed_results(
        final_clusters, merge_actions, facial_encodings, output_dir
    )
    
    return output_path, final_clusters, merge_actions