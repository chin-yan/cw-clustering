# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from collections import Counter

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

def post_process_clusters(clusters, facial_encodings, target_clusters=None, 
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