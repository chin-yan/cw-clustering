# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from collections import Counter


def merge_small_clusters_intelligently(clusters, facial_encodings, small_cluster_threshold=50,
                                     merge_threshold=0.45, safety_checks=True):
    """
    Phase 2: Intelligently merge small clusters with safety checks
    
    Args:
        clusters: Clusters after Phase 1 processing
        facial_encodings: Facial encoding dictionary
        small_cluster_threshold: Small cluster threshold
        merge_threshold: Merge threshold
        safety_checks: Whether to enable safety checks
        
    Returns:
        Final clusters and merge actions
    """
    print(f"🔍 Phase 2: Processing small clusters (size < {small_cluster_threshold})...")
    
    # Identify large and small clusters
    large_clusters = []
    small_clusters = []
    
    for i, cluster in enumerate(clusters):
        if len(cluster) >= small_cluster_threshold:
            large_clusters.append((i, cluster))
        else:
            small_clusters.append((i, cluster))
    
    print(f"   Large clusters: {len(large_clusters)}, Small clusters: {len(small_clusters)}")
    
    if not small_clusters:
        print("   No small clusters to process")
        return clusters, []
    
    # Find best merge target for each small cluster
    merge_proposals = []
    
    for small_idx, small_cluster in small_clusters:
        best_match = None
        best_similarity = 0
        
        print(f"   Analyzing small cluster {small_idx} ({len(small_cluster)} faces)...")
        
        # Compare with all large clusters
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
            # Safety checks
            if safety_checks:
                is_safe = perform_safety_checks(
                    small_cluster, clusters[best_match], facial_encodings, merge_threshold
                )
                if not is_safe:
                    print(f"      ❌ Safety check failed, skipping merge")
                    continue
            
            merge_proposals.append({
                'small_cluster': small_idx,
                'target_cluster': best_match,
                'similarity': best_similarity,
                'small_size': len(small_cluster)
            })
            print(f"      ✅ Suggest merge to Cluster {best_match} (similarity: {best_similarity:.3f})")
        else:
            print(f"      ❌ No suitable merge target found (highest similarity: {best_similarity:.3f})")
    
    # Execute merging
    new_clusters = [cluster.copy() for cluster in clusters]
    merge_actions = []
    
    # Sort by similarity, prioritize high similarity merges
    merge_proposals.sort(key=lambda x: x['similarity'], reverse=True)
    
    merged_indices = set()
    
    for proposal in merge_proposals:
        small_idx = proposal['small_cluster']
        target_idx = proposal['target_cluster']
        
        if small_idx not in merged_indices:
            # Execute merge
            new_clusters[target_idx].extend(new_clusters[small_idx])
            new_clusters[small_idx] = []  # Clear merged cluster
            merged_indices.add(small_idx)
            
            merge_actions.append({
                'type': 'small_cluster_merge',
                'cluster_i': target_idx,
                'cluster_j': small_idx,
                'faces_added': proposal['small_size'],
                'similarity': proposal['similarity']
            })
            print(f"   ✅ Merged Cluster {small_idx} → Cluster {target_idx}")
    
    # Remove empty clusters
    final_clusters = [cluster for cluster in new_clusters if len(cluster) > 0]
    
    print(f"✅ Phase 2 complete: Merged {len(merge_actions)} small clusters")
    print(f"   Final cluster count: {len(final_clusters)}")
    
    return final_clusters, merge_actions


def perform_safety_checks(small_cluster, target_cluster, facial_encodings, base_threshold):
    """
    Perform safety checks to prevent incorrect merging
    
    Args:
        small_cluster: Small cluster to be merged
        target_cluster: Target cluster
        facial_encodings: Facial encoding dictionary
        base_threshold: Base threshold
        
    Returns:
        bool: True if safe, False if unsafe
    """
    # Safety check 1: Avoid extreme size differences (prevent noise from being merged into main cluster)
    size_ratio = len(small_cluster) / len(target_cluster)
    if size_ratio < 0.02 and len(small_cluster) < 5:  # Too small and extremely low ratio
        print(f"        Safety check 1 failed: cluster too small and ratio too low ({len(small_cluster)}/{len(target_cluster)})")
        return False
    
    # Safety check 2: Check internal consistency of small cluster
    small_internal_similarity = calculate_internal_cluster_consistency(small_cluster, facial_encodings)
    if small_internal_similarity < base_threshold - 0.1:  # Small cluster internally inconsistent, likely noise
        print(f"        Safety check 2 failed: small cluster internal consistency too low ({small_internal_similarity:.3f})")
        return False
    
    # Safety check 3: Check that merging won't significantly reduce target cluster consistency
    target_internal = calculate_internal_cluster_consistency(target_cluster[:50], facial_encodings)  # Sample 50
    
    # Simulate post-merge consistency
    combined_sample = target_cluster[:25] + small_cluster[:25]  # 25 samples each
    combined_internal = calculate_internal_cluster_consistency(combined_sample, facial_encodings)
    
    consistency_drop = target_internal - combined_internal
    if consistency_drop > 0.15:  # Consistency drops too much
        print(f"        Safety check 3 failed: merge would significantly reduce target cluster consistency (drop {consistency_drop:.3f})")
        return False
    
    print(f"        ✅ Passed all safety checks")
    return True


def calculate_internal_cluster_consistency(cluster, facial_encodings, max_samples=20):
    """
    Calculate internal consistency of cluster (average pairwise similarity)
    """
    if len(cluster) < 2:
        return 1.0
    
    # Sample to avoid excessive computation
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

def post_process_clusters(clusters, facial_encodings, strategy='small_to_large_only', **kwargs):
    """
    New unified post-processing entry point - only merge small clusters to large clusters
    """
    if strategy == 'small_to_large_only':
        # Only merge small clusters to large clusters, no large-to-large merging
        final_clusters, merge_actions = merge_small_clusters_to_large_only(
            clusters, facial_encodings,
            min_large_cluster_size=kwargs.get('min_large_cluster_size', 50),  # Stricter threshold
            small_cluster_percentage=kwargs.get('small_cluster_percentage', 0.05),  # 5% of total faces
            merge_threshold=kwargs.get('merge_threshold', 0.4),  # Much stricter threshold
            max_merges_per_cluster=kwargs.get('max_merges_per_cluster', 10),  # Fewer merges allowed
            safety_checks=kwargs.get('safety_checks', True)
        )
        
        return final_clusters, merge_actions
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def merge_small_clusters_to_large_only(clusters, facial_encodings, min_large_cluster_size=50,
                                      small_cluster_percentage=0.08, merge_threshold=0.4, 
                                      max_merges_per_cluster=10, safety_checks=True):
    """
    Only merge small clusters to large clusters with strict conditions
    
    Args:
        clusters: Original clusters
        facial_encodings: Facial encoding dictionary
        min_large_cluster_size: Minimum size to be considered as large cluster
        small_cluster_percentage: Small clusters defined as < this percentage of total faces
        merge_threshold: Minimum similarity for merging (much stricter)
        max_merges_per_cluster: Maximum merges per large cluster (reduced)
        safety_checks: Whether to enable safety checks
        
    Returns:
        Final clusters and merge actions
    """
    total_faces = sum(len(cluster) for cluster in clusters)
    small_cluster_threshold = max(5, int(total_faces * small_cluster_percentage))  # At least 5 faces
    
    print(f"Processing small-to-large cluster merging only...")
    print(f"   Total faces: {total_faces}")
    print(f"   Large cluster threshold: >= {min_large_cluster_size} faces")
    print(f"   Small cluster threshold: <= {small_cluster_threshold} faces ({small_cluster_percentage*100}% of total)")
    print(f"   Merge threshold: {merge_threshold}")
    
    # Identify large and small clusters with adaptive criteria
    large_clusters = []
    small_clusters = []
    medium_clusters = []
    
    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_large_cluster_size:
            large_clusters.append((i, cluster))
        elif len(cluster) <= small_cluster_threshold:
            small_clusters.append((i, cluster))
        else:
            medium_clusters.append((i, cluster))
    
    print(f"   Large clusters: {len(large_clusters)}")
    print(f"   Small clusters: {len(small_clusters)}")
    print(f"   Medium clusters (will not be processed): {len(medium_clusters)}")
    
    if not small_clusters or not large_clusters:
        print("   No valid small or large clusters to process")
        return clusters, []
    
    # Find merge candidates for small clusters only
    merge_proposals = []
    
    for small_idx, small_cluster in small_clusters:
        best_match = None
        best_similarity = 0
        
        print(f"   Analyzing small cluster {small_idx} ({len(small_cluster)} faces)...")
        
        # Compare with all large clusters
        for large_idx, large_cluster in large_clusters:
            similarities = calculate_inter_cluster_similarity(
                small_cluster, large_cluster, facial_encodings
            )
            
            # Much stricter criteria for merging
            centroid_sim = similarities['centroid']
            max_pairwise = similarities['max_pairwise']
            avg_pairwise = similarities['avg_pairwise']
            
            # All conditions must be met for a potential merge
            meets_criteria = (
                centroid_sim > merge_threshold and
                max_pairwise > merge_threshold + 0.05 and
                avg_pairwise > merge_threshold - 0.1
            )
            
            if meets_criteria:
                # Calculate conservative combined score
                combined_score = min(centroid_sim, max_pairwise, avg_pairwise + 0.1)
                
                if combined_score > best_similarity:
                    best_similarity = combined_score
                    best_match = large_idx
                    
                print(f"      Potential match with Cluster {large_idx}: score {combined_score:.3f}")
                print(f"         Centroid: {centroid_sim:.3f}, Max: {max_pairwise:.3f}, Avg: {avg_pairwise:.3f}")
        
        if best_match is not None:
            # Enhanced safety checks
            if safety_checks:
                is_safe = enhanced_safety_checks(
                    small_cluster, clusters[best_match], facial_encodings, merge_threshold
                )
                if not is_safe:
                    print(f"      Safety check failed, skipping merge")
                    continue
            
            merge_proposals.append({
                'small_cluster': small_idx,
                'target_cluster': best_match,
                'similarity': best_similarity,
                'small_size': len(small_cluster)
            })
            print(f"      Approved for merge to Cluster {best_match} (score: {best_similarity:.3f})")
        else:
            print(f"      No suitable merge target found")
    
    # Execute merging with strict limits
    new_clusters = [cluster.copy() for cluster in clusters]
    merge_actions = []
    
    # Sort by similarity (highest first)
    merge_proposals.sort(key=lambda x: x['similarity'], reverse=True)
    
    merged_indices = set()
    large_cluster_merge_counts = {}
    
    for proposal in merge_proposals:
        small_idx = proposal['small_cluster']
        target_idx = proposal['target_cluster']
        
        # Check merge limits
        current_merges = large_cluster_merge_counts.get(target_idx, 0)
        if current_merges >= max_merges_per_cluster:
            print(f"   Cluster {target_idx} has reached merge limit ({max_merges_per_cluster}), skipping")
            continue
        
        if small_idx not in merged_indices:
            # Execute merge
            new_clusters[target_idx].extend(new_clusters[small_idx])
            new_clusters[small_idx] = []  # Clear merged cluster
            merged_indices.add(small_idx)
            large_cluster_merge_counts[target_idx] = current_merges + 1
            
            merge_actions.append({
                'type': 'small_to_large_merge',
                'cluster_i': target_idx,
                'cluster_j': small_idx,
                'faces_added': proposal['small_size'],
                'similarity': proposal['similarity']
            })
            print(f"   Merged Cluster {small_idx} -> Cluster {target_idx}")
    
    # Remove empty clusters
    final_clusters = [cluster for cluster in new_clusters if len(cluster) > 0]
    
    print(f"Merge complete: {len(merge_actions)} small clusters merged")
    print(f"   Final cluster count: {len(final_clusters)}")
    
    return final_clusters, merge_actions


def enhanced_safety_checks(small_cluster, target_cluster, facial_encodings, base_threshold):
    """
    Enhanced safety checks with stricter criteria
    """
    # Check 1: Minimum cluster size ratio (stricter)
    size_ratio = len(small_cluster) / len(target_cluster)
    if size_ratio < 0.03 and len(small_cluster) < 3:  # Stricter size requirements
        print(f"        Safety check 1 failed: cluster too small ({len(small_cluster)}/{len(target_cluster)})")
        return False
    
    # Check 2: Small cluster internal consistency (stricter)
    small_internal_similarity = calculate_internal_cluster_consistency(small_cluster, facial_encodings)
    if small_internal_similarity < base_threshold:  # Must meet full threshold
        print(f"        Safety check 2 failed: small cluster inconsistent ({small_internal_similarity:.3f})")
        return False
    
    # Check 3: Target cluster consistency preservation (stricter)
    target_internal = calculate_internal_cluster_consistency(target_cluster[:30], facial_encodings)
    
    # Simulate merged consistency
    combined_sample = target_cluster[:20] + small_cluster
    combined_internal = calculate_internal_cluster_consistency(combined_sample, facial_encodings)
    
    consistency_drop = target_internal - combined_internal
    if consistency_drop > 0.08:  # Stricter consistency preservation
        print(f"        Safety check 3 failed: consistency drop too large ({consistency_drop:.3f})")
        return False
    
    # Check 4: Cross-validation with random samples
    random_similarities = []
    import random
    for _ in range(5):  # 5 random cross-checks
        if len(small_cluster) > 1 and len(target_cluster) > 1:
            small_sample = random.choice(small_cluster)
            target_sample = random.choice(target_cluster)
            sim = np.dot(facial_encodings[small_sample], facial_encodings[target_sample])
            random_similarities.append(sim)
    
    if random_similarities and np.mean(random_similarities) < base_threshold - 0.05:
        print(f"        Safety check 4 failed: random cross-validation failed ({np.mean(random_similarities):.3f})")
        return False
    
    print(f"        Passed all enhanced safety checks")
    return True

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