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
    Phase 1: Check and merge similar large clusters
    
    Args:
        clusters: Original clusters
        facial_encodings: Facial encoding dictionary
        min_cluster_size: Minimum size to be considered as "large cluster"
        similarity_threshold: Merge threshold (large clusters need higher similarity)
        max_cross_sample_check: Maximum samples per cluster for comparison
        
    Returns:
        Merged clusters and merge actions
    """
    print(f"🔍 Phase 1: Analyzing large clusters (size >= {min_cluster_size})...")
    
    # Progress display improvement
    from tqdm import tqdm
    
    large_clusters = [(i, cluster) for i, cluster in enumerate(clusters) if len(cluster) >= min_cluster_size]
    print(f"   Found {len(large_clusters)} large clusters")
    
    if len(large_clusters) < 2:
        print("   Insufficient large clusters, skipping Phase 1")
        return clusters, []
    
    # Use tqdm to show comparison progress
    total_comparisons = len(large_clusters) * (len(large_clusters) - 1) // 2
    merge_candidates = []
    
    pbar = tqdm(total=total_comparisons, desc="Comparing large clusters")
    for i in range(len(large_clusters)):
        for j in range(i+1, len(large_clusters)):
            # Comparison logic...
            pbar.update(1)
    pbar.close()

    # Identify large clusters
    large_clusters = []
    small_clusters = []
    
    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_cluster_size:
            large_clusters.append((i, cluster))
        else:
            small_clusters.append((i, cluster))
    
    print(f"   Found {len(large_clusters)} large clusters, {len(small_clusters)} small clusters")
    
    if len(large_clusters) < 2:
        print("   Insufficient large clusters, skipping Phase 1")
        return clusters, []
    
    # Calculate similarity between large clusters
    merge_candidates = []
    
    for i in range(len(large_clusters)):
        for j in range(i+1, len(large_clusters)):
            idx_i, cluster_i = large_clusters[i]
            idx_j, cluster_j = large_clusters[j]
            
            print(f"   Comparing Cluster {idx_i} ({len(cluster_i)} faces) vs Cluster {idx_j} ({len(cluster_j)} faces)")
            
            # Calculate cross-cluster similarity
            similarities = calculate_cross_cluster_similarity(
                cluster_i, cluster_j, facial_encodings, max_cross_sample_check
            )
            
            # Multiple judgment conditions
            centroid_sim = similarities['centroid']
            max_pairwise = similarities['max_pairwise']
            avg_pairwise = similarities['avg_pairwise']
            high_sim_ratio = similarities['high_similarity_ratio']  # Ratio of high similarity pairs
            
            print(f"      Centroid similarity: {centroid_sim:.3f}")
            print(f"      Max pairwise similarity: {max_pairwise:.3f}")
            print(f"      Average pairwise similarity: {avg_pairwise:.3f}")
            print(f"      High similarity ratio: {high_sim_ratio:.3f}")
            
            # Decision to merge (using multiple conditions)
            should_merge = (
                centroid_sim > similarity_threshold and
                (max_pairwise > similarity_threshold + 0.1 or  # Very similar faces exist
                 (avg_pairwise > similarity_threshold - 0.05 and high_sim_ratio > 0.3))  # Overall similar with sufficient high similarity pairs
            )
            
            if should_merge:
                merge_candidates.append({
                    'cluster_i': idx_i,
                    'cluster_j': idx_j,
                    'centroid_sim': centroid_sim,
                    'avg_pairwise': avg_pairwise,
                    'confidence': (centroid_sim + avg_pairwise + high_sim_ratio) / 3
                })
                print(f"      ✅ Marked for merge candidate")
            else:
                print(f"      ❌ Does not meet merge criteria")
    
    # Execute merging (sort by confidence)
    merge_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Use Union-Find to handle multi-way merging
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
        
        # Check if already in the same group
        if find_root(i) != find_root(j):
            union(i, j)
            merge_actions.append({
                'type': 'large_cluster_merge',
                'cluster_i': i,
                'cluster_j': j,
                'confidence': candidate['confidence'],
                'centroid_sim': candidate['centroid_sim']
            })
            print(f"   ✅ Merged Cluster {i} and Cluster {j} (confidence: {candidate['confidence']:.3f})")
    
    # Rebuild clusters
    group_to_cluster = {}
    for i, cluster in enumerate(clusters):
        root = find_root(i)
        if root not in group_to_cluster:
            group_to_cluster[root] = []
        group_to_cluster[root].extend(cluster)
    
    new_clusters = list(group_to_cluster.values())
    
    print(f"✅ Phase 1 complete: {len(clusters)} → {len(new_clusters)} clusters")
    return new_clusters, merge_actions


def calculate_cross_cluster_similarity(cluster1, cluster2, facial_encodings, max_samples=20):
    """
    Calculate detailed similarity metrics between two clusters
    """
    # Sample to avoid excessive computation
    sample1 = cluster1[:max_samples] if len(cluster1) <= max_samples else \
              np.random.choice(cluster1, max_samples, replace=False).tolist()
    sample2 = cluster2[:max_samples] if len(cluster2) <= max_samples else \
              np.random.choice(cluster2, max_samples, replace=False).tolist()
    
    # Calculate centroid similarity
    center1 = calculate_cluster_center(sample1, facial_encodings)
    center2 = calculate_cluster_center(sample2, facial_encodings)
    centroid_sim = np.dot(center1, center2) if center1 is not None and center2 is not None else 0
    
    # Calculate pairwise similarity
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
        'high_similarity_ratio': np.sum(similarities > 0.7) / len(similarities)  # High similarity ratio
    }


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

def identify_oversplit_clusters(clusters, facial_encodings, min_large_cluster_size=30):
    """
    Automatically identify all clusters that might be oversplit and need merging
    
    Args:
        clusters: List of clusters
        facial_encodings: Dictionary of facial encodings
        min_large_cluster_size: Minimum size to be considered as potential "parent" cluster
        
    Returns:
        Dictionary of potential merges
    """
    print("🔍 Automatically identifying potentially oversplit clusters...")
    
    merge_candidates = {}
    
    # Automatically identify large clusters as potential "parent" clusters
    large_clusters = []
    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_large_cluster_size:
            large_clusters.append(i)
            print(f"   Large cluster {i}: {len(cluster)} faces")
    
    # Use all identified large clusters as potential parents
    print(f"   Found {len(large_clusters)} large clusters as potential merge targets")
    
    # For each large cluster, find similar small clusters
    for large_idx in large_clusters:
        large_cluster = clusters[large_idx]
        large_center = calculate_cluster_center(large_cluster, facial_encodings)
        
        if large_center is None:
            continue
        
        print(f"\n   Analyzing cluster {large_idx} ({len(large_cluster)} faces)...")
        
        similar_clusters = []
        
        # Check similarity with all other clusters (focus on smaller clusters)
        for small_idx, small_cluster in enumerate(clusters):
            if small_idx == large_idx:
                continue
            
            # Focus on clusters that are significantly smaller than the large cluster
            # but not tiny noise clusters
            max_small_size = len(large_cluster) * 0.3  # At most 30% of large cluster size
            min_small_size = 3  # At least 3 faces to be considered valid
            
            if not (min_small_size <= len(small_cluster) <= max_small_size):
                continue
            
            small_center = calculate_cluster_center(small_cluster, facial_encodings)
            if small_center is None:
                continue
            
            # Calculate similarity
            centroid_sim = np.dot(large_center, small_center)
            
            # Also calculate cross-cluster similarity
            similarities = calculate_inter_cluster_similarity(large_cluster, small_cluster, facial_encodings)
            
            # Enhanced decision criteria for potential merge
            # Check multiple conditions to determine if they might be the same person
            is_potential_match = (
                centroid_sim > 0.45 or  # Centroid similarity > 0.45
                similarities['max_pairwise'] > 0.65 or  # Some faces very similar
                (similarities['avg_pairwise'] > 0.5 and centroid_sim > 0.35)  # Good overall similarity with decent centroid
            )
            
            if is_potential_match:
                # Calculate confidence score for ranking
                confidence_score = (
                    centroid_sim * 0.4 +
                    similarities['max_pairwise'] * 0.3 +
                    similarities['avg_pairwise'] * 0.3
                )
                
                similar_clusters.append({
                    'cluster_idx': small_idx,
                    'size': len(small_cluster),
                    'centroid_sim': centroid_sim,
                    'max_pairwise': similarities['max_pairwise'],
                    'avg_pairwise': similarities['avg_pairwise'],
                    'confidence': confidence_score
                })
                
                print(f"      Potential merge: Cluster {small_idx} ({len(small_cluster)} faces)")
                print(f"         Centroid sim: {centroid_sim:.3f}")
                print(f"         Max pairwise: {similarities['max_pairwise']:.3f}")
                print(f"         Avg pairwise: {similarities['avg_pairwise']:.3f}")
                print(f"         Confidence: {confidence_score:.3f}")
        
        if similar_clusters:
            # Sort by confidence score (highest first)
            similar_clusters.sort(key=lambda x: x['confidence'], reverse=True)
            merge_candidates[large_idx] = similar_clusters
    
    return merge_candidates

def merge_clusters_intelligently(clusters, facial_encodings, merge_candidates, 
                                merge_threshold=0.5, max_merges_per_cluster=8,
                                enable_cross_validation=True):
    """
    Intelligently merge clusters based on candidates with enhanced validation
    
    Args:
        clusters: Original clusters
        facial_encodings: Dictionary of facial encodings
        merge_candidates: Output from identify_oversplit_clusters
        merge_threshold: Minimum confidence score for merging
        max_merges_per_cluster: Maximum number of clusters to merge into one
        enable_cross_validation: Whether to perform additional validation
        
    Returns:
        New merged clusters
    """
    print(f"\n🔧 Performing intelligent cluster merging...")
    print(f"   Merge threshold: {merge_threshold}")
    print(f"   Max merges per cluster: {max_merges_per_cluster}")
    
    # Create a copy of clusters
    new_clusters = [cluster.copy() for cluster in clusters]
    merged_indices = set()  # Track which clusters have been merged
    
    merge_actions = []
    
    # Process all merge candidates
    for parent_idx, candidates in merge_candidates.items():
        if parent_idx in merged_indices:
            continue
        
        print(f"\n   Processing merges for cluster {parent_idx}...")
        
        merges_count = 0
        
        for candidate in candidates:
            if merges_count >= max_merges_per_cluster:
                print(f"      Reached maximum merges limit ({max_merges_per_cluster}) for cluster {parent_idx}")
                break
            
            child_idx = candidate['cluster_idx']
            
            # Skip if already merged
            if child_idx in merged_indices:
                continue
            
            # Check if confidence meets threshold
            confidence = candidate['confidence']
            if confidence > merge_threshold:
                
                # Additional cross-validation if enabled
                validation_passed = True
                if enable_cross_validation:
                    validation_passed = validate_merge_quality(
                        clusters[child_idx], clusters[parent_idx], facial_encodings
                    )
                
                if validation_passed:
                    # Merge child cluster into parent
                    print(f"      ✅ Merging cluster {child_idx} → {parent_idx}")
                    print(f"         Adding {len(new_clusters[child_idx])} faces (confidence: {confidence:.3f})")
                    
                    new_clusters[parent_idx].extend(new_clusters[child_idx])
                    new_clusters[child_idx] = []  # Empty the merged cluster
                    
                    merged_indices.add(child_idx)
                    merges_count += 1
                    
                    merge_actions.append({
                        'type': 'intelligent_cluster_merge',
                        'cluster_i': parent_idx,
                        'cluster_j': child_idx,
                        'faces_added': len(clusters[child_idx]),
                        'confidence': confidence,
                        'centroid_sim': candidate['centroid_sim'],
                        'max_pairwise': candidate['max_pairwise']
                    })
                else:
                    print(f"      ❌ Merge validation failed for cluster {child_idx} → {parent_idx}")
            else:
                print(f"      ❌ Confidence too low for cluster {child_idx} → {parent_idx} ({confidence:.3f} < {merge_threshold})")
    
    # Remove empty clusters
    final_clusters = [cluster for cluster in new_clusters if len(cluster) > 0]
    
    print(f"\n📊 Merge Summary:")
    print(f"   Original clusters: {len(clusters)}")
    print(f"   Final clusters: {len(final_clusters)}")
    print(f"   Clusters merged: {len(merged_indices)}")
    print(f"   Merge actions performed: {len(merge_actions)}")
    
    for action in merge_actions:
        print(f"   Cluster {action['cluster_j']} → {action['cluster_i']}: "
              f"+{action['faces_added']} faces (confidence: {action.get('confidence', action.get('similarity', 0)):.3f})")
    
    return final_clusters, merge_actions


def validate_merge_quality(small_cluster, large_cluster, facial_encodings, 
                          min_internal_similarity=0.4, max_consistency_drop=0.1):
    """
    Validate if merging two clusters would maintain good quality
    
    Args:
        small_cluster: Small cluster to be merged
        large_cluster: Large cluster (merge target)
        facial_encodings: Dictionary of facial encodings
        min_internal_similarity: Minimum internal similarity for small cluster
        max_consistency_drop: Maximum allowed consistency drop
        
    Returns:
        bool: True if merge is validated, False otherwise
    """
    # Check 1: Small cluster should have reasonable internal consistency
    small_consistency = calculate_internal_cluster_consistency(small_cluster, facial_encodings)
    if small_consistency < min_internal_similarity:
        return False
    
    # Check 2: Merge shouldn't significantly hurt large cluster consistency
    large_consistency = calculate_internal_cluster_consistency(large_cluster[:30], facial_encodings)
    
    # Simulate merged consistency
    combined_sample = large_cluster[:20] + small_cluster[:10]
    combined_consistency = calculate_internal_cluster_consistency(combined_sample, facial_encodings)
    
    consistency_drop = large_consistency - combined_consistency
    if consistency_drop > max_consistency_drop:
        return False
    
    return True

def post_process_clusters_legacy(clusters, facial_encodings, min_large_cluster_size=30,
                         merge_threshold=0.5, max_merges_per_cluster=8):
    """
    Main post-processing function that automatically processes all clusters
    
    Args:
        clusters: Original clusters from clustering algorithm
        facial_encodings: Dictionary of facial encodings
        min_large_cluster_size: Minimum size to be considered as large cluster
        merge_threshold: Minimum confidence score for merging
        max_merges_per_cluster: Maximum merges per parent cluster
        
    Returns:
        Processed clusters and merge report
    """
    print("🚀 Starting automatic cluster post-processing...")
    
    # Step 1: Automatically identify merge candidates for all clusters
    merge_candidates = identify_oversplit_clusters(
        clusters, facial_encodings, min_large_cluster_size
    )
    
    if not merge_candidates:
        print("   No merge candidates found.")
        return clusters, []
    
    print(f"   Found merge candidates for {len(merge_candidates)} large clusters")
    
    # Step 2: Perform intelligent merging
    final_clusters, merge_actions = merge_clusters_intelligently(
        clusters, facial_encodings, merge_candidates, 
        merge_threshold, max_merges_per_cluster
    )
    
    return final_clusters, merge_actions

def post_process_clusters(clusters, facial_encodings, strategy='small_to_large_only', **kwargs):
    """
    New unified post-processing entry point - only merge small clusters to large clusters
    """
    if strategy == 'legacy':
        return post_process_clusters_legacy(clusters, facial_encodings, **kwargs)
    elif strategy == 'small_to_large_only':
        # Only merge small clusters to large clusters, no large-to-large merging
        final_clusters, merge_actions = merge_small_clusters_to_large_only(
            clusters, facial_encodings,
            min_large_cluster_size=kwargs.get('min_large_cluster_size', 50),  # Stricter threshold
            small_cluster_percentage=kwargs.get('small_cluster_percentage', 0.05),  # 5% of total faces
            merge_threshold=kwargs.get('merge_threshold', 0.65),  # Much stricter threshold
            max_merges_per_cluster=kwargs.get('max_merges_per_cluster', 3),  # Fewer merges allowed
            safety_checks=kwargs.get('safety_checks', True)
        )
        
        return final_clusters, merge_actions
    elif strategy == 'intelligent':
        # Use new intelligent two-phase strategy
        total_faces = sum(len(cluster) for cluster in clusters)
        
        # Phase 1
        phase1_clusters, phase1_actions = merge_similar_large_clusters(
            clusters, facial_encodings,
            min_cluster_size=kwargs.get('min_large_cluster_size', max(10, total_faces * 0.02)),
            similarity_threshold=kwargs.get('large_cluster_threshold', 0.55),
            max_cross_sample_check=kwargs.get('max_cross_sample_check', 20)
        )
        
        # Phase 2
        final_clusters, phase2_actions = merge_small_clusters_intelligently(
            phase1_clusters, facial_encodings,
            small_cluster_threshold=kwargs.get('small_cluster_threshold', total_faces * 0.05),
            merge_threshold=kwargs.get('small_cluster_merge_threshold', 0.45),
            safety_checks=kwargs.get('safety_checks', True)
        )
        
        return final_clusters, phase1_actions + phase2_actions
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def merge_small_clusters_to_large_only(clusters, facial_encodings, min_large_cluster_size=50,
                                      small_cluster_percentage=0.1, merge_threshold=0.55, 
                                      max_merges_per_cluster=5, safety_checks=True):
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

# Example usage function
def apply_post_processing_to_existing_results(centers_data_path, output_dir):
    """
    Apply automatic post-processing to existing clustering results
    
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
    
    # Apply automatic post-processing to all clusters
    final_clusters, merge_actions = post_process_clusters(
        clusters, facial_encodings, 
        strategy='intelligent',
        min_large_cluster_size=30,  # Automatically detect large clusters
        merge_threshold=0.5,  # Confidence threshold for merging
        max_merges_per_cluster=8  # Allow up to 8 merges per large cluster
    )
    
    # Save results
    output_path = save_post_processed_results(
        final_clusters, merge_actions, facial_encodings, output_dir
    )
    
    return output_path, final_clusters, merge_actions