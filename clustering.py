# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import networkx as nx
from random import shuffle
from tqdm import tqdm
import re
from scipy.spatial import distance
import math

def face_distance(face_encodings, face_to_compare):
    """
    Calculate cosine similarity between facial encodings

    Args:
        face_encodings: face encoding list
        face_to_compare: face encoding for comparison

    Returns:
        Similarity score array
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    
    # Use cosine similarity (dot product) to calculate similarity
    return np.sum(face_encodings * face_to_compare, axis=1)

def extract_frame_info(image_path):
    """
    Extract frame number and face index from image path
    
    Args:
        image_path: Path to the face image
        
    Returns:
        Frame number and face index
    """
    # Extract the base filename
    basename = os.path.basename(image_path)
    # Expected format: frame_XXXXXX_face_Y.jpg
    match = re.match(r'frame_(\d+)_face_(\d+)\.jpg', basename)
    
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    else:
        # If the format doesn't match, return defaults
        return -1, -1

def compute_face_quality(face_path):
    """
    Compute the quality score for a face image with more permissive criteria
    Higher scores indicate better quality (more frontal, better lighting)
    
    Args:
        face_path: Path to the face image
        
    Returns:
        Quality score (0-1)
    """
    # Load the image
    img = cv2.imread(face_path)
    if img is None:
        return 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute quality metrics
    # 1. Variance of Laplacian (measure of image sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian) / 100.0  # Normalize
    sharpness = min(1.0, sharpness)  # Cap at 1.0
    
    # 2. Histogram spread (measure of contrast)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()  # Normalize histogram
    non_zero_bins = np.count_nonzero(hist_norm > 0.0005)  # More permissive threshold
    contrast = non_zero_bins / 256.0
    
    # 3. Face size relative to image size (larger faces are usually better)
    height, width = gray.shape
    face_area = height * width
    face_size_score = min(1.0, face_area / (160.0 * 160.0))
    
    # Combine metrics (adjusted weights - less emphasis on sharpness)
    #quality_score = 0.35 * sharpness + 0.35 * contrast + 0.3 * face_size_score
    quality_score = 0.5 * sharpness + 0.2 * contrast + 0.3 * face_size_score
    
    # Make overall scoring more permissive
    #quality_score = min(1.0, quality_score * 1.2)  # Boost scores by 20%
    quality_score = min(1.0, quality_score * 1.1)

    return quality_score

def cluster_facial_encodings(facial_encodings, threshold=None,iterations=50,temporal_weight=0.25):
    """
    Improved clustering for face encoding using Chinese Whispers algorithm
    with temporal analysis but more balanced parameters to avoid over-clustering
    
    Args:
        facial_encodings: mapping of face paths to encodings
        threshold: face matching threshold, default is 0.65 (stricter than previous)
        iterations: number of iterations
        temporal_weight: weight for temporal continuity (reduced from previous)
        
    Returns:
        Sorted list of clusters
    """
    # Initialize variables
    original_threshold = threshold
    iterations_completed = 0
    converged = False

    # Prepare data
    encoding_list = list(facial_encodings.items())
    if len(encoding_list) <= 1:
        print("Insufficient number of encodings to cluster")
        
        # Return empty result with metadata
        clustering_metadata = {
            'threshold': threshold if threshold else 0.0,
            'threshold_type': 'N/A',
            'iterations_completed': 0,
            'max_iterations': iterations,
            'converged': False,
            'total_faces': len(encoding_list),
            'num_clusters': 0,
            'temporal_weight': temporal_weight
        }
        return [], clustering_metadata
    
    print(f"Using adjusted Chinese Whispers algorithm to cluster {len(encoding_list)} faces...")
    
    # Adaptive threshold calculation
    original_threshold = threshold  # Store original value for metadata
    if threshold is None or (isinstance(threshold, str) and threshold == 'adaptive'):
        print("Calculating adaptive threshold based on similarity distribution...")
        
        # Sample faces for threshold estimation (use all if dataset is small)
        sample_size = min(1000, len(encoding_list))
        if len(encoding_list) > sample_size:
            sample_indices = np.random.choice(len(encoding_list), sample_size, replace=False)
            sample_encodings = [encoding_list[i][1] for i in sample_indices]
        else:
            sample_encodings = [enc for _, enc in encoding_list]
        
        # Calculate pairwise similarities for sample
        sample_similarities = []
        for i in range(min(500, len(sample_encodings))):
            for j in range(i+1, min(i+50, len(sample_encodings))):
                sim = np.dot(sample_encodings[i], sample_encodings[j])
                sample_similarities.append(sim)
        
        # Statistical analysis of similarity distribution
        sample_similarities = np.array(sample_similarities)
        mean_sim = np.mean(sample_similarities)
        std_sim = np.std(sample_similarities)
        median_sim = np.median(sample_similarities)
        
        # Calculate adaptive threshold using percentile method
        # For FaceNet, we should use higher percentiles (90-95)
        percentile_90 = np.percentile(sample_similarities, 90)
        percentile_95 = np.percentile(sample_similarities, 95)
        percentile_85 = np.percentile(sample_similarities, 85)
        
        # Adaptive threshold formula: weighted combination
        # Use percentile-based approach suitable for face recognition
        if std_sim < 0.15:  # Tight distribution - very consistent faces
            # Use higher threshold, weighted toward 95th percentile
            threshold = 0.2 * percentile_85 + 0.5 * percentile_90 + 0.3 * percentile_95
        elif std_sim < 0.25:  # Medium distribution
            # Balance between 90th and 95th percentile
            threshold = 0.3 * percentile_85 + 0.5 * percentile_90 + 0.2 * percentile_95
        else:  # Loose distribution - varied faces
            # More conservative, use 90th percentile primarily
            threshold = 0.4 * percentile_85 + 0.6 * percentile_90
        
        # Apply bounds suitable for FaceNet (typically 0.6-0.8 for same person)
        # For clustering, we can be slightly more permissive but still strict
        threshold = np.clip(threshold, 0.55, 0.75)
        
        # Additional heuristic: if mean similarity is very low, increase threshold
        if mean_sim < 0.3:
            # Dataset has very diverse faces, use more conservative threshold
            threshold = max(threshold, 0.60)
        
        print(f"📊 Similarity distribution statistics:")
        print(f"   Mean: {mean_sim:.4f}, Std: {std_sim:.4f}, Median: {median_sim:.4f}")
        print(f"   85th percentile: {percentile_85:.4f}, 90th percentile: {percentile_90:.4f}, 95th percentile: {percentile_95:.4f}")
        print(f"✅ Adaptive threshold selected: {threshold:.4f}")
    else:
        # Use provided threshold
        threshold = float(threshold)
        print(f"Using manual threshold: {threshold:.4f}")

    # Extract frame information for each face
    frame_info = {}
    quality_scores = {}
    
    print("Extracting frame information and computing quality scores...")
    for path, _ in tqdm(encoding_list):
        frame_num, face_idx = extract_frame_info(path)
        frame_info[path] = (frame_num, face_idx)
        # Compute quality score for each face
        quality_scores[path] = compute_face_quality(path)
    
    # Create graph with enhanced edge weights
    sorted_clusters = _chinese_whispers_adjusted(
        encoding_list, frame_info, quality_scores, threshold, iterations, temporal_weight
    )
    
    # Post-process clusters with stricter parameters
    final_clusters = _post_process_clusters(sorted_clusters, facial_encodings, frame_info, threshold + 0.1)
    #final_clusters = sorted_clusters

    print(f"Clustering completed, a total of {len(final_clusters)} clusters")

    # Store clustering metadata for logging
    clustering_metadata = {
        'threshold': threshold,
        'threshold_type': 'adaptive' if (original_threshold is None or 
                                        (isinstance(original_threshold, str) and original_threshold == 'adaptive')) 
                                    else 'manual',
        'iterations_completed': iterations_completed + 1,
        'max_iterations': iterations,
        'converged': converged,
        'total_faces': len(encoding_list),
        'num_clusters': len(final_clusters),
        'temporal_weight': temporal_weight
    }

    #Return both clusters and metadata
    return final_clusters, clustering_metadata

def _chinese_whispers_adjusted(encoding_list, frame_info, quality_scores, threshold=0.55, 
                              iterations=50, temporal_weight=0.25):
    """
    Adjusted implementation of Chinese Whispers Clustering Algorithm
    with better balance between facial similarity and temporal continuity
    
    Args:
        encoding_list: list of (image path, face encoding) tuples
        frame_info: dictionary mapping image paths to (frame_num, face_idx)
        quality_scores: dictionary mapping image paths to face quality scores
        threshold: face matching threshold
        iterations: number of iterations
        temporal_weight: weight for temporal continuity (0-1)
        
    Returns:
        List of clusters sorted by size
    """
    # Convergence parameters
    convergence_threshold = 0.99  # Stop if 99% nodes don't change cluster
    consecutive_stable_iterations = 5 # Require 3 consecutive stable iterations

    # Prepare data
    image_paths, encodings = zip(*encoding_list)
    encodings = np.array(encodings)
    
    # Create graph
    nodes = []
    edges = []
    
    # Maximum frame difference to consider for temporal continuity
    max_frame_diff = 3  # Reduced from 5 to be more conservative
    
    print("Creating enhanced graph with temporal continuity...")
    for idx, face_encoding_to_check in enumerate(tqdm(encodings)):
        # Add nodes
        node_id = idx + 1
        node = (node_id, {
            'cluster': image_paths[idx], 
            'path': image_paths[idx],
            'quality': quality_scores[image_paths[idx]]
        })
        nodes.append(node)
        
        # If it is the last element, no edge is created
        if (idx + 1) >= len(encodings):
            break
            
        # Calculate facial similarity with other encodings
        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        
        # Get frame info for current face
        curr_frame_num, curr_face_idx = frame_info[image_paths[idx]]
        
        # Add edges with facial similarity
        encoding_edges = []
        for i, distance in enumerate(distances):
            compare_idx = idx + i + 1
            compare_path = image_paths[compare_idx]
            compare_frame_num, compare_face_idx = frame_info[compare_path]
            
            # Only apply temporal boost if face similarity is already close to threshold
            if distance >= (threshold * 0.9):  # Only apply temporal boost for faces that are already similar
                # Calculate temporal similarity - higher weight for faces from close frames
                temporal_similarity = 0
                if curr_frame_num > 0 and compare_frame_num > 0:
                    frame_diff = abs(curr_frame_num - compare_frame_num)
                    if frame_diff <= max_frame_diff:
                        temporal_similarity = 1.0 - (frame_diff / max_frame_diff)
                
                # Combine facial and temporal similarity
                # Only use temporal if it can improve similarity
                #potential_combined = (1 - temporal_weight) * distance + temporal_weight * temporal_similarity
                #combined_similarity = max(distance, potential_combined)
                combined_similarity = (1 - temporal_weight) * distance + temporal_weight * temporal_similarity

            else:
                # For faces that aren't similar enough, don't apply temporal boost
                combined_similarity = distance
            
            if combined_similarity > threshold:
                edge_id = compare_idx + 1
                encoding_edges.append((node_id, edge_id, {'weight': combined_similarity}))
                
        edges.extend(encoding_edges)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Iterative clustering with convergence detection
    print(f"Starting adaptive Chinese Whispers iteration (max {iterations} times)...")
    stable_count = 0
    iteration = 0  # Initialize iteration counter for metadata

    for iteration in range(iterations):
        cluster_nodes = list(G.nodes)
        shuffle(cluster_nodes)
        
        # Track changes in this iteration
        changes = 0
        
        for node in cluster_nodes:
            # Store previous cluster assignment
            prev_cluster = G.nodes[node]['cluster']
            
            neighbors = G[node]
            clusters = {}
            
            # Collect clustering information of neighbors
            for ne in neighbors:
                if isinstance(ne, int):
                    if G.nodes[ne]['cluster'] in clusters:
                        weight = G[node][ne]['weight'] * (0.7 + 0.3 * G.nodes[ne]['quality'])
                        clusters[G.nodes[ne]['cluster']] += weight
                    else:
                        weight = G[node][ne]['weight'] * (0.7 + 0.3 * G.nodes[ne]['quality'])
                        clusters[G.nodes[ne]['cluster']] = weight
            
            # Find the cluster with the highest weight sum
            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster
            
            # Update cluster assignment
            G.nodes[node]['cluster'] = max_cluster
            
            # Count changes
            if prev_cluster != max_cluster:
                changes += 1
        
        # Calculate stability ratio
        total_nodes = len(cluster_nodes)
        stability_ratio = 1.0 - (changes / total_nodes) if total_nodes > 0 else 1.0
        
        # Check convergence
        if stability_ratio >= convergence_threshold:
            stable_count += 1
            print(f"  Iteration {iteration + 1}: Stability {stability_ratio:.4f} (stable {stable_count}/{consecutive_stable_iterations})")
            
            if stable_count >= consecutive_stable_iterations:
                print(f"✅ Converged after {iteration + 1} iterations")
                break
        else:
            stable_count = 0
            if (iteration + 1) % 5 == 0:  # Print progress every 5 iterations
                print(f"  Iteration {iteration + 1}: Stability {stability_ratio:.4f}")
    
    # Preparing clustering output
    clusters = {}
    for (_, data) in G.nodes.items():
        cluster = data['cluster']
        path = data['path']
        
        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)
    
    # Sort clusters by size
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    
    # Filter out clusters that are too small (likely noise)
    filtered_clusters = [cluster for cluster in sorted_clusters if len(cluster) >= 3]  # Min 3 faces per cluster
    
    return filtered_clusters

def _post_process_clusters(clusters, facial_encodings, frame_info, merge_threshold=0.75):
    """
    Post-process clusters with stricter parameters to avoid incorrect merging
    
    Args:
        clusters: Initial clusters
        facial_encodings: mapping of face paths to encodings
        frame_info: dictionary mapping image paths to (frame_num, face_idx)
        merge_threshold: similarity threshold for merging (higher than clustering threshold)
        
    Returns:
        Merged clusters
    """
    if len(clusters) <= 1:
        return clusters
    
    print("Post-processing clusters with stricter parameters...")
    
    # Calculate cluster centroids
    centroids = []
    for cluster in clusters:
        cluster_encodings = [facial_encodings[path] for path in cluster]
        centroid = np.mean(cluster_encodings, axis=0)
        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        centroids.append(centroid)
    
    # Check frame overlap between clusters
    frame_ranges = []
    for cluster in clusters:
        frames = [frame_info[path][0] for path in cluster if frame_info[path][0] > 0]
        if frames:
            frame_ranges.append((min(frames), max(frames)))
        else:
            frame_ranges.append((-1, -1))
    
    # Find clusters to merge
    merge_list = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            # Check similarity between centroids
            similarity = np.dot(centroids[i], centroids[j])
            
            # Check frame overlap - stricter criteria
            frame_overlap = False
            if frame_ranges[i][0] > 0 and frame_ranges[j][0] > 0:
                # Check if the frame ranges overlap significantly
                range_i = frame_ranges[i][1] - frame_ranges[i][0]
                range_j = frame_ranges[j][1] - frame_ranges[j][0]
                
                overlap_start = max(frame_ranges[i][0], frame_ranges[j][0])
                overlap_end = min(frame_ranges[i][1], frame_ranges[j][1])
                
                if overlap_end >= overlap_start:
                    overlap_length = overlap_end - overlap_start
                    # Require at least 20% overlap of the smaller range
                    min_range = min(range_i, range_j)
                    if min_range > 0 and overlap_length >= 0.2 * min_range:
                        frame_overlap = True
            
            # Merge if very similar and have significant frame overlap
            if similarity > merge_threshold and frame_overlap:
                # Additional check: Calculate direct face similarity between clusters
                # Sample a few faces from each cluster
                max_samples = 5
                sample_i = clusters[i][:min(max_samples, len(clusters[i]))]
                sample_j = clusters[j][:min(max_samples, len(clusters[j]))]
                
                # Calculate average similarity between samples
                total_sim = 0
                count = 0
                for path_i in sample_i:
                    for path_j in sample_j:
                        sim = np.dot(facial_encodings[path_i], facial_encodings[path_j])
                        total_sim += sim
                        count += 1
                
                avg_similarity = total_sim / max(1, count)
                
                # Only merge if samples are also very similar
                if avg_similarity > merge_threshold:
                    merge_list.append((i, j))
    
    # Perform merging
    if merge_list:
        # Create a graph to represent merge relationships
        G = nx.Graph()
        for i in range(len(clusters)):
            G.add_node(i)
        
        for i, j in merge_list:
            G.add_edge(i, j)
        
        # Find connected components (clusters to merge)
        merged_clusters = []
        for component in nx.connected_components(G):
            merged_cluster = []
            for cluster_idx in component:
                merged_cluster.extend(clusters[cluster_idx])
            merged_clusters.append(merged_cluster)
        
        # Add remaining clusters
        processed = set()
        for component in nx.connected_components(G):
            for idx in component:
                processed.add(idx)
        
        for i in range(len(clusters)):
            if i not in processed:
                merged_clusters.append(clusters[i])
        
        # Sort by size
        merged_clusters = sorted(merged_clusters, key=len, reverse=True)
        return merged_clusters
    
    return clusters

def find_cluster_centers_adjusted(clusters, facial_encodings, method='weighted_average'):
    """
    Find the center of each cluster with improved methods
    
    Args:
        clusters: list of clusters
        facial_encodings: face encoding dictionary
        method: center calculation method, 'best_quality', 'weighted_average', or 'min_distance'
        
    Returns:
        Cluster center list and center image path
    """
    print("Computing adjusted cluster centers...")
    cluster_centers = []
    center_paths = []
    
    for cluster in tqdm(clusters):
        # Get the encoding of all faces in the cluster
        cluster_encodings = np.array([facial_encodings[path] for path in cluster])
        
        if method == 'best_quality':
            # Method 1: Use the highest quality face as center
            quality_scores = [compute_face_quality(path) for path in cluster]
            max_idx = np.argmax(quality_scores)
            center_encoding = cluster_encodings[max_idx]
            
            cluster_centers.append(center_encoding)
            center_paths.append(cluster[max_idx])
            
        elif method == 'weighted_average':
            # Method 2: Weighted average of encodings based on face quality
            quality_scores = np.array([compute_face_quality(path) for path in cluster])
            # Normalize quality scores
            if np.sum(quality_scores) > 0:
                weights = quality_scores / np.sum(quality_scores)
            else:
                weights = np.ones(len(quality_scores)) / len(quality_scores)
            
            # Calculate weighted average
            center = np.zeros_like(cluster_encodings[0])
            for i, encoding in enumerate(cluster_encodings):
                center += weights[i] * encoding
            
            # Normalize the center vector
            center_norm = np.linalg.norm(center)
            if center_norm > 0:
                center = center / center_norm
            
            cluster_centers.append(center)
            
            # Find the face closest to the weighted center
            similarities = np.sum(cluster_encodings * center, axis=1)
            max_idx = np.argmax(similarities)
            center_paths.append(cluster[max_idx])
            
        elif method == 'min_distance':
            # Method 3: Find the sample with the smallest average distance as the center
            min_avg_distance = float('inf')
            center_idx = 0
            center_encoding = None
            
            # Calculate the average distance from each sample to all other samples
            for i, encoding in enumerate(cluster_encodings):
                distances = np.sum((cluster_encodings - encoding) ** 2, axis=1)
                avg_distance = np.mean(distances)
                
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    center_idx = i
                    center_encoding = encoding
            
            cluster_centers.append(center_encoding)
            center_paths.append(cluster[center_idx])
    
    print(f"Completed, total {len(cluster_centers)} centers")
    return cluster_centers, center_paths