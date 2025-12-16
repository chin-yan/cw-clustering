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
import random

def face_distance(face_encodings, face_to_compare):
    """
    Calculate cosine similarity between facial encodings
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    
    # Use cosine similarity (dot product) to calculate similarity
    return np.sum(face_encodings * face_to_compare, axis=1)

def extract_frame_info(image_path):
    """
    Extract frame number and face index from image path
    """
    basename = os.path.basename(image_path)
    # Expected format: frame_XXXXXX_face_Y.jpg
    match = re.match(r'frame_(\d+)_face_(\d+)\.jpg', basename)
    
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    else:
        return -1, -1

def compute_face_quality(face_path):
    """
    Compute comprehensive face quality score.
    Factors: Sharpness, Contrast, Size, and Symmetry (for frontal bias).
    """
    # Load the image
    img = cv2.imread(face_path)
    if img is None:
        return 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Sharpness (Variance of Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian) / 100.0
    sharpness = min(1.0, sharpness)
    
    # 2. Contrast (Histogram spread)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()
    non_zero_bins = np.count_nonzero(hist_norm > 0.0005)
    contrast = non_zero_bins / 256.0
    
    # 3. Symmetry (Prefer frontal faces)
    # Calculate correlation between left and right half (mirrored)
    h, w = gray.shape
    if w > 10: # Ensure valid width
        center_x = w // 2
        left_side = gray[:, :center_x]
        # Get right side and flip it to match left
        right_side = cv2.flip(gray[:, w-center_x:], 1)
        
        # Crop to same size if odd width
        min_w = min(left_side.shape[1], right_side.shape[1])
        left_side = left_side[:, :min_w]
        right_side = right_side[:, :min_w]
        
        # Calculate correlation
        try:
            res = cv2.matchTemplate(left_side, right_side, cv2.TM_CCOEFF_NORMED)
            symmetry = max(0, res[0][0])
        except:
            symmetry = 0.5
    else:
        symmetry = 0.5

    # 4. Size score
    face_area = h * w
    face_size_score = min(1.0, face_area / (160.0 * 160.0))
    
    # Weighted Score
    # Symmetry is high priority to avoid profile faces as centers
    # Sharpness helps avoid blurry/closed-eye faces (closed eyes often have less local contrast/edges)
    quality_score = (0.3 * sharpness) + (0.2 * contrast) + (0.4 * symmetry) + (0.1 * face_size_score)
    
    return quality_score

def cluster_facial_encodings(facial_encodings, threshold=0.55, iterations=100, temporal_weight=0.1):
    """
    Clustering with dynamic iterations
    """
    encoding_list = list(facial_encodings.items())
    if len(encoding_list) <= 1:
        return []
    
    print(f"Using adjusted Chinese Whispers algorithm to cluster {len(encoding_list)} faces...")
    
    frame_info = {}
    quality_scores = {}
    
    print("Extracting frame information and computing quality scores...")
    for path, _ in tqdm(encoding_list):
        frame_num, face_idx = extract_frame_info(path)
        frame_info[path] = (frame_num, face_idx)
        quality_scores[path] = compute_face_quality(path)
    
    sorted_clusters = _chinese_whispers_adjusted(
        encoding_list, frame_info, quality_scores, 
        threshold=threshold, 
        max_iterations=iterations, 
        temporal_weight=temporal_weight,
        patience=20
    )
    
    final_clusters = _post_process_clusters(sorted_clusters, facial_encodings, frame_info, threshold + 0.1)
    
    print(f"Clustering completed, a total of {len(final_clusters)} clusters")
    return final_clusters

def _chinese_whispers_adjusted(encoding_list, frame_info, quality_scores, threshold=0.55, 
                              max_iterations=100, temporal_weight=0.1, patience=20):
    """
    Adjusted Chinese Whispers with Stability Check
    """
    image_paths, encodings = zip(*encoding_list)
    encodings = np.array(encodings)
    
    nodes = []
    edges = []
    max_frame_diff = 3
    
    print("Creating enhanced graph with temporal continuity...")
    for idx, face_encoding_to_check in enumerate(tqdm(encodings)):
        node_id = idx + 1
        node = (node_id, {
            'cluster': idx,
            'path': image_paths[idx],
            'quality': quality_scores[image_paths[idx]]
        })
        nodes.append(node)
        
        if (idx + 1) >= len(encodings):
            break
            
        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        
        curr_frame_num, curr_face_idx = frame_info[image_paths[idx]]
        
        encoding_edges = []
        for i, distance in enumerate(distances):
            compare_idx = idx + i + 1
            compare_path = image_paths[compare_idx]
            compare_frame_num, compare_face_idx = frame_info[compare_path]
            
            if distance >= (threshold * 0.8):
                temporal_similarity = 0
                if curr_frame_num > 0 and compare_frame_num > 0:
                    frame_diff = abs(curr_frame_num - compare_frame_num)
                    if frame_diff <= max_frame_diff:
                        temporal_similarity = 1.0 - (frame_diff / max_frame_diff)
                
                if temporal_similarity > 0:
                    combined_similarity = (1 - temporal_weight) * distance + temporal_weight * temporal_similarity
                else:
                    combined_similarity = distance
            else:
                combined_similarity = distance
                
            if combined_similarity > threshold:
                edge_id = compare_idx + 1
                encoding_edges.append((node_id, edge_id, {'weight': combined_similarity}))
                
        edges.extend(encoding_edges)
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    print(f"Starting CW clustering (Max {max_iterations} iterations, Patience {patience})...")
    
    stable_counter = 0
    
    for i in range(max_iterations):
        cluster_nodes = list(G.nodes)
        shuffle(cluster_nodes)
        
        changed_nodes_count = 0
        
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}
            
            for ne in neighbors:
                if isinstance(ne, int):
                    neighbor_cluster = G.nodes[ne]['cluster']
                    edge_weight = G[node][ne]['weight']
                    quality_factor = 0.5 + 0.5 * G.nodes[ne]['quality']
                    final_weight = edge_weight * quality_factor
                    
                    if neighbor_cluster in clusters:
                        clusters[neighbor_cluster] += final_weight
                    else:
                        clusters[neighbor_cluster] = final_weight
            
            if clusters:
                max_cluster = max(clusters, key=clusters.get)
            else:
                max_cluster = G.nodes[node]['cluster']
            
            if G.nodes[node]['cluster'] != max_cluster:
                G.nodes[node]['cluster'] = max_cluster
                changed_nodes_count += 1
        
        if changed_nodes_count == 0:
            stable_counter += 1
        else:
            stable_counter = 0
            
        if stable_counter >= patience:
            print(f"Converged! Graph stable for {patience} consecutive iterations.")
            break
    
    clusters = {}
    for (_, data) in G.nodes.items():
        cluster_id = data['cluster']
        path = data['path']
        
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(path)
    
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    filtered_clusters = [cluster for cluster in sorted_clusters if len(cluster) >= 3]
    
    return filtered_clusters


def _post_process_clusters(clusters, facial_encodings, frame_info, merge_threshold=0.7):
    """
    Post-process clusters to merge very similar ones
    """
    if len(clusters) <= 1:
        return clusters
    
    print("Post-processing clusters...")
    
    centroids = []
    for cluster in clusters:
        cluster_encodings = [facial_encodings[path] for path in cluster]
        centroid = np.mean(cluster_encodings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        centroids.append(centroid)
    
    frame_ranges = []
    for cluster in clusters:
        frames = [frame_info[path][0] for path in cluster if frame_info[path][0] > 0]
        if frames:
            frame_ranges.append((min(frames), max(frames)))
        else:
            frame_ranges.append((-1, -1))
    
    merge_list = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            similarity = np.dot(centroids[i], centroids[j])
            
            frame_overlap = False
            if frame_ranges[i][0] > 0 and frame_ranges[j][0] > 0:
                range_i = frame_ranges[i][1] - frame_ranges[i][0]
                range_j = frame_ranges[j][1] - frame_ranges[j][0]
                overlap_start = max(frame_ranges[i][0], frame_ranges[j][0])
                overlap_end = min(frame_ranges[i][1], frame_ranges[j][1])
                
                if overlap_end >= overlap_start:
                    overlap_length = overlap_end - overlap_start
                    min_range = min(range_i, range_j)
                    if min_range > 0 and overlap_length >= 0.2 * min_range:
                        frame_overlap = True
            
            if similarity > merge_threshold and frame_overlap:
                max_samples = 5
                sample_i = clusters[i][:min(max_samples, len(clusters[i]))]
                sample_j = clusters[j][:min(max_samples, len(clusters[j]))]

                total_sim = 0
                count = 0
                for path_i in sample_i:
                    for path_j in sample_j:
                        sim = np.dot(facial_encodings[path_i], facial_encodings[path_j])
                        total_sim += sim
                        count += 1
                
                avg_similarity = total_sim / max(1, count)
                
                if avg_similarity > merge_threshold:
                    merge_list.append((i, j))
    
    if merge_list:
        G = nx.Graph()
        for i in range(len(clusters)):
            G.add_node(i)
        for i, j in merge_list:
            G.add_edge(i, j)
        
        merged_clusters = []
        processed = set()
        for component in nx.connected_components(G):
            merged_cluster = []
            for cluster_idx in component:
                merged_cluster.extend(clusters[cluster_idx])
                processed.add(cluster_idx)
            merged_clusters.append(merged_cluster)
        
        for i in range(len(clusters)):
            if i not in processed:
                merged_clusters.append(clusters[i])
        
        merged_clusters = sorted(merged_clusters, key=len, reverse=True)
        return merged_clusters
    
    return clusters

def find_cluster_centers_adjusted(clusters, facial_encodings, method='smart_center'):
    """
    Find cluster centers using smart selection to avoid bad quality faces.
    
    Args:
        method: 'smart_center' is recommended.
    """
    print("Computing adjusted cluster centers (Smart Selection)...")
    cluster_centers = []
    center_paths = []
    
    for cluster in tqdm(clusters):
        cluster_encodings = np.array([facial_encodings[path] for path in cluster])
        
        # 1. Calculate Geometric Centroid (Average Embedding)
        # This represents the "true identity" of the cluster
        centroid = np.mean(cluster_encodings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        cluster_centers.append(centroid)
        
        # 2. Find the best representative face
        # Instead of just picking the closest to centroid (which might be blurry/closed eyes),
        # we score faces based on a balance of "Centrality" and "Visual Quality".
        
        best_score = -float('inf')
        best_path = cluster[0]
        
        for i, path in enumerate(cluster):
            # Similarity to centroid (0 ~ 1)
            # Higher is better (more representative)
            centrality = np.dot(cluster_encodings[i], centroid)
            
            # Visual Quality (0 ~ 1)
            # Higher is better (sharper, more frontal, high contrast)
            quality = compute_face_quality(path)
            
            # Weighted Score
            # Weight quality heavily to ensure we don't pick ugly faces even if they are central
            # 0.4 Centrality + 0.6 Quality
            score = (centrality * 0.4) + (quality * 0.6)
            
            if score > best_score:
                best_score = score
                best_path = path
                
        center_paths.append(best_path)
    
    print(f"Completed, total {len(cluster_centers)} centers")
    return cluster_centers, center_paths