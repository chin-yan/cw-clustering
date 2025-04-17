# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from random import shuffle
from tqdm import tqdm

def face_distance(face_encodings, face_to_compare):
    """
        Calculate Euclidean distance/similarity between facial encodings

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

def cluster_facial_encodings(facial_encodings, threshold=0.7, iterations=20):
    """
        Clustering face encoding using Chinese Whispers algorithm

        Args:
            facial_encodings: mapping of face paths to encodings
            threshold: face matching threshold, default is 0.7
            iterations: number of iterations

        Returns:
            Sorted list of clusters
    """
    # Prepare data
    encoding_list = list(facial_encodings.items())
    if len(encoding_list) <= 1:
        print("Insufficient number of encodings to cluster")
        return []
    
    print(f"Use the Chinese Whispers algorithm to cluster {len(encoding_list)} faces...")
    sorted_clusters = _chinese_whispers(encoding_list, threshold, iterations)
    print(f"Clustering completed, a total of {len(sorted_clusters)} clusters")
    
    return sorted_clusters

def _chinese_whispers(encoding_list, threshold=0.7, iterations=20):
    """
    Implementation of Chinese Whispers Clustering Algorithm

    Args:
        encoding_list: list of (image path, face encoding) tuples
        threshold: face matching threshold
        iterations: number of iterations

    Returns:
        List of clusters sorted by size
    """
    # Prepare data
    image_paths, encodings = zip(*encoding_list)
    encodings = np.array(encodings)
    
    # Create graph
    nodes = []
    edges = []
    
    print("Creating graph...")
    for idx, face_encoding_to_check in enumerate(tqdm(encodings)):
        # Add nodes
        node_id = idx + 1
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)
        
        # If it is the last element, no edge is created
        if (idx + 1) >= len(encodings):
            break
            
        # Calculate distance to other encodings
        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        
        # Add edges
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))
                
        edges.extend(encoding_edges)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Iterative clustering
    print(f"Starting Chinese Whispers iteration ({iterations} times)...")
    for _ in tqdm(range(iterations)):
        cluster_nodes = list(G.nodes)
        shuffle(cluster_nodes)
        
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}
            
            # Collect clustering information of neighbors
            for ne in neighbors:
                if isinstance(ne, int):
                    if G.nodes[ne]['cluster'] in clusters:
                        clusters[G.nodes[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.nodes[ne]['cluster']] = G[node][ne]['weight']
            
            # Find the cluster with the highest weight sum
            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster
            
            # Set the clustering of the target node
            G.nodes[node]['cluster'] = max_cluster
    
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
    return sorted_clusters

def find_cluster_centers(clusters, facial_encodings, method='average'):
    """
        Find the center of each cluster

        Args:
            clusters: list of clusters
            facial_encodings: face encoding dictionary
            method: center calculation method, 'average' or 'min_distance'

        Returns:
            Cluster center list and center image path
    """
    print("Compute cluster centers...")
    cluster_centers = []
    center_paths = []
    
    for cluster in tqdm(clusters):
        # Get the encoding of all faces in the cluster
        cluster_encodings = np.array([facial_encodings[path] for path in cluster])
        
        if method == 'average':
            # Method 1: Use the mean as the center
            center = np.mean(cluster_encodings, axis=0)
            cluster_centers.append(center)
            
            # Find the face closest to the center
            distances = np.sum((cluster_encodings - center) ** 2, axis=1)
            min_idx = np.argmin(distances)
            center_paths.append(cluster[min_idx])
            
        elif method == 'min_distance':
            # Method 2: Find the sample with the smallest average distance as the center
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