# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import networkx as nx
from tqdm import tqdm

def visualize_clusters(clusters, facial_encodings, cluster_centers, output_dir):
    """
        Visualizing clustering results

        Args:
            clusters: list of clusters
            facial_encodings: face encoding dictionary
            cluster_centers: list of cluster centers (tuple of centers and center_paths)
            output_dir: output directory
    """
    print("Creating cluster overview graph...")
    create_cluster_overview(clusters, output_dir)
    
    print("Creating cluster network graph...")
    create_cluster_network(clusters, facial_encodings, cluster_centers, output_dir)
    
    print("Creating thumbnail sets for each cluster...")
    create_cluster_thumbnails(clusters, output_dir)
    
    print("Creating cluster center view...")
    if isinstance(cluster_centers, tuple):
        centers, center_paths = cluster_centers
        create_cluster_centers_view(centers, center_paths, output_dir)
    else:
        print("Warning: Cluster center format is incorrect, skipping center view creation")
        
def create_cluster_centers_view(centers, center_paths, output_dir):
    n_centers = len(center_paths)
    if n_centers == 0:
        return
        
    # Counting the number of rows and columns
    n_cols = min(5, n_centers)
    n_rows = (n_centers + n_cols - 1) // n_cols
    
    # Create a chart
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Show center image
    for i, path in enumerate(center_paths):
        try:
            img = mpimg.imread(path)
            axes[i].imshow(img)
            axes[i].set_title(f'Cluster {i} Center', fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            print(f"Unable to load image {path}: {e}")
    
    # Hide redundant sub-images
    for i in range(n_centers, len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)
        
    plt.suptitle('All cluster centers', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the chart
    plt.savefig(os.path.join(output_dir, 'cluster_centers.png'), dpi=200)
    plt.close()

def create_cluster_overview(clusters, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Clusters sorted by size
    sizes = [len(cluster) for cluster in clusters]
    plt.bar(range(len(sizes)), sizes)
    plt.title('Cluster size distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of face')
    plt.grid(True, alpha=0.3)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'), dpi=300)
    plt.close()

def create_cluster_network(clusters, facial_encodings, cluster_centers, output_dir):
    # Create a network graph for each cluster
    for idx, cluster in enumerate(clusters):
        if len(cluster) > 30:  # For large clusters, only 30 samples are taken
            sample_cluster = np.random.choice(cluster, 30, replace=False)
        else:
            sample_cluster = cluster
            
        if len(sample_cluster) < 3:  # Clusters that are too small are skipped
            continue
            
        print(f"Create a network graph of cluster {idx}...")
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, path in enumerate(sample_cluster):
            G.add_node(i, path=path)
            
        # Add Edges - Connect all nodes
        for i in range(len(sample_cluster)):
            for j in range(i+1, len(sample_cluster)):
                # Calculate similarity as edge weight
                similarity = np.dot(
                    facial_encodings[sample_cluster[i]], 
                    facial_encodings[sample_cluster[j]]
                )
                G.add_edge(i, j, weight=similarity)
        
        # Draw a network diagram
        plt.figure(figsize=(12, 12))
        
        # Using spring_layout
        pos = nx.spring_layout(G)
        
        # Draw edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, alpha=0.3, 
            width=[w*3 for w in edge_weights],
            edge_color='grey'
        )
        
        #Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=300, 
            node_color='skyblue',
            alpha=0.8
        )
        
        # Add face thumbnail
        for node, (x, y) in pos.items():
            path = G.nodes[node]['path']
            try:
                img = mpimg.imread(path)
                imagebox = OffsetImage(img, zoom=0.1)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                plt.gca().add_artist(ab)
            except Exception as e:
                print(f"Unable to load image {path}: {e}")
        
        plt.title(f'Cluster {idx} network graph ({len(sample_cluster)} faces)')
        plt.axis('off')
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cluster_{idx}_network.png'), dpi=300)
        plt.close()

def create_cluster_thumbnails(clusters, output_dir):
    max_clusters = min(len(clusters), 20)  # Only show the first 20 clusters
    
    for idx, cluster in enumerate(clusters[:max_clusters]):
        # Limit the number of images displayed to 25 per cluster
        sample_size = min(len(cluster), 25)
        if sample_size == 0:
            continue
            
        # If the cluster is large, random sampling
        if len(cluster) > sample_size:
            sample_paths = np.random.choice(cluster, sample_size, replace=False)
        else:
            sample_paths = cluster
        
        # Counting the number of rows and columns
        n_cols = min(5, sample_size)
        n_rows = (sample_size + n_cols - 1) // n_cols
        
        # Create graph
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Show image
        for i, path in enumerate(sample_paths):
            try:
                img = mpimg.imread(path)
                axes[i].imshow(img)
                axes[i].set_title(os.path.basename(path), fontsize=8)
                axes[i].axis('off')
            except Exception as e:
                print(f"Unable to load image {path}: {e}")
        
        # Hide redundant sub-images
        for i in range(sample_size, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
            
        plt.suptitle(f'Cluster {idx} thumbnails ({len(cluster)} faces)', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, f'cluster_{idx}_thumbnails.png'), dpi=200)
        plt.close()