# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from annoy import AnnoyIndex
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

import face_detection
import feature_extraction

def load_centers_data(centers_data_path):
    """
    Load previously saved cluster center data
    
    Args:
        centers_data_path: Path to the cluster center data
        
    Returns:
        Dictionary containing cluster information
    """
    print("Loading cluster center data...")
    with open(centers_data_path, 'rb') as f:
        centers_data = pickle.load(f)
    
    # Check data integrity
    if 'cluster_centers' not in centers_data:
        raise ValueError("Missing cluster center information in the data")
    
    centers, center_paths = centers_data['cluster_centers']
    print(f"Successfully loaded {len(centers)} cluster centers")
    return centers, center_paths, centers_data

def build_annoy_index(centers, n_trees=10):
    """
    Build Annoy index
    
    Args:
        centers: Feature vectors of cluster centers
        n_trees: Number of trees, increasing improves accuracy but uses more memory
        
    Returns:
        Built Annoy index
    """
    print("Step 1: Building Annoy index...")
    embedding_size = centers[0].shape[0]
    annoy_index = AnnoyIndex(embedding_size, 'euclidean')
    
    # Add cluster centers to the index
    for i, center in enumerate(centers):
        annoy_index.add_item(i, center)
    
    # Build tree structure - K-means method for node partitioning
    print(f"Using {n_trees} trees to build random forest index")
    annoy_index.build(n_trees)
    
    return annoy_index

def extract_frames_and_faces(video_path, output_dir, sess, interval=10):
    """
    Extract frames from video and detect faces
    
    Args:
        video_path: Input video path
        output_dir: Output directory
        sess: TensorFlow session
        interval: Frame interval
        
    Returns:
        List of detected face paths
    """
    print("Extracting frames from video...")
    frames_dir = os.path.join(output_dir, 'retrieval_frames')
    faces_dir = os.path.join(output_dir, 'retrieval_faces')
    
    # Create directories
    for dir_path in [frames_dir, faces_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_path = os.path.join(frames_dir, f"retrieval_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_paths.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frames_paths)} frames")
    
    # Detect faces
    print("Using MTCNN to detect faces...")
    face_paths = face_detection.detect_faces_in_frames(
        sess, frames_paths, faces_dir,
        min_face_size=20, face_size=160
    )
    
    return face_paths

def compute_face_encodings(sess, face_paths, batch_size=100):
    """
    Compute facial feature encodings
    
    Args:
        sess: TensorFlow session
        face_paths: List of face image paths
        batch_size: Batch size
        
    Returns:
        Dictionary of facial feature encodings
    """
    print("Computing facial feature encodings...")
    
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    
    # Calculate embedding vectors
    nrof_images = len(face_paths)
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    
    facial_encodings = feature_extraction.compute_facial_encodings(
        sess, images_placeholder, embeddings, phase_train_placeholder,
        160, embedding_size, nrof_images, nrof_batches,
        emb_array, batch_size, face_paths
    )
    
    return facial_encodings

def search_similar_faces(annoy_index, facial_encodings, center_paths, n_results=5):
    """
    Search for faces similar to cluster centers
    
    Args:
        annoy_index: Annoy index
        facial_encodings: Dictionary of facial feature encodings
        center_paths: Cluster center image paths
        n_results: Number of results to return per query
        
    Returns:
        Dictionary of retrieval results organized by center ID
    """
    print("Steps 3 and 4: Performing similar face search and sorting results...")
    retrieval_results = {}
    
    # Execute search for each encoding
    for path, encoding in tqdm(facial_encodings.items()):
        # Step 3: Use random forest for querying
        nearest_indices, distances = annoy_index.get_nns_by_vector(
            encoding, n_results, include_distances=True
        )
        
        # Step 4: Sort results based on distance
        for idx, dist in zip(nearest_indices, distances):
            if idx not in retrieval_results:
                retrieval_results[idx] = []
            
            retrieval_results[idx].append({
                'path': path,
                'distance': dist
            })
    
    # Sort each center's results by distance
    for idx in retrieval_results:
        retrieval_results[idx] = sorted(retrieval_results[idx], key=lambda x: x['distance'])
    
    return retrieval_results

def visualize_retrieval_results(retrieval_results, center_paths, output_dir, max_results=10):
    """
    Visualize retrieval results
    
    Args:
        retrieval_results: Dictionary of retrieval results
        center_paths: Cluster center image paths
        output_dir: Output directory
        max_results: Maximum number of results to display per center
    """
    print("Visualizing retrieval results...")
    vis_dir = os.path.join(output_dir, 'retrieval_visualization')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Create visualization for each center
    for idx, results in retrieval_results.items():
        if idx >= len(center_paths):
            continue
            
        center_path = center_paths[idx]
        center_name = os.path.basename(center_path)
        
        # Limit result count
        display_results = results[:max_results]
        n_results = len(display_results)
        
        if n_results == 0:
            continue
        
        # Create plot
        fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 4))
        
        # Display center image
        center_img = plt.imread(center_path)
        axes[0].imshow(center_img)
        axes[0].set_title(f'Center {idx}', fontsize=12)
        axes[0].axis('off')
        
        # Display retrieval results
        for i, result in enumerate(display_results):
            try:
                img = plt.imread(result['path'])
                axes[i+1].imshow(img)
                axes[i+1].set_title(f'Distance: {result["distance"]:.4f}', fontsize=10)
                axes[i+1].axis('off')
            except Exception as e:
                print(f"Unable to load image {result['path']}: {e}")
        
        plt.suptitle(f'Retrieval Results for Center {idx}', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save plot
        plt.savefig(os.path.join(vis_dir, f'retrieval_center_{idx}.png'), dpi=200)
        plt.close()

def face_retrieval(video_path, centers_data_path, output_dir, model_dir, frame_interval=10, 
                 batch_size=100, n_trees=10, n_results=10):
    """
    Main function for face retrieval
    
    Args:
        video_path: Input video path
        centers_data_path: Path to cluster center data
        output_dir: Output directory
        model_dir: FaceNet model directory
        frame_interval: Frame extraction interval
        batch_size: Feature extraction batch size
        n_trees: Number of trees for Annoy index
        n_results: Number of results to return per query
    """
    # Load cluster center data
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    # Build Annoy index
    annoy_index = build_annoy_index(centers, n_trees)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Extract frames and detect faces from video
            face_paths = extract_frames_and_faces(video_path, output_dir, sess, frame_interval)
            
            # Load FaceNet model
            print("Loading FaceNet model...")
            model_dir = os.path.expanduser(model_dir)
            feature_extraction.load_model(sess, model_dir)
            
            # Compute facial feature encodings
            facial_encodings = compute_face_encodings(sess, face_paths, batch_size)
            
            # Search for similar faces
            retrieval_results = search_similar_faces(annoy_index, facial_encodings, center_paths, n_results)
            
            # Visualize retrieval results
            visualize_retrieval_results(retrieval_results, center_paths, output_dir, n_results)
    
    print("Face retrieval completed!")
    return retrieval_results

if __name__ == "__main__":
    # Configure parameters
    video_path = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    centers_data_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result\centers\centers_data.pkl"
    output_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result"
    model_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759"
    
    # Execute face retrieval
    retrieval_results = face_retrieval(
        video_path=video_path,
        centers_data_path=centers_data_path,
        output_dir=output_dir,
        model_dir=model_dir,
        frame_interval=15,  # Extract frames every 15 frames, can be adjusted as needed
        batch_size=100,
        n_trees=10,  # Number of trees for Annoy index
        n_results=10  # Number of results to return per query
    )