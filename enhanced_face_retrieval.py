# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pickle
from annoy import AnnoyIndex
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from collections import deque

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

def build_annoy_index(centers, n_trees=15):
    """
    Build Annoy index with more trees for better accuracy
    
    Args:
        centers: Feature vectors of cluster centers
        n_trees: Number of trees, increased for better accuracy
        
    Returns:
        Built Annoy index
    """
    print("Step 1: Building improved Annoy index...")
    embedding_size = centers[0].shape[0]
    
    # Use cosine distance instead of euclidean for better face matching
    annoy_index = AnnoyIndex(embedding_size, 'angular')
    
    # Add cluster centers to the index
    for i, center in enumerate(centers):
        annoy_index.add_item(i, center)
    
    # Build tree structure with more trees
    print(f"Using {n_trees} trees to build random forest index")
    annoy_index.build(n_trees)
    
    return annoy_index

def extract_frames_and_faces(video_path, output_dir, interval=5):
    """
    Extract frames from video and detect faces with smaller interval using InsightFace
    
    Args:
        video_path: Input video path
        output_dir: Output directory
        interval: Frame interval
        
    Returns:
        List of detected face paths and frame paths
    """
    print("Extracting frames from video with smaller interval...")
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
    
    # Use InsightFace for detection
    print("Using InsightFace (SCRFD) to detect faces...")
    face_paths = face_detection.detect_faces_with_insightface(
        frames_paths, faces_dir,
        min_face_size=30, face_size=112
    )
    
    return face_paths, frames_paths

def compute_face_encodings(face_paths, model_handler, batch_size=64):
    """
    Compute facial feature encodings using InsightFace
    
    Args:
        face_paths: List of face image paths
        model_handler: Loaded InsightFace model handler
        batch_size: Batch size (kept for compatibility, though InsightFace processes sequentially in current impl)
        
    Returns:
        Dictionary of facial feature encodings
    """
    print("Computing facial feature encodings...")
    
    embedding_size = 512 # ArcFace standard
    nrof_images = len(face_paths)
    emb_array = np.zeros((nrof_images, embedding_size))
    
    # Use feature_extraction module updated for InsightFace
    facial_encodings = feature_extraction.compute_facial_encodings(
        None, None, None, None, # Legacy TF args are None
        112, embedding_size, nrof_images, 0,
        emb_array, batch_size, face_paths,
        model_handler=model_handler
    )
    
    return facial_encodings

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
    
    # Expected format: retrieval_frame_XXXXXX_face_Y.jpg
    import re
    # Updated pattern to match face_detection output
    match = re.match(r'.*frame_(\d+)_face_(\d+)\.jpg', basename)
    
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    
    return -1, -1

def compute_face_quality(face_path):
    """
    Compute the quality score for a face image
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
    # Adjusted for 112x112 standard input
    face_size_score = min(1.0, face_area / (112.0 * 112.0))
    
    # Combine metrics (adjusted weights)
    quality_score = 0.35 * sharpness + 0.35 * contrast + 0.3 * face_size_score
    
    # Boost scores by 20%
    quality_score = min(1.0, quality_score * 1.2)
    
    return quality_score

def search_similar_faces_with_temporal(annoy_index, facial_encodings, center_paths, 
                                    frame_info_dict=None, n_results=5, 
                                    similarity_threshold=0.5, temporal_weight=0.2):
    """
    Search for faces similar to cluster centers with temporal consistency
    
    Args:
        annoy_index: Annoy index
        facial_encodings: Dictionary of facial feature encodings
        center_paths: Cluster center image paths
        frame_info_dict: Dictionary mapping paths to frame numbers
        n_results: Number of results to return per query
        similarity_threshold: Minimum similarity threshold
        temporal_weight: Weight for temporal consistency
        
    Returns:
        Dictionary of retrieval results organized by center ID and by frame
    """
    print("Steps 3 and 4: Performing similar face search with temporal consistency...")
    
    # Prepare frame information if not provided
    if frame_info_dict is None:
        frame_info_dict = {}
        for path in facial_encodings.keys():
            frame_num, face_idx = extract_frame_info(path)
            frame_info_dict[path] = frame_num
    
    # Dictionary to store retrieval results by center
    retrieval_results = {}
    
    # Dictionary to store retrieval results by frame
    frame_results = {}
    
    # Sliding window for temporal consistency - store recent matches
    recent_matches = {}  # center_id -> deque of recent quality scores
    window_size = 5  # Number of frames to consider for temporal consistency
    
    # Sort paths by frame number for temporal processing
    sorted_paths = sorted(facial_encodings.keys(), 
                         key=lambda p: frame_info_dict[p] if frame_info_dict[p] > 0 else float('inf'))
    
    # Initialize sliding windows for each center
    for i in range(len(center_paths)):
        recent_matches[i] = deque(maxlen=window_size)
    
    # Process faces in temporal order
    for path in tqdm(sorted_paths):
        frame_num = frame_info_dict[path]
        encoding = facial_encodings[path]
        
        # Compute face quality
        quality = compute_face_quality(path)
        
        # Step 3: Use Annoy for approximate nearest neighbor search
        nearest_indices, distances = annoy_index.get_nns_by_vector(
            encoding, n_results * 2, include_distances=True  # Get more candidates
        )
        
        # Convert distances to similarities (Annoy's angular distance)
        # For angular distance, similarity = 1 - distance^2 / 2
        similarities = [max(0, 1 - (d * d / 2)) for d in distances]
        
        # Apply temporal consistency
        adjusted_similarities = []
        for i, (idx, sim) in enumerate(zip(nearest_indices, similarities)):
            # Get recent match quality for this center
            temporal_boost = 0
            if len(recent_matches[idx]) > 0:
                temporal_boost = sum(recent_matches[idx]) / len(recent_matches[idx])
            
            # Combine similarity with temporal boost
            adjusted_sim = (1 - temporal_weight) * sim + temporal_weight * temporal_boost
            adjusted_similarities.append((idx, adjusted_sim, sim))
        
        # Sort by adjusted similarity
        adjusted_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and keep top matches
        filtered_results = [
            (idx, adj_sim, orig_sim) 
            for idx, adj_sim, orig_sim in adjusted_similarities[:n_results] 
            if adj_sim > similarity_threshold
        ]
        
        # Update results by center
        for idx, adj_sim, orig_sim in filtered_results:
            if idx not in retrieval_results:
                retrieval_results[idx] = []
            
            retrieval_results[idx].append({
                'path': path,
                'frame': frame_num,
                'adjusted_similarity': adj_sim,
                'original_similarity': orig_sim,
                'quality': quality
            })
            
            # Update recent matches for this center
            recent_matches[idx].append(quality * orig_sim)  # Weight by both quality and similarity
        
        # Update results by frame
        if frame_num not in frame_results:
            frame_results[frame_num] = []
        
        if filtered_results:  # Only add if we found matches
            best_idx, best_adj_sim, best_orig_sim = filtered_results[0]
            frame_results[frame_num].append({
                'path': path,
                'center_id': best_idx,
                'similarity': best_orig_sim,
                'adjusted_similarity': best_adj_sim,
                'quality': quality
            })
    
    # Sort each center's results by similarity
    for idx in retrieval_results:
        retrieval_results[idx] = sorted(
            retrieval_results[idx], 
            key=lambda x: x['adjusted_similarity'], 
            reverse=True
        )
    
    # Return both by-center and by-frame results
    return retrieval_results, frame_results

def visualize_retrieval_results(retrieval_results, center_paths, output_dir, max_results=10):
    """
    Visualize retrieval results with improved layout and multi-page summaries
    
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
    
    # Create visualization for each center that has results
    active_centers = [idx for idx in retrieval_results if len(retrieval_results[idx]) > 0]
    
    for idx in active_centers:
        if idx >= len(center_paths):
            continue
            
        center_path = center_paths[idx]
        # center_name = os.path.basename(center_path) # Unused
        
        # Limit result count
        display_results = retrieval_results[idx][:max_results]
        n_results = len(display_results)
        
        if n_results == 0:
            continue
        
        # Create plot
        n_cols = 4  # Fixed number of columns for better layout
        n_rows = (n_results + n_cols - 1) // n_cols + 1  # +1 for center image row
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        # Display center image in its own row
        ax_center = plt.subplot2grid((n_rows, n_cols), (0, 1), colspan=2)
        try:
            center_img = plt.imread(center_path)
            ax_center.imshow(center_img)
            ax_center.set_title(f'Cluster Center {idx}', fontsize=14)
            ax_center.axis('off')
        except Exception as e:
            print(f"Unable to load center image {center_path}: {e}")
        
        # Display retrieval results
        for i, result in enumerate(display_results):
            row = 1 + i // n_cols
            col = i % n_cols
            
            ax = plt.subplot2grid((n_rows, n_cols), (row, col))
            
            try:
                img = plt.imread(result['path'])
                ax.imshow(img)
                
                # Show more detailed information
                title_parts = [
                    f"Sim: {result['original_similarity']:.3f}",
                    f"Adj: {result['adjusted_similarity']:.3f}",
                    f"Qual: {result['quality']:.2f}"
                ]
                
                # Add frame number if available
                if 'frame' in result:
                    title_parts.append(f"Frame: {result['frame']}")
                
                ax.set_title('\n'.join(title_parts), fontsize=9)
                ax.axis('off')
            except Exception as e:
                print(f"Unable to load image {result['path']}: {e}")
        
        plt.suptitle(f'Retrieval Results for Center {idx} ({n_results} matches)', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save plot
        plt.savefig(os.path.join(vis_dir, f'retrieval_center_{idx}.png'), dpi=200)
        plt.close()
    
    # Create multi-page summary visualization showing all centers with matches
    if active_centers:
        print("Creating multi-page summary visualization...")
        
        # Configuration for each page
        centers_per_page = 16  # Maximum centers per page (4x4 grid)
        # max_cols = 4  # Maximum columns per page # Unused variable
        
        # Calculate number of pages needed
        total_pages = (len(active_centers) + centers_per_page - 1) // centers_per_page
        
        print(f"Creating {total_pages} summary pages for {len(active_centers)} active centers...")
        
        # Create each page
        for page_num in range(total_pages):
            # Calculate centers for this page
            start_idx = page_num * centers_per_page
            end_idx = min(start_idx + centers_per_page, len(active_centers))
            page_centers = active_centers[start_idx:end_idx]
            
            n_centers = len(page_centers)
            
            # Calculate optimal grid layout for this page
            if n_centers <= 4:
                n_cols = n_centers
                n_rows = 1
            elif n_centers <= 8:
                n_cols = 4
                n_rows = 2
            elif n_centers <= 12:
                n_cols = 4
                n_rows = 3
            else:
                n_cols = 4
                n_rows = 4
            
            # Create figure with reasonable size
            fig_width = min(16, 4 * n_cols)  # 4 inches per column, max 16
            fig_height = min(12, 3.5 * n_rows)  # 3.5 inches per row, max 12
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            
            # Handle different subplot configurations
            if n_centers == 1:
                axes = [axes]
            elif n_rows == 1:
                if n_cols == 1:
                    axes = [axes]
                else:
                    axes = list(axes) if hasattr(axes, '__iter__') else [axes]
            else:
                axes = axes.flatten()
            
            # Display centers for this page
            for i, center_idx in enumerate(page_centers):
                if center_idx >= len(center_paths):
                    continue
                    
                center_path = center_paths[center_idx]
                n_matches = len(retrieval_results[center_idx])
                
                try:
                    center_img = plt.imread(center_path)
                    axes[i].imshow(center_img)
                    axes[i].set_title(f'Center {center_idx}\n{n_matches} matches', fontsize=11, pad=10)
                    axes[i].axis('off')
                    
                    # Add a border to make it more visually appealing
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor('lightgray')
                        spine.set_linewidth(1)
                        
                except Exception as e:
                    print(f"Unable to load center image {center_path}: {e}")
                    # Create a placeholder with text
                    axes[i].text(0.5, 0.5, f'Center {center_idx}\n{n_matches} matches\n(Image Error)', 
                               transform=axes[i].transAxes, ha='center', va='center', 
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(page_centers), len(axes)):
                axes[i].axis('off')
                axes[i].set_visible(False)
            
            # Create page title
            if total_pages == 1:
                page_title = f'Active Centers Summary ({len(active_centers)} centers found matches)'
            else:
                page_title = f'Active Centers Summary - Page {page_num + 1}/{total_pages}\n({len(page_centers)} centers on this page, {len(active_centers)} total)'
            
            plt.suptitle(page_title, fontsize=14, y=0.95)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            
            # Determine filename
            if total_pages == 1:
                filename = 'active_centers_summary.png'
            else:
                filename = f'active_centers_summary_page_{page_num + 1:02d}_of_{total_pages:02d}.png'
            
            # Save with error handling
            try:
                save_path = os.path.join(vis_dir, filename)
                plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Failed to save {filename} with DPI 200: {e}")
                print("Trying with lower DPI...")
                try:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                    print(f"Saved: {filename} (DPI 150)")
                except Exception as e2:
                    print(f"Failed to save {filename} with DPI 150: {e2}")
            
            plt.close()
        
        # Create an index file listing all summary pages
        if total_pages > 1:
            print("Creating summary index...")
            
            index_content = f"""# Active Centers Summary Index

Total active centers: {len(active_centers)}
Total summary pages: {total_pages}
Centers per page: up to {centers_per_page}

## Summary Pages:
"""
            
            for page_num in range(total_pages):
                start_idx = page_num * centers_per_page
                end_idx = min(start_idx + centers_per_page, len(active_centers))
                page_centers = active_centers[start_idx:end_idx]
                
                filename = f'active_centers_summary_page_{page_num + 1:02d}_of_{total_pages:02d}.png'
                index_content += f"\n### Page {page_num + 1}: {filename}\n"
                index_content += f"Centers: {', '.join(map(str, page_centers))}\n"
                index_content += f"Count: {len(page_centers)} centers\n"
            
            # Add statistics
            index_content += f"\n## Statistics:\n"
            index_content += f"- Total matches found: {sum(len(retrieval_results[idx]) for idx in active_centers)}\n"
            index_content += f"- Average matches per center: {sum(len(retrieval_results[idx]) for idx in active_centers) / len(active_centers):.1f}\n"
            
            # Top centers by number of matches
            top_centers = sorted(active_centers, key=lambda x: len(retrieval_results[x]), reverse=True)[:5]
            index_content += f"- Top 5 centers by matches: {', '.join(f'Center {idx} ({len(retrieval_results[idx])} matches)' for idx in top_centers)}\n"
            
            # Save index file
            index_path = os.path.join(vis_dir, 'summary_index.md')
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_content)
            
            print(f"Summary index saved: {index_path}")
    
    print(f"Visualization completed. Results saved in: {vis_dir}")
    
    # Print final summary
    if active_centers:
        total_matches = sum(len(retrieval_results[idx]) for idx in active_centers)
        print(f"Summary: {len(active_centers)} active centers, {total_matches} total matches")

def visualize_frame_results(frame_results, center_paths, output_dir):
    """
    Visualize retrieval results organized by frame
    
    Args:
        frame_results: Dictionary of retrieval results by frame
        center_paths: Cluster center image paths
        output_dir: Output directory
    """
    print("Visualizing frame-based results...")
    vis_dir = os.path.join(output_dir, 'frame_visualization')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Prepare data for timeline visualization
    frames = sorted(frame_results.keys())
    
    if not frames:
        print("No frame results to visualize")
        return
    
    # Count matches per center across frames
    center_matches = {}
    for frame in frames:
        for match in frame_results[frame]:
            center_id = match['center_id']
            if center_id not in center_matches:
                center_matches[center_id] = []
            
            center_matches[center_id].append((frame, match['similarity']))
    
    # Visualize matches over time for each center
    for center_id, matches in center_matches.items():
        if center_id >= len(center_paths) or len(matches) < 3:  # Skip centers with few matches
            continue
        
        # Extract data for plotting
        match_frames, match_similarities = zip(*matches)
        
        plt.figure(figsize=(12, 6))
        
        # Plot matches over time
        plt.scatter(match_frames, match_similarities, alpha=0.7, s=30)
        plt.plot(match_frames, match_similarities, 'b-', alpha=0.3)
        
        # Add center image as an inset
        try:
            center_img = plt.imread(center_paths[center_id])
            inset_ax = plt.axes([0.05, 0.7, 0.2, 0.2])
            inset_ax.imshow(center_img)
            inset_ax.axis('off')
        except Exception as e:
            print(f"Unable to load center image {center_paths[center_id]}: {e}")
        
        plt.title(f'Center {center_id} Appearances Over Time ({len(matches)} matches)')
        plt.xlabel('Frame Number')
        plt.ylabel('Similarity Score')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Save timeline
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'center_{center_id}_timeline.png'), dpi=200)
        plt.close()
    
    # Create overall timeline showing all centers
    plt.figure(figsize=(15, 8))
    
    # Plot each center's appearances in a different color
    # colors = plt.cm.tab20(np.linspace(0, 1, len(center_matches))) # Unused
    
    for i, (center_id, matches) in enumerate(center_matches.items()):
        if len(matches) < 3:  # Skip centers with few matches
            continue
            
        match_frames, match_similarities = zip(*matches)
        plt.scatter(match_frames, [i] * len(match_frames), c=match_similarities, 
                   cmap='viridis', alpha=0.7, s=50, vmin=0, vmax=1)
        
        # Label y-axis with center id
        plt.text(frames[0] - (frames[-1] - frames[0]) * 0.05, i, f'Center {center_id}', 
                va='center', ha='right')
    
    plt.title('Timeline of Character Appearances')
    plt.xlabel('Frame Number')
    plt.yticks([])
    plt.xlim(frames[0] - (frames[-1] - frames[0]) * 0.05, frames[-1] * 1.01)
    plt.colorbar(label='Similarity Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'character_timeline.png'), dpi=200)
    plt.close()

def enhanced_face_retrieval(video_path, centers_data_path, output_dir, model_dir, 
                         frame_interval=5, batch_size=64, n_trees=15, n_results=10,
                         similarity_threshold=0.5, temporal_weight=0.2):
    """
    Main function for enhanced face retrieval with InsightFace
    
    Args:
        video_path: Input video path
        centers_data_path: Path to cluster center data
        output_dir: Output directory
        model_dir: Ignored (InsightFace manages models automatically)
        frame_interval: Frame extraction interval (reduced for better coverage)
        batch_size: Feature extraction batch size (reduced for stability)
        n_trees: Number of trees for Annoy index (increased for accuracy)
        n_results: Number of results to return per query
        similarity_threshold: Minimum similarity threshold (lowered)
        temporal_weight: Weight for temporal consistency
    """
    # Load cluster center data
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    
    # Build Annoy index with more trees
    annoy_index = build_annoy_index(centers, n_trees)
    
    # No TensorFlow session needed for InsightFace
    
    # Extract frames and detect faces from video with enhanced preprocessing (now using InsightFace)
    face_paths, frame_paths = extract_frames_and_faces(
        video_path, output_dir, frame_interval
    )
    
    # Load InsightFace model handler
    print("Loading InsightFace model...")
    model_handler = feature_extraction.load_model(None, None)
    
    # Compute facial feature encodings
    facial_encodings = compute_face_encodings(face_paths, model_handler, batch_size)
    
    # Prepare frame information for temporal consistency
    frame_info_dict = {}
    for path in facial_encodings.keys():
        frame_num, _ = extract_frame_info(path)
        frame_info_dict[path] = frame_num
    
    # Search for similar faces with temporal consistency
    retrieval_results, frame_results = search_similar_faces_with_temporal(
        annoy_index, facial_encodings, center_paths, frame_info_dict,
        n_results, similarity_threshold, temporal_weight
    )
    
    # Save retrieval results
    results_data = {
        'by_center': retrieval_results,
        'by_frame': frame_results
    }
    results_dir = os.path.join(output_dir, 'retrieval')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    results_path = os.path.join(results_dir, 'enhanced_retrieval_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    # Visualize retrieval results
    visualize_retrieval_results(retrieval_results, center_paths, output_dir, n_results)
    visualize_frame_results(frame_results, center_paths, output_dir)
    
    print("Enhanced face retrieval completed!")
    return retrieval_results, frame_results

if __name__ == "__main__":
    # Configure parameters
    # Note: These paths are placeholders, update as needed
    video_path = r"path/to/video.mp4"
    centers_data_path = r"path/to/centers_data.pkl"
    output_dir = r"path/to/output"
    
    # Execute enhanced face retrieval
    # retrieval_results, frame_results = enhanced_face_retrieval(...)
    pass