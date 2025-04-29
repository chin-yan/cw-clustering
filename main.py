# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import shutil

import face_detection
import feature_extraction
import clustering
import visualization
import face_retrieval  
import clustering  
import enhanced_face_preprocessing  
import speaking_face_annotation 
import enhanced_face_retrieval
import enhanced_video_annotation
import robust_temporal_consistency

tf.disable_v2_behavior()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Adjusted video face clustering system')
    parser.add_argument('--input_video', type=str, 
                        default=r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4",
                        help='input video path')
    parser.add_argument('--output_dir', type=str,
                        default=r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result",
                        help='output directory')
    parser.add_argument('--model_dir', type=str,
                        default=r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759",
                        help='FaceNet model directory')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--face_size', type=int, default=160, help='face size')
    parser.add_argument('--cluster_threshold', type=float, default=0.55, help='cluster threshold (adjusted)')
    parser.add_argument('--frames_interval', type=int, default=30, help='frames interval')
    parser.add_argument('--visualize', action='store_true', default=True, help='visualize')
    parser.add_argument('--do_retrieval', action='store_true', default=True, help='perform face retrieval')
    parser.add_argument('--retrieval_frames_interval', type=int, default=15, help='frames interval for retrieval')
    parser.add_argument('--annoy_trees', type=int, default=15, help='number of trees for Annoy index')
    parser.add_argument('--retrieval_results', type=int, default=10, help='number of retrieval results per query')
    parser.add_argument('--method', type=str, default='hybrid', 
                        choices=['original', 'adjusted', 'hybrid'],
                        help='Method to use: original (old method), adjusted (new method), or hybrid (mix)')
    parser.add_argument('--temporal_weight', type=float, default=0.25,
                        help='weight for temporal continuity in clustering (0-1)')
    parser.add_argument('--enhanced_retrieval', action='store_true', default=True, 
                        help='Use enhanced face retrieval')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, 
                        help='face similarity threshold')
    parser.add_argument('--enhanced_annotation', action='store_true', default=True, 
                        help='Use enhanced video annotation')
    parser.add_argument('--temporal_consistency', action='store_true', default=True,
                        help = 'Use time consistency to enhance face recognition')
    parser.add_argument('--temporal_window', type=int, default=10,
                        help='Number of historical frames considered for time consistency')
    parser.add_argument('--min_votes', type=int, default=3,
                        help='Minimum number of votes required for temporal consistency')
    
    return parser.parse_args()

def create_directories(output_dir):
    dirs = ['faces', 'clusters', 'centers', 'visualization', 'retrieval']
    for dir_name in dirs:
        dir_path = os.path.join(output_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return {name: os.path.join(output_dir, name) for name in dirs}

def extract_frames(video_path, output_dir, interval=30):
    print("Capturing frames from video...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_paths.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    print(f"Captured {len(frames_paths)} frames")
    return frames_paths

def main():
    args = parse_arguments()
    dirs = create_directories(args.output_dir)
    frames_paths = extract_frames(args.input_video, dirs['faces'], args.frames_interval)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print(f"Running with method: {args.method}")
            
            print("Step 1: Detect faces from frames...")
            if args.method == 'adjusted' or args.method == 'hybrid':
                print("Using adjusted face detection and preprocessing...")
                face_paths = enhanced_face_preprocessing.detect_faces_adjusted(
                    sess, frames_paths, dirs['faces'], 
                    min_face_size=20, face_size=args.face_size
                )
            else:
                # Original method
                face_paths = face_detection.detect_faces_in_frames(
                    sess, frames_paths, dirs['faces'], 
                    min_face_size=20, face_size=args.face_size
                )
                
            print("Step 2: Load the FaceNet model and extract features...")
            model_dir = os.path.expanduser(args.model_dir)
            feature_extraction.load_model(sess, model_dir)
             
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Calculate the embedding vector
            nrof_images = len(face_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            
            facial_encodings = feature_extraction.compute_facial_encodings(
                sess, images_placeholder, embeddings, phase_train_placeholder,
                args.face_size, embedding_size, nrof_images, nrof_batches,
                emb_array, args.batch_size, face_paths
            )
            
            print("Step 3: Clustering faces...")
            if args.method == 'adjusted':
                print("Using adjusted clustering algorithm...")
                clusters = clustering.cluster_facial_encodings(
                    facial_encodings, 
                    threshold=args.cluster_threshold,
                    iterations=25,
                    temporal_weight=args.temporal_weight
                )
            elif args.method == 'hybrid':
                # Hybrid method: use both algorithms and take the better result
                print("Using hybrid clustering approach (running both methods)...")
                
                # Run original clustering
                original_clusters = clustering.cluster_facial_encodings(
                    facial_encodings, threshold=args.cluster_threshold
                )
                
                # Run adjusted clustering
                adjusted_clusters = clustering.cluster_facial_encodings(
                    facial_encodings, 
                    threshold=args.cluster_threshold,
                    iterations=25,
                    temporal_weight=args.temporal_weight
                )
                
                # Compare results and choose the better one
                # Criteria: more balanced cluster sizes (avoid one giant cluster)
                original_sizes = [len(c) for c in original_clusters]
                adjusted_sizes = [len(c) for c in adjusted_clusters]
                
                original_std = np.std(original_sizes) / np.mean(original_sizes) if np.mean(original_sizes) > 0 else float('inf')
                adjusted_std = np.std(adjusted_sizes) / np.mean(adjusted_sizes) if np.mean(adjusted_sizes) > 0 else float('inf')
                
                if len(adjusted_clusters) > 0 and (len(original_clusters) == 0 or 
                                                  (len(adjusted_clusters) >= len(original_clusters) * 0.8 and
                                                  adjusted_std <= original_std * 1.2)):
                    print(f"Selected adjusted clustering method: {len(adjusted_clusters)} clusters vs {len(original_clusters)} original")
                    clusters = adjusted_clusters
                else:
                    print(f"Selected original clustering method: {len(original_clusters)} clusters vs {len(adjusted_clusters)} adjusted")
                    clusters = original_clusters
            else:
                # Original method
                clusters = clustering.cluster_facial_encodings(
                    facial_encodings, threshold=args.cluster_threshold
                )
            
            # Save clustering results
            print(f"A total of {len(clusters)} clusters were generated")
            for idx, cluster in enumerate(clusters):
                cluster_dir = os.path.join(dirs['clusters'], f"cluster_{idx}")
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                    
                for face_path in cluster:
                    face_name = os.path.basename(face_path)
                    dst_path = os.path.join(cluster_dir, face_name)
                    shutil.copy2(face_path, dst_path)

            print("Step 4: Calculate the center of each cluster...")
            if args.method == 'adjusted' or args.method == 'hybrid':
                # Use best quality method for better center selection
                cluster_centers = clustering.find_cluster_centers_adjusted(
                    clusters, facial_encodings, method='min_distance'
                )
            else:
                cluster_centers = clustering.find_cluster_centers_adjusted(
                    clusters, facial_encodings
                )
            
            # Preservation center and related data
            centers_data = {
                'clusters': clusters,
                'facial_encodings': facial_encodings,
                'cluster_centers': cluster_centers
            }
            
            centers_data_path = os.path.join(dirs['centers'], 'centers_data.pkl')
            with open(centers_data_path, 'wb') as f:
                pickle.dump(centers_data, f)
            
            if args.visualize:
                print("Step 5: Visualize the results...")
                visualization.visualize_clusters(
                    clusters, facial_encodings, cluster_centers, 
                    dirs['visualization']
                )
    
    # If retrieval mode is enabled, perform face retrieval and video annotation
    if args.do_retrieval:
        print("Step 6: Performing face retrieval using Annoy...")
        centers_data_path = os.path.join(dirs['centers'], 'centers_data.pkl')
        
        if args.enhanced_retrieval:
            print("Using enhanced face retrieval...")
            retrieval_results, frame_results = enhanced_face_retrieval.enhanced_face_retrieval(
                video_path=args.input_video,
                centers_data_path=centers_data_path,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                frame_interval=args.retrieval_frames_interval,
                batch_size=args.batch_size,
                n_trees=args.annoy_trees,
                n_results=args.retrieval_results,
                similarity_threshold=args.similarity_threshold,
                temporal_weight=args.temporal_weight
            )
            
            # Save Search Results
            retrieval_results_path = os.path.join(dirs['retrieval'], 'enhanced_retrieval_results.pkl')
            with open(retrieval_results_path, 'wb') as f:
                pickle.dump({'by_center': retrieval_results, 'by_frame': frame_results}, f)
        else:
            # original results
            retrieval_results = face_retrieval.face_retrieval(
                video_path=args.input_video,
                centers_data_path=centers_data_path,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                frame_interval=args.retrieval_frames_interval,
                batch_size=args.batch_size,
                n_trees=args.annoy_trees,
                n_results=args.retrieval_results
            )
            
            # Save Search Results
            retrieval_results_path = os.path.join(dirs['retrieval'], 'retrieval_results.pkl')
            with open(retrieval_results_path, 'wb') as f:
                pickle.dump(retrieval_results, f)
        
        # Annotate Video
        print("Step 7: Annotate Video...")
        if args.enhanced_annotation:
            print("Using Enhanced Video Annotation...")
            output_video = os.path.join(args.output_dir, 'enhanced_annotated_video.avi')
            enhanced_video_annotation.annotate_video_with_enhanced_detection(
                input_video=args.input_video,
                output_video=output_video,
                centers_data_path=centers_data_path,
                model_dir=args.model_dir,
                detection_interval=2,
                similarity_threshold=args.similarity_threshold,
                temporal_weight=args.temporal_weight
            )
            
            # Try to label speaking faces
            try:
                speaking_output_video = os.path.join(args.output_dir, 'enhanced_speaking_face_video.avi')
                enhanced_video_annotation.annotate_speaking_face_with_enhanced_detection(
                    input_video=args.input_video,
                    output_video=speaking_output_video,
                    centers_data_path=centers_data_path,
                    model_dir=args.model_dir,
                    detection_interval=2
                )
            except Exception as e:
                print(f"An error occurred while labeling the speaking face: {e}")
                print("Make sure you have installed librosa and soundfile for audio processing")
        else:
            # Original video annotation
            output_video = os.path.join(args.output_dir, 'annotated_video.avi')
            enhanced_video_annotation.annotate_video(
                input_video=args.input_video,
                output_video=output_video,
                centers_data_path=centers_data_path,
                model_dir=args.model_dir
            )
    if args.do_retrieval and args.temporal_consistency:
        print("Step 8: Apply time consistency enhancement...")

        # Get the original annotation result file path
        if args.enhanced_annotation:
            original_video = os.path.join(args.output_dir, 'enhanced_annotated_video.avi')
            detection_results_path = os.path.join(dirs['retrieval'], 'enhanced_detection_results.pkl')
        else:
            original_video = os.path.join(args.output_dir, 'annotated_video.avi')
            detection_results_path = os.path.join(dirs['retrieval'], 'detection_results.pkl')

        # Confirm that the annotation result file exists
        if os.path.exists(detection_results_path):
            # Apply time consistency enhancement
            temporal_enhanced_video = os.path.join(args.output_dir, 'temporal_enhanced_video.avi')

            # Calling time consistency enhancement function
            robust_temporal_consistency.enhance_video_temporal_consistency(
            input_video=args.input_video,
            annotation_file=detection_results_path,
            output_video=temporal_enhanced_video,
            centers_data_path=centers_data_path,
            temporal_window=args.temporal_window,
            confidence_threshold=args.similarity_threshold,
            min_votes = args.min_votes
            )

            print(f"The temporal consistency enhanced video has been saved to: {temporal_enhanced_video}")
        else:
            print(f"Warning: Labeled result file {detection_results_path} not found, unable to apply temporal consistency enhancement")
            
    print("Done!")

if __name__ == "__main__":
    main()