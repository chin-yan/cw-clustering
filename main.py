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
import face_retrieval  # Import the newly created face retrieval module

tf.disable_v2_behavior()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video face clustering system')
    parser.add_argument('--input_video', type=str, 
                        default=r"C:\Users\VIPLAB\Desktop\Yan\Drama_Lee'sFamily\Lee's Family Reunion EP233 preview.mp4",
                        help='input video path')
    parser.add_argument('--output_dir', type=str,
                        default=r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result",
                        help='output directory')
    parser.add_argument('--model_dir', type=str,
                        default=r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759",
                        help='FaceNet model directory')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--face_size', type=int, default=160, help='face size')
    parser.add_argument('--cluster_threshold', type=float, default=0.7, help='cluster threshold')
    parser.add_argument('--frames_interval', type=int, default=30, help='frames interval')
    parser.add_argument('--visualize', action='store_true', default=True, help='visualize')  # Visualization enabled by default
    parser.add_argument('--do_retrieval', action='store_true', default=True, help='perform face retrieval')  # Face retrieval enabled by default
    parser.add_argument('--retrieval_frames_interval', type=int, default=15, help='frames interval for retrieval')
    parser.add_argument('--annoy_trees', type=int, default=10, help='number of trees for Annoy index')
    parser.add_argument('--retrieval_results', type=int, default=10, help='number of retrieval results per query')
    
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
    print(f"Capture {len(frames_paths)} frames")
    return frames_paths

def main():
    args = parse_arguments()
    dirs = create_directories(args.output_dir)
    frames_paths = extract_frames(args.input_video, dirs['faces'], args.frames_interval)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print("Step 1: Use MTCNN to detect faces...")
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
            
            print("Step 3: Clustering using Chinese Whispers algorithm...")
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
            cluster_centers = clustering.find_cluster_centers(clusters, facial_encodings)
            
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
    
    # New: If retrieval mode is enabled, perform face retrieval
    if args.do_retrieval:
        print("Step 6: Perform face retrieval using Annoy...")
        centers_data_path = os.path.join(dirs['centers'], 'centers_data.pkl')
        
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
        
        # Save retrieval results
        retrieval_results_path = os.path.join(dirs['retrieval'], 'retrieval_results.pkl')
        with open(retrieval_results_path, 'wb') as f:
            pickle.dump(retrieval_results, f)
        
        # Annotate video with face identities
        print("Step 7: Annotating video with face identities...")
        import video_annotation
        output_video = os.path.join(args.output_dir, 'annotated_video.avi')
        video_annotation.annotate_video(
            input_video=args.input_video,
            output_video=output_video,
            centers_data_path=centers_data_path,
            model_dir=args.model_dir
        )
    
    print("Done!")

if __name__ == "__main__":
    main()