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
import subprocess
import sys
import time
import face_detection
import feature_extraction
import clustering
import visualization
import face_retrieval  
import enhanced_face_preprocessing  
import speaking_face_annotation 
import enhanced_face_retrieval
import enhanced_video_annotation
import robust_temporal_consistency

tf.disable_v2_behavior()

# ============================================================================
# UTILITY FUNCTIONS (defined first to avoid reference errors)
# ============================================================================

def check_ffmpeg_installation():
    """
    Check if FFmpeg is properly installed and accessible
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and accessible")
            return True
        else:
            print("❌ FFmpeg command failed")
            return False
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg command timed out")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found in system PATH")
        print("Please install FFmpeg:")
        print("  Windows: winget install ffmpeg")
        print("  macOS: brew install ffmpeg") 
        print("  Linux: sudo apt install ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Error checking FFmpeg: {e}")
        return False

def check_file_exists_with_details(file_path, description="File"):
    """
    Check if file exists and print detailed information
    
    Args:
        file_path: Path to check
        description: Description of the file for logging
    
    Returns:
        bool: True if file exists, False otherwise
    """
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {description} exists: {file_path} ({size / (1024*1024):.2f} MB)")
        return True
    else:
        print(f"❌ {description} not found: {file_path}")
        return False

def merge_audio_with_video(input_video, silent_video, output_video, verbose=True):
    """
    Merge audio from original video with processed video using FFmpeg
    
    Args:
        input_video: Path to original video file (with audio)
        silent_video: Path to processed video file (without audio)
        output_video: Path to final output video file (with audio)
        verbose: Whether to print detailed progress information
    
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print(f"🎵 Starting audio merge process...")
        print(f"   Input video (audio source): {input_video}")
        print(f"   Silent video (visual source): {silent_video}")
        print(f"   Output video: {output_video}")
    
    # Check if input files exist
    if not os.path.exists(input_video):
        print(f"❌ Original video not found: {input_video}")
        return False
    
    if not os.path.exists(silent_video):
        print(f"❌ Processed video not found: {silent_video}")
        return False
    
    # Check FFmpeg availability
    if not check_ffmpeg_installation():
        return False
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # FFmpeg command with more robust parameters
        cmd = [
            'ffmpeg', '-y',           # Overwrite output file
            '-i', silent_video,       # Input: processed video (visual)
            '-i', input_video,        # Input: original video (audio)
            '-c:v', 'libx264',        # Use H.264 codec for video
            '-c:a', 'aac',            # Use AAC codec for audio
            '-map', '0:v:0',          # Map video from first input
            '-map', '1:a:0?',         # Map audio from second input (optional)
            '-shortest',              # Use shortest stream duration
            '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
            output_video
        ]
        
        if verbose:
            print("🔄 Running FFmpeg command...")
            print(f"Command: {' '.join(cmd)}")
        
        # Run FFmpeg with detailed output capture
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            if verbose:
                print("✅ FFmpeg completed successfully")
            
            # Verify output file was created
            if os.path.exists(output_video):
                file_size = os.path.getsize(output_video)
                if file_size > 0:
                    print(f"✅ Successfully merged audio, output saved to: {output_video}")
                    print(f"   File size: {file_size / (1024*1024):.2f} MB")
                    
                    # Remove temporary silent video file
                    try:
                        if os.path.exists(silent_video):
                            os.remove(silent_video)
                            if verbose:
                                print(f"🗑️  Removed temporary file: {silent_video}")
                    except Exception as e:
                        if verbose:
                            print(f"⚠️  Could not remove temporary file: {e}")
                    
                    return True
                else:
                    print(f"❌ Output file created but is empty: {output_video}")
                    return False
            else:
                print(f"❌ Output file was not created: {output_video}")
                return False
        else:
            print(f"❌ FFmpeg failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg operation timed out (took longer than 5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Audio merging failed with exception: {e}")
        return False

def create_directories(output_dir):
    """Create necessary output directories"""
    dirs = ['faces', 'clusters', 'centers', 'visualization', 'retrieval']
    for dir_name in dirs:
        dir_path = os.path.join(output_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"📁 Created directory: {dir_path}")
    return {name: os.path.join(output_dir, name) for name in dirs}

def extract_frames(video_path, output_dir, interval=30):
    """Extract frames from video at specified intervals"""
    print("🎞️  Capturing frames from video...")
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
    print(f"✅ Captured {len(frames_paths)} frames")
    return frames_paths

def annotate_video_with_audio_preservation(input_video, output_video, centers_data_path, model_dir,
                                         detection_interval=2, similarity_threshold=0.55, 
                                         temporal_weight=0.3, enhanced=True):
    """
    Annotate video with face detection while preserving original audio
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        centers_data_path: Path to cluster centers data file
        model_dir: Directory containing FaceNet model
        detection_interval: Process every N frames for detection
        similarity_threshold: Minimum similarity threshold for face matching
        temporal_weight: Weight for temporal consistency
        enhanced: Whether to use enhanced annotation
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎬 Starting video annotation with audio preservation...")
    print(f"   Enhanced mode: {enhanced}")
    print(f"   Detection interval: {detection_interval}")
    print(f"   Similarity threshold: {similarity_threshold}")
    
    # Ensure output is MP4 format for better compatibility
    if not output_video.lower().endswith('.mp4'):
        output_video = output_video.replace('.avi', '.mp4')
        print(f"   Changed output format to MP4: {output_video}")
    
    # Create temporary video file name (without audio)
    temp_video = os.path.join(os.path.dirname(output_video), 
                         f"temp_silent_{int(time.time())}.avi")
    print(f"   Temporary file: {temp_video}")
    
    try:
        # Step 1: Process video frames and create annotated video without audio
        print("📹 Step 1: Creating annotated video (without audio)...")
        
        if enhanced:
            enhanced_video_annotation.annotate_video_with_enhanced_detection(
                input_video=input_video,
                output_video=temp_video,  # Output to temporary file
                centers_data_path=centers_data_path,
                model_dir=model_dir,
                detection_interval=detection_interval,
                similarity_threshold=similarity_threshold,
                temporal_weight=temporal_weight
            )
        else:
            # Use original annotation method
            enhanced_video_annotation.annotate_video(
                input_video=input_video,
                output_video=temp_video,
                centers_data_path=centers_data_path,
                model_dir=model_dir
            )
        
        # Verify temporary video was created
        if not os.path.exists(temp_video):
            print(f"❌ Temporary video file not created: {temp_video}")
            return False
        
        temp_size = os.path.getsize(temp_video)
        print(f"✅ Temporary video created successfully ({temp_size / (1024*1024):.2f} MB)")
        
        # Step 2: Merge audio from original video
        print("🎵 Step 2: Merging audio from original video...")
        success = merge_audio_with_video(input_video, temp_video, output_video, verbose=True)
        
        if not success:
            print("⚠️  Audio merging failed, saving video without audio...")
            # If audio merging fails, rename temp file to final file
            try:
                if os.path.exists(temp_video):
                    fallback_output = output_video.replace('.mp4', '_no_audio.avi')
                    os.rename(temp_video, fallback_output)
                    print(f"💾 Silent video saved to: {fallback_output}")
                    return False  # Return False since audio preservation failed
            except Exception as e:
                print(f"❌ Failed to save fallback video: {e}")
                return False
        
        return success
        
    except Exception as e:
        print(f"❌ Video processing failed: {e}")
        # Clean up temporary file if it exists
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
                print(f"🗑️  Cleaned up temporary file")
        except:
            pass
        return False

def parse_arguments():
    """Parse command line arguments with audio preservation option"""
    parser = argparse.ArgumentParser(description='Video face clustering system with audio preservation')
    parser.add_argument('--input_video', type=str, 
                        default=r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4",
                        help='input video path')
    parser.add_argument('--output_dir', type=str,
                        default=r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_0831",
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
                        help='Use time consistency to enhance face recognition')
    parser.add_argument('--temporal_window', type=int, default=10,
                        help='Number of historical frames considered for time consistency')
    parser.add_argument('--min_votes', type=int, default=3,
                        help='Minimum number of votes required for temporal consistency')
    parser.add_argument('--preserve_audio', action='store_true', default=True,
                        help='Preserve original audio in output video')
    
    return parser.parse_args()

def main():
    """Main function for complete video face clustering and annotation system with audio preservation"""
    args = parse_arguments()
    
    print("🚀 Starting Video Face Clustering and Annotation System")
    print("=" * 60)
    
    # Check input requirements
    print("📋 Checking input requirements...")
    if not check_file_exists_with_details(args.input_video, "Input video"):
        return
    
    if not check_file_exists_with_details(args.model_dir, "FaceNet model directory"):
        return
    
    # Check FFmpeg if audio preservation is requested
    if args.preserve_audio:
        print("🎵 Audio preservation requested, checking FFmpeg...")
        if not check_ffmpeg_installation():
            print("⚠️  FFmpeg not available. Audio will not be preserved.")
            args.preserve_audio = False
    
    # Create output directories
    print("📁 Creating output directories...")
    dirs = create_directories(args.output_dir)
    
    # Extract frames for clustering
    frames_paths = extract_frames(args.input_video, dirs['faces'], args.frames_interval)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print(f"🔧 Running with method: {args.method}")
            
            print("👤 Step 1: Detect faces from frames...")
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
                
            print("🧠 Step 2: Load the FaceNet model and extract features...")
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
            
            print("🎯 Step 3: Clustering faces...")
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
            print(f"💾 Saving clustering results: {len(clusters)} clusters generated")
            for idx, cluster in enumerate(clusters):
                cluster_dir = os.path.join(dirs['clusters'], f"cluster_{idx}")
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                    
                for face_path in cluster:
                    face_name = os.path.basename(face_path)
                    dst_path = os.path.join(cluster_dir, face_name)
                    shutil.copy2(face_path, dst_path)

            print("🎯 Step 4: Calculate the center of each cluster...")
            if args.method == 'adjusted' or args.method == 'hybrid':
                # Use best quality method for better center selection
                cluster_centers = clustering.find_cluster_centers_adjusted(
                    clusters, facial_encodings, method='min_distance'
                )
            else:
                cluster_centers = clustering.find_cluster_centers_adjusted(
                    clusters, facial_encodings
                )
            
            # Save centers and related data
            centers_data = {
                'clusters': clusters,
                'facial_encodings': facial_encodings,
                'cluster_centers': cluster_centers
            }
            
            centers_data_path = os.path.join(dirs['centers'], 'centers_data.pkl')
            with open(centers_data_path, 'wb') as f:
                pickle.dump(centers_data, f)
            
            print(f"✅ Cluster centers saved to: {centers_data_path}")
            
            if args.visualize:
                print("📊 Step 5: Visualize the results...")
                visualization.visualize_clusters(
                    clusters, facial_encodings, cluster_centers, 
                    dirs['visualization']
                )
    
    # If retrieval mode is enabled, perform face retrieval and video annotation
    if args.do_retrieval:
        print("🔍 Step 6: Performing face retrieval using Annoy...")
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
            
            # Save search results
            retrieval_results_path = os.path.join(dirs['retrieval'], 'enhanced_retrieval_results.pkl')
            with open(retrieval_results_path, 'wb') as f:
                pickle.dump({'by_center': retrieval_results, 'by_frame': frame_results}, f)
        else:
            # Original retrieval
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
            
            # Save search results
            retrieval_results_path = os.path.join(dirs['retrieval'], 'retrieval_results.pkl')
            with open(retrieval_results_path, 'wb') as f:
                pickle.dump(retrieval_results, f)
        
        # Annotate video
        print("🎬 Step 7: Annotate video...")
        if args.enhanced_annotation:
            print("Using enhanced video annotation...")
            
            if args.preserve_audio:
                # Use new audio preservation method
                output_video = os.path.join(args.output_dir, 'enhanced_annotated_with_audio.mp4')
                
                success = annotate_video_with_audio_preservation(
                    input_video=args.input_video,
                    output_video=output_video,
                    centers_data_path=centers_data_path,
                    model_dir=args.model_dir,
                    detection_interval=2,
                    similarity_threshold=args.similarity_threshold,
                    temporal_weight=args.temporal_weight,
                    enhanced=True
                )
                
                if success:
                    print(f"✅ Enhanced video annotation completed with audio preserved!")
                else:
                    print("⚠️ Enhanced video annotation completed but audio preservation failed")
            else:
                # Original method without audio preservation
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
            
            # Try to create speaking face annotation
            try:
                print("🗣️  Creating speaking face annotation...")
                if args.preserve_audio:
                    speaking_output_video = os.path.join(args.output_dir, 'enhanced_speaking_face_with_audio.mp4')
                    
                    # Create temporary speaking face video
                    temp_speaking_video = os.path.join(args.output_dir, 'temp_speaking_face.avi')
                    enhanced_video_annotation.annotate_speaking_face_with_enhanced_detection(
                        input_video=args.input_video,
                        output_video=temp_speaking_video,
                        centers_data_path=centers_data_path,
                        model_dir=args.model_dir,
                        detection_interval=2
                    )
                    
                    # Check if temporary file was created
                    if check_file_exists_with_details(temp_speaking_video, "Temporary speaking face video"):
                        # Merge audio
                        if merge_audio_with_video(args.input_video, temp_speaking_video, speaking_output_video):
                            print(f"✅ Speaking face video with audio created: {speaking_output_video}")
                        else:
                            print("⚠️ Speaking face video created without audio")
                else:
                    speaking_output_video = os.path.join(args.output_dir, 'enhanced_speaking_face_video.avi')
                    enhanced_video_annotation.annotate_speaking_face_with_enhanced_detection(
                        input_video=args.input_video,
                        output_video=speaking_output_video,
                        centers_data_path=centers_data_path,
                        model_dir=args.model_dir,
                        detection_interval=2
                    )
            except Exception as e:
                print(f"❌ Error creating speaking face annotation: {e}")
                print("Make sure you have installed librosa and soundfile for audio processing")
        else:
            # Original video annotation
            if args.preserve_audio:
                output_video = os.path.join(args.output_dir, 'annotated_with_audio.mp4')
                
                success = annotate_video_with_audio_preservation(
                    input_video=args.input_video,
                    output_video=output_video,
                    centers_data_path=centers_data_path,
                    model_dir=args.model_dir,
                    enhanced=False
                )
                
                if success:
                    print(f"✅ Video annotation completed with audio preserved!")
                else:
                    print("⚠️ Video annotation completed but audio preservation failed")
            else:
                output_video = os.path.join(args.output_dir, 'annotated_video.avi')
                enhanced_video_annotation.annotate_video(
                    input_video=args.input_video,
                    output_video=output_video,
                    centers_data_path=centers_data_path,
                    model_dir=args.model_dir
                )
    
    # Apply temporal consistency enhancement if requested
    if args.do_retrieval and args.temporal_consistency:
        print("⏰ Step 8: Apply temporal consistency enhancement...")

        # Get the original annotation result file path
        if args.enhanced_annotation:
            if args.preserve_audio:
                original_video = os.path.join(args.output_dir, 'enhanced_annotated_with_audio.mp4')
            else:
                original_video = os.path.join(args.output_dir, 'enhanced_annotated_video.avi')
            detection_results_path = os.path.join(dirs['retrieval'], 'enhanced_detection_results.pkl')
        else:
            if args.preserve_audio:
                original_video = os.path.join(args.output_dir, 'annotated_with_audio.mp4')
            else:
                original_video = os.path.join(args.output_dir, 'annotated_video.avi')
            detection_results_path = os.path.join(dirs['retrieval'], 'detection_results.pkl')

        # Check if annotation result file exists
        if check_file_exists_with_details(detection_results_path, "Detection results"):
            # Apply temporal consistency enhancement
            if args.preserve_audio:
                temporal_enhanced_video = os.path.join(args.output_dir, 'temporal_enhanced_with_audio.mp4')
                
                # Create temporary enhanced video
                temp_enhanced_video = os.path.join(args.output_dir, 'temp_temporal_enhanced.avi')
                robust_temporal_consistency.enhance_video_temporal_consistency(
                    input_video=args.input_video,
                    annotation_file=detection_results_path,
                    output_video=temp_enhanced_video,
                    centers_data_path=centers_data_path,
                    temporal_window=args.temporal_window,
                    confidence_threshold=args.similarity_threshold,
                    min_votes=args.min_votes
                )
                
                # Check if temporary enhanced video was created
                if check_file_exists_with_details(temp_enhanced_video, "Temporary temporal enhanced video"):
                    # Merge audio
                    if merge_audio_with_video(args.input_video, temp_enhanced_video, temporal_enhanced_video):
                        print(f"✅ Temporal enhanced video with audio created: {temporal_enhanced_video}")
                    else:
                        print("⚠️ Temporal enhanced video created without audio")
            else:
                temporal_enhanced_video = os.path.join(args.output_dir, 'temporal_enhanced_video.avi')
                robust_temporal_consistency.enhance_video_temporal_consistency(
                    input_video=args.input_video,
                    annotation_file=detection_results_path,
                    output_video=temporal_enhanced_video,
                    centers_data_path=centers_data_path,
                    temporal_window=args.temporal_window,
                    confidence_threshold=args.similarity_threshold,
                    min_votes=args.min_votes
                )

            print(f"✅ Temporal consistency enhanced video saved to: {temporal_enhanced_video}")
        else:
            print(f"⚠️ Warning: Annotation result file not found, unable to apply temporal consistency enhancement")
    
    print("🎉 All processing completed successfully!")
    print("=" * 60)
    
    # Enhanced summary of outputs with file verification
    print("\n📁 Final Output Summary:")
    print("-" * 40)
    
    # Check and report cluster data
    centers_data_path = os.path.join(dirs['centers'], 'centers_data.pkl')
    check_file_exists_with_details(centers_data_path, "Cluster data")
    
    # Check visualizations
    if args.visualize:
        viz_files = [f for f in os.listdir(dirs['visualization']) if f.endswith('.png')]
        if viz_files:
            print(f"✅ Visualizations: {dirs['visualization']} ({len(viz_files)} files)")
        else:
            print(f"⚠️  Visualizations directory exists but no files found: {dirs['visualization']}")
    
    # Check video outputs
    if args.do_retrieval:
        video_outputs = []
        
        # Main annotated video
        if args.enhanced_annotation:
            if args.preserve_audio:
                main_video = os.path.join(args.output_dir, 'enhanced_annotated_with_audio.mp4')
                if check_file_exists_with_details(main_video, "Enhanced annotated video (with audio)"):
                    video_outputs.append(("Main annotated video (with audio)", main_video))
            else:
                main_video = os.path.join(args.output_dir, 'enhanced_annotated_video.avi')
                if check_file_exists_with_details(main_video, "Enhanced annotated video"):
                    video_outputs.append(("Main annotated video", main_video))
        else:
            if args.preserve_audio:
                main_video = os.path.join(args.output_dir, 'annotated_with_audio.mp4')
                if check_file_exists_with_details(main_video, "Annotated video (with audio)"):
                    video_outputs.append(("Main annotated video (with audio)", main_video))
            else:
                main_video = os.path.join(args.output_dir, 'annotated_video.avi')
                if check_file_exists_with_details(main_video, "Annotated video"):
                    video_outputs.append(("Main annotated video", main_video))
        
        # Speaking face video
        if args.preserve_audio:
            speaking_video = os.path.join(args.output_dir, 'enhanced_speaking_face_with_audio.mp4')
            if os.path.exists(speaking_video):
                check_file_exists_with_details(speaking_video, "Speaking face video (with audio)")
                video_outputs.append(("Speaking face video (with audio)", speaking_video))
        else:
            speaking_video = os.path.join(args.output_dir, 'enhanced_speaking_face_video.avi')
            if os.path.exists(speaking_video):
                check_file_exists_with_details(speaking_video, "Speaking face video")
                video_outputs.append(("Speaking face video", speaking_video))
        
        # Temporal enhanced video
        if args.temporal_consistency:
            if args.preserve_audio:
                temporal_video = os.path.join(args.output_dir, 'temporal_enhanced_with_audio.mp4')
                if os.path.exists(temporal_video):
                    check_file_exists_with_details(temporal_video, "Temporal enhanced video (with audio)")
                    video_outputs.append(("Temporal enhanced video (with audio)", temporal_video))
            else:
                temporal_video = os.path.join(args.output_dir, 'temporal_enhanced_video.avi')
                if os.path.exists(temporal_video):
                    check_file_exists_with_details(temporal_video, "Temporal enhanced video")
                    video_outputs.append(("Temporal enhanced video", temporal_video))
        
        # Print video output summary
        if video_outputs:
            print("\n🎬 Generated Video Files:")
            for description, path in video_outputs:
                print(f"   - {description}: {path}")
        else:
            print("⚠️  No video outputs were generated successfully")
    
    # Check for any error files or fallback outputs
    error_files = []
    potential_error_files = [
        'enhanced_annotated_video_no_audio.avi',
        'annotated_video_no_audio.avi',
        'enhanced_speaking_face_video_no_audio.avi',
        'temporal_enhanced_video_no_audio.avi'
    ]
    
    for error_file in potential_error_files:
        error_path = os.path.join(args.output_dir, error_file)
        if os.path.exists(error_path):
            error_files.append(error_path)
    
    if error_files:
        print("\n⚠️  Fallback files (without audio) found:")
        for error_file in error_files:
            check_file_exists_with_details(error_file, "Fallback video")
    
    # Troubleshooting guide
    if args.preserve_audio and not any('with_audio' in output[1] for output in video_outputs):
        print("\n🔧 Troubleshooting: Audio preservation failed")
        print("Possible solutions:")
        print("1. Install FFmpeg: winget install ffmpeg (Windows) or brew install ffmpeg (macOS)")
        print("2. Ensure FFmpeg is in your system PATH")
        print("3. Check if input video has audio track")
        print("4. Try running without --preserve_audio flag first")
        print("5. Check the console output above for specific FFmpeg error messages")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)