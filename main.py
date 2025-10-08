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
import cluster_post_processing

tf.disable_v2_behavior()

# ============================================================================
# AUDIO PROCESSING FUNCTIONS
# ============================================================================

def check_ffmpeg_available():
    """
    Check if FFmpeg is available
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def simple_audio_test(video_path):
    """
    Test if video has audio using FFmpeg
    
    Args:
        video_path: Path to video file
        
    Returns:
        bool: True if audio detected, False otherwise
    """
    try:
        temp_audio = f"temp_audio_test_{int(time.time())}.wav"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-t', '1',
            '-vn', '-acodec', 'pcm_s16le', temp_audio
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        has_audio = False
        if os.path.exists(temp_audio):
            size = os.path.getsize(temp_audio)
            has_audio = size > 1000  # At least 1KB for 1 second of audio
            
            # Clean up
            try:
                os.remove(temp_audio)
            except:
                pass
        
        return has_audio
        
    except Exception as e:
        return False

def merge_audio_to_video(video_without_audio, video_with_audio, output_video_with_audio):
    """
    Merge audio from one video to another video
    
    Args:
        video_without_audio: Path to video file without audio
        video_with_audio: Path to video file with audio (audio source)
        output_video_with_audio: Path to output video file with audio
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎵 Merging audio...")
    print(f"   Video source: {os.path.basename(video_without_audio)}")
    print(f"   Audio source: {os.path.basename(video_with_audio)}")
    print(f"   Output: {os.path.basename(output_video_with_audio)}")
    
    # Validate inputs
    if not os.path.exists(video_without_audio):
        print(f"❌ Video file not found: {video_without_audio}")
        return False
    
    if not os.path.exists(video_with_audio):
        print(f"❌ Audio source file not found: {video_with_audio}")
        return False
    
    if not check_ffmpeg_available():
        print("❌ FFmpeg not available")
        return False
    
    # Test if audio source has audio
    if not simple_audio_test(video_with_audio):
        print("❌ Audio source has no audio!")
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_with_audio)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure output file is different from inputs
    if os.path.abspath(output_video_with_audio) == os.path.abspath(video_without_audio):
        print("❌ Output file cannot be the same as input video file")
        return False
    
    try:
        # Use the successful strategy from your manual test
        cmd = [
            'ffmpeg', '-y',
            '-i', video_without_audio,  # Video source
            '-i', video_with_audio,     # Audio source
            '-c:v', 'copy',             # Copy video without re-encoding
            '-c:a', 'aac',              # Encode audio to AAC
            '-map', '0:v:0',            # Map video from first input
            '-map', '1:a:0',            # Map audio from second input
            '-shortest',                # Use shortest duration
            output_video_with_audio
        ]
        
        print(f"🔄 Running FFmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Verify output
            if os.path.exists(output_video_with_audio):
                size = os.path.getsize(output_video_with_audio)
                if size > 1024:  # At least 1KB
                    # Test if output has audio
                    if simple_audio_test(output_video_with_audio):
                        print(f"✅ Audio successfully merged!")
                        print(f"📁 Output: {output_video_with_audio}")
                        print(f"📊 Size: {size / (1024*1024):.2f} MB")
                        return True
                    else:
                        print("❌ Output file has no audio")
                        return False
                else:
                    print("❌ Output file is too small")
                    return False
            else:
                print("❌ Output file not created")
                return False
        else:
            print(f"❌ FFmpeg failed with return code: {result.returncode}")
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                print(f"Error: {error_lines[-1]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg operation timed out")
        return False
    except Exception as e:
        print(f"❌ Audio merging failed: {e}")
        return False

# ============================================================================
# MODIFIED VIDEO ANNOTATION FUNCTIONS
# ============================================================================

def create_annotated_video_mp4(input_video, output_video_mp4, centers_data_path, model_dir,
                              detection_interval=2, similarity_threshold=0.55, 
                              temporal_weight=0.3, preserve_audio=True):
    """
    Create annotated video in MP4 format with optional audio preservation
    
    Args:
        input_video: Path to input video file
        output_video_mp4: Path to output MP4 video file
        centers_data_path: Path to cluster centers data file
        model_dir: Directory containing FaceNet model
        detection_interval: Process every N frames for detection
        similarity_threshold: Minimum similarity threshold for face matching
        temporal_weight: Weight for temporal consistency
        preserve_audio: Whether to preserve original audio
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎬 Creating annotated video...")
    print(f"   Input: {os.path.basename(input_video)}")
    print(f"   Output: {os.path.basename(output_video_mp4)}")
    print(f"   Preserve audio: {preserve_audio}")
    
    if preserve_audio:
        # Step 1: Create temporary video without audio
        temp_silent_video = output_video_mp4.replace('.mp4', f'_temp_silent_{int(time.time())}.avi')
        
        try:
            # Create annotated video without audio
            print("📹 Step 1: Creating annotated video (without audio)...")
            enhanced_video_annotation.annotate_video_with_enhanced_detection(
                input_video=input_video,
                output_video=temp_silent_video,
                centers_data_path=centers_data_path,
                model_dir=model_dir,
                detection_interval=detection_interval,
                similarity_threshold=similarity_threshold,
                temporal_weight=temporal_weight
            )
            
            # Check if temporary video was created
            if not os.path.exists(temp_silent_video):
                print(f"❌ Temporary annotated video not created")
                return False
            
            temp_size = os.path.getsize(temp_silent_video)
            print(f"✅ Temporary video created ({temp_size / (1024*1024):.2f} MB)")
            
            # Step 2: Merge audio
            print("🎵 Step 2: Adding audio from original video...")
            success = merge_audio_to_video(temp_silent_video, input_video, output_video_mp4)
            
            # Clean up temporary file
            try:
                if os.path.exists(temp_silent_video):
                    os.remove(temp_silent_video)
                    print(f"🗑️ Cleaned up temporary file")
            except:
                pass
            
            if not success:
                print("⚠️ Audio merging failed, but annotated video was created")
                # As fallback, convert temp video to MP4 without audio
                try:
                    fallback_cmd = [
                        'ffmpeg', '-y', '-i', temp_silent_video,
                        '-c:v', 'libx264', '-crf', '23',
                        output_video_mp4.replace('.mp4', '_no_audio.mp4')
                    ]
                    subprocess.run(fallback_cmd, capture_output=True, timeout=120)
                except:
                    pass
            
            return success
            
        except Exception as e:
            print(f"❌ Error creating annotated video: {e}")
            # Clean up
            try:
                if os.path.exists(temp_silent_video):
                    os.remove(temp_silent_video)
            except:
                pass
            return False
    
    else:
        # Create video without audio preservation
        print("📹 Creating annotated video without audio preservation...")
        try:
            enhanced_video_annotation.annotate_video_with_enhanced_detection(
                input_video=input_video,
                output_video=output_video_mp4,
                centers_data_path=centers_data_path,
                model_dir=model_dir,
                detection_interval=detection_interval,
                similarity_threshold=similarity_threshold,
                temporal_weight=temporal_weight
            )
            
            if os.path.exists(output_video_mp4):
                print(f"✅ Annotated video created: {output_video_mp4}")
                return True
            else:
                print("❌ Failed to create annotated video")
                return False
                
        except Exception as e:
            print(f"❌ Error creating annotated video: {e}")
            return False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories(output_dir):
    """Create necessary output directories"""
    dirs = ['faces', 'clusters', 'centers', 'visualization', 'retrieval']
    for dir_name in dirs:
        dir_path = os.path.join(output_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return {name: os.path.join(output_dir, name) for name in dirs}

def extract_frames(video_path, output_dir, interval=30):
    """Extract frames from video at specified intervals"""
    print("🎞️ Extracting frames from video...")
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
    print(f"✅ Extracted {len(frames_paths)} frames")
    return frames_paths

def check_file_exists(file_path, description="File"):
    """Check if file exists and print status"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {description}: {file_path} ({size / (1024*1024):.2f} MB)")
        return True
    else:
        print(f"❌ {description} not found: {file_path}")
        return False

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video face clustering system with MP4 output and audio preservation')
    
    # Basic arguments
    parser.add_argument('--input_video', type=str, 
                        default=r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4",
                        help='Input video path')
    parser.add_argument('--output_dir', type=str,
                        default=r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_0831",
                        help='Output directory')
    parser.add_argument('--model_dir', type=str,
                        default=r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\models\20180402-114759",
                        help='FaceNet model directory')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for feature extraction')
    parser.add_argument('--face_size', type=int, default=160, help='Face image size')
    parser.add_argument('--cluster_threshold', type=float, default=0.55, help='Clustering threshold')
    parser.add_argument('--frames_interval', type=int, default=30, help='Frame extraction interval')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, help='Face similarity threshold')
    parser.add_argument('--temporal_weight', type=float, default=0.35, help='Temporal continuity weight')
    
    # Method selection
    parser.add_argument('--method', type=str, default='hybrid', 
                        choices=['original', 'adjusted', 'hybrid'],
                        help='Clustering method: original, adjusted, or hybrid')
    
    # Feature toggles
    parser.add_argument('--visualize', action='store_true', default=True, help='Create visualization')
    parser.add_argument('--do_retrieval', action='store_true', default=True, help='Perform face retrieval')
    parser.add_argument('--enhanced_retrieval', action='store_true', default=True, help='Use enhanced retrieval')
    parser.add_argument('--enhanced_annotation', action='store_true', default=True, help='Use enhanced annotation')
    
    # Audio and output options
    parser.add_argument('--preserve_audio', action='store_true', default=True, help='Preserve original audio')
    parser.add_argument('--detection_interval', type=int, default=2, help='Face detection frame interval')
    
    return parser.parse_args()

def main():
    """Main function for video face clustering and annotation with MP4 output"""
    args = parse_arguments()
    
    print("🚀 Video Face Clustering and Annotation System (MP4 + Audio)")
    print("=" * 70)
    
    # Check input requirements
    print("📋 Checking requirements...")
    if not check_file_exists(args.input_video, "Input video"):
        return
    
    if not check_file_exists(args.model_dir, "FaceNet model directory"):
        return
    
    # Check FFmpeg if audio preservation is requested
    if args.preserve_audio:
        if check_ffmpeg_available():
            print("✅ FFmpeg available for audio processing")
        else:
            print("⚠️ FFmpeg not available, audio will not be preserved")
            args.preserve_audio = False
    
    # Create output directories
    dirs = create_directories(args.output_dir)
    
    # ========================================================================
    # PHASE 1: FACE CLUSTERING
    # ========================================================================
    print("\n" + "="*50)
    print("PHASE 1: FACE CLUSTERING")
    print("="*50)
    
    # Extract frames
    frames_paths = extract_frames(args.input_video, dirs['faces'], args.frames_interval)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print(f"🔧 Using clustering method: {args.method}")
            
            # Step 1: Face detection
            print("\n👤 Step 1: Detecting faces...")
            if args.method in ['adjusted', 'hybrid']:
                face_paths = enhanced_face_preprocessing.detect_faces_adjusted(
                    sess, frames_paths, dirs['faces'], 
                    min_face_size=60, face_size=args.face_size
                )
            else:
                face_paths = face_detection.detect_faces_in_frames(
                    sess, frames_paths, dirs['faces'], 
                    min_face_size=20, face_size=args.face_size
                )
                
            # Step 2: Feature extraction
            print("\n🧠 Step 2: Extracting facial features...")
            model_dir = os.path.expanduser(args.model_dir)
            feature_extraction.load_model(sess, model_dir)
             
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            nrof_images = len(face_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            
            facial_encodings = feature_extraction.compute_facial_encodings(
                sess, images_placeholder, embeddings, phase_train_placeholder,
                args.face_size, embedding_size, nrof_images, nrof_batches,
                emb_array, args.batch_size, face_paths
            )
            
            # Step 3: Clustering
            print("\n🎯 Step 3: Clustering faces...")
            if args.method == 'adjusted':
                clusters = clustering.cluster_facial_encodings(
                    facial_encodings, 
                    threshold=args.cluster_threshold,
                    iterations=30,
                    temporal_weight=args.temporal_weight
                )
            elif args.method == 'hybrid':
                original_clusters = clustering.cluster_facial_encodings(
                    facial_encodings, threshold=args.cluster_threshold
                )
                adjusted_clusters = clustering.cluster_facial_encodings(
                    facial_encodings, 
                    threshold=args.cluster_threshold,
                    iterations=30,
                    temporal_weight=args.temporal_weight
                )
                
                # Choose better result
                original_sizes = [len(c) for c in original_clusters]
                adjusted_sizes = [len(c) for c in adjusted_clusters]
                
                original_std = np.std(original_sizes) / np.mean(original_sizes) if np.mean(original_sizes) > 0 else float('inf')
                adjusted_std = np.std(adjusted_sizes) / np.mean(adjusted_sizes) if np.mean(adjusted_sizes) > 0 else float('inf')
                
                if len(adjusted_clusters) > 0 and (len(original_clusters) == 0 or 
                    (len(adjusted_clusters) >= len(original_clusters) * 0.8 and adjusted_std <= original_std * 1.2)):
                    print(f"Selected adjusted clustering: {len(adjusted_clusters)} clusters")
                    clusters = adjusted_clusters
                else:
                    print(f"Selected original clustering: {len(original_clusters)} clusters")
                    clusters = original_clusters
            else:
                clusters = clustering.cluster_facial_encodings(
                    facial_encodings, threshold=args.cluster_threshold
                )
            
            # Step 4: Post-processing - Aggressive merging for same person across different scenes
            print("\n🔧 Step 4: Post-processing clusters...")
            
            # Apply aggressive small-to-large cluster merging
            processed_clusters, merge_actions = cluster_post_processing.post_process_clusters(
                clusters, facial_encodings,
                min_large_cluster_size=50,  # Large cluster threshold
                small_cluster_percentage=0.05,  # Small clusters = 5% of total faces
                merge_threshold=0.7,  # Much lower base threshold for aggressive merging
                max_merges_per_cluster=10,  # Allow more merges per large cluster
                safety_checks=True
            )
            
            # Update clusters to processed results
            clusters = processed_clusters
            
            print(f"✅ Post-processing completed:")
            print(f"   Merge actions: {len(merge_actions)}")
            for action in merge_actions:
                # All merge actions now use consistent naming: cluster_i (target), cluster_j (source)
                source_cluster = action.get('cluster_j', 'unknown')
                target_cluster = action.get('cluster_i', 'unknown')
                faces_added = action.get('faces_added', 'unknown')
                
                # Get the appropriate score based on action type
                score = action.get('confidence', action.get('similarity', 0))
                action_type = action.get('type', 'merge')
                
                print(f"   {action_type}: Cluster {source_cluster} → Cluster {target_cluster} "
                      f"(+{faces_added} faces, score: {score:.3f})")

            # Step 5: Save clustering results
            print(f"\n💾 Saving {len(clusters)} clusters...")
            for idx, cluster in enumerate(clusters):
                cluster_dir = os.path.join(dirs['clusters'], f"cluster_{idx}")
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                for face_path in cluster:
                    face_name = os.path.basename(face_path)
                    dst_path = os.path.join(cluster_dir, face_name)
                    shutil.copy2(face_path, dst_path)

            # Step 6: Calculate cluster centers
            print("\n🎯 Step 6: Calculating cluster centers...")
            if args.method in ['adjusted', 'hybrid']:
                cluster_centers = clustering.find_cluster_centers_adjusted(
                    clusters, facial_encodings, method='min_distance'
                )
            else:
                cluster_centers = clustering.find_cluster_centers_adjusted(
                    clusters, facial_encodings
                )
            
            # Save centers data
            centers_data = {
                'clusters': clusters,
                'facial_encodings': facial_encodings,
                'cluster_centers': cluster_centers
            }
            
            centers_data_path = os.path.join(dirs['centers'], 'centers_data.pkl')
            with open(centers_data_path, 'wb') as f:
                pickle.dump(centers_data, f)
            
            print(f"✅ Cluster centers saved to: {centers_data_path}")
            
            # Visualization
            if args.visualize:
                print("\n📊 Step 7: Creating visualizations...")
                visualization.visualize_clusters(
                    clusters, facial_encodings, cluster_centers, 
                    dirs['visualization']
                )
    
    # ========================================================================
    # PHASE 2: VIDEO ANNOTATION WITH AUDIO
    # ========================================================================
    print("\n" + "="*50)
    print("PHASE 2: VIDEO ANNOTATION WITH AUDIO")
    print("="*50)
    
    if args.do_retrieval:
        print("\n🔍 Face retrieval and video annotation...")
        
        if args.enhanced_retrieval:
            print("Using enhanced face retrieval...")
            enhanced_face_retrieval.enhanced_face_retrieval(
                video_path=args.input_video,
                centers_data_path=centers_data_path,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                frame_interval=15,
                batch_size=args.batch_size,
                n_trees=15,
                n_results=10,
                similarity_threshold=args.similarity_threshold,
                temporal_weight=args.temporal_weight
            )
        
        # Create final annotated video with audio
        final_output_video = os.path.join(args.output_dir, 'annotated_video_with_audio.mp4')
        
        print(f"\n🎬 Creating final annotated video...")
        success = create_annotated_video_mp4(
            input_video=args.input_video,
            output_video_mp4=final_output_video,
            centers_data_path=centers_data_path,
            model_dir=args.model_dir,
            detection_interval=args.detection_interval,
            similarity_threshold=args.similarity_threshold,
            temporal_weight=args.temporal_weight,
            preserve_audio=args.preserve_audio
        )
        
        if success:
            print(f"🎉 SUCCESS! Final video created: {final_output_video}")
            
            # Verify final output
            if check_file_exists(final_output_video, "Final annotated video"):
                if args.preserve_audio:
                    has_audio = simple_audio_test(final_output_video)
                    if has_audio:
                        print("✅ Final video has audio!")
                    else:
                        print("⚠️ Final video has no audio")
        else:
            print("❌ Failed to create final annotated video")
    
    print("\n🎊 Processing completed!")
    print("=" * 50)
    
    # Final summary
    print("\n📁 Final Output:")
    if args.do_retrieval:
        final_video = os.path.join(args.output_dir, 'annotated_video_with_audio.mp4')
        if os.path.exists(final_video):
            size = os.path.getsize(final_video) / (1024*1024)
            print(f"🎬 Annotated video: {final_video} ({size:.1f} MB)")
            
            if args.preserve_audio and simple_audio_test(final_video):
                print("🎵 ✅ Audio preserved successfully!")
            elif args.preserve_audio:
                print("🎵 ⚠️ Audio not detected in final video")

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