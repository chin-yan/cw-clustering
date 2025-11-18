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
import enhanced_face_preprocessing  
import speaking_face_annotation 
import enhanced_face_retrieval
import enhanced_video_annotation
import cluster_post_processing
import speaker_subtitle_annotation

tf.disable_v2_behavior()

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
    print("Extracting frames from video...")
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
    print(f"Extracted {len(frames_paths)} frames")
    return frames_paths

def check_file_exists(file_path, description="File"):
    """Check if file exists and print status"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"[OK] {description}: {file_path} ({size / (1024*1024):.2f} MB)")
        return True
    else:
        print(f"[ERROR] {description} not found: {file_path}")
        return False

def check_ffmpeg_available():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video face clustering system with speaker-aware subtitle annotation')
    
    # Basic arguments
    parser.add_argument('--input_video', type=str, required=True,
                        help='Input video path')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='FaceNet model directory')
    
    # Subtitle annotation arguments
    parser.add_argument('--srt', type=str, default=None,
                        help='Path to subtitle file (SRT format). If provided, will create speaker-aware subtitle annotation')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for feature extraction')
    parser.add_argument('--face_size', type=int, default=160, help='Face image size')
    parser.add_argument('--cluster_threshold', type=float, default=0.55, help='Clustering threshold')
    parser.add_argument('--frames_interval', type=int, default=30, help='Frame extraction interval')
    parser.add_argument('--similarity_threshold', type=float, default=0.65, help='Face similarity threshold for matching')
    parser.add_argument('--temporal_weight', type=float, default=0.25, help='Temporal continuity weight')
    parser.add_argument('--speaking_threshold', type=float, default=0.7, 
                        help='Speaking detection threshold (0.5-1.5, higher = stricter)')
    
    # Method selection
    parser.add_argument('--method', type=str, default='hybrid', 
                        choices=['original', 'adjusted', 'hybrid'],
                        help='Clustering method: original, adjusted, or hybrid')
    
    # Feature toggles
    parser.add_argument('--visualize', action='store_true', default=True, help='Create visualization')
    parser.add_argument('--do_retrieval', action='store_true', default=True, help='Perform face retrieval')
    parser.add_argument('--enhanced_retrieval', action='store_true', default=True, help='Use enhanced retrieval')
    
    # Audio and output options
    parser.add_argument('--preserve_audio', action='store_true', default=True, help='Preserve original audio')
    parser.add_argument('--detection_interval', type=int, default=2, help='Face detection frame interval for annotation')
    parser.add_argument('--smoothing_alpha', type=float, default=0.3,
                        help='BBox smoothing factor for subtitle annotation (0-1, lower=more smooth)')
    parser.add_argument('--generate_json', action='store_true', default=True,
                        help='Generate JSON annotation file alongside video (default: True)')
    
    # Subtitle synchronization arguments (NEW)
    parser.add_argument('--subtitle_offset', type=float, default=0.0,
                        help='Manual subtitle time offset in seconds (positive = delay subtitles, negative = advance subtitles)')
    parser.add_argument('--auto_align', action='store_true', default=False,
                        help='Automatically calculate optimal subtitle-audio alignment offset before annotation')
    parser.add_argument('--vad_mode', type=str, default='webrtc',
                        choices=['energy', 'webrtc', 'silero'],
                        help='Voice Activity Detection method for auto-alignment (energy=simple, webrtc=balanced, silero=advanced)')
    parser.add_argument('--vad_aggressiveness', type=int, default=2,
                        choices=[0, 1, 2, 3],
                        help='WebRTC VAD aggressiveness level: 0=least aggressive, 3=most aggressive (only used if vad_mode=webrtc)')
    
    # Legacy annotation option (if no subtitle file provided)
    parser.add_argument('--use_legacy_annotation', action='store_true', default=False,
                        help='Use legacy annotation style without subtitles (only if subtitle_file not provided)')
    
    return parser.parse_args()

def main():
    """Main function for video face clustering and annotation with speaker-aware subtitles"""
    args = parse_arguments()
    
    print("=" * 70)
    print("VIDEO FACE CLUSTERING AND ANNOTATION SYSTEM")
    print("=" * 70)
    
    # Check input requirements
    print("\nChecking requirements...")
    if not check_file_exists(args.input_video, "Input video"):
        return
    
    if not check_file_exists(args.model_dir, "FaceNet model directory"):
        return
    
    # Check subtitle file if provided
    if args.srt:
        if not check_file_exists(args.srt, "Subtitle file"):
            print("Warning: Subtitle file not found. Will proceed without subtitle annotation.")
            args.srt = None
    
    # Check FFmpeg if audio preservation is requested
    if args.preserve_audio:
        if check_ffmpeg_available():
            print("[OK] FFmpeg available for audio processing")
        else:
            print("[WARNING] FFmpeg not available, audio will not be preserved")
            args.preserve_audio = False
    
    # Create output directories
    dirs = create_directories(args.output_dir)
    
    # ========================================================================
    # PHASE 1: FACE CLUSTERING
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: FACE CLUSTERING")
    print("="*70)
    
    # Extract frames
    frames_paths = extract_frames(args.input_video, dirs['faces'], args.frames_interval)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print(f"\nUsing clustering method: {args.method}")
            
            # Step 1: Face detection
            print("\nStep 1: Detecting faces...")
            if args.method in ['adjusted', 'hybrid']:
                from speaking_based_filter import detect_faces_with_speaking_filter
                face_paths = detect_faces_with_speaking_filter(
                    sess, frames_paths, dirs['faces'], 
                    video_path=args.input_video,
                    min_face_size=60, face_size=args.face_size
                )
            else:
                face_paths = face_detection.detect_faces_in_frames(
                    sess, frames_paths, dirs['faces'], 
                    min_face_size=20, face_size=args.face_size
                )
                
            # Step 1.5: Quality filtering
            print("\nStep 1.5: Filtering low-quality faces...")
            from face_quality_filter import integrate_quality_filter_with_main

            face_paths = integrate_quality_filter_with_main(face_paths, args.output_dir, strict_mode=False)

            if len(face_paths) == 0:
                print("[ERROR] No faces passed quality check!")
                return

            print(f"[OK] Quality check completed, {len(face_paths)} high-quality face images retained")

            # Step 2: Feature extraction
            print("\nStep 2: Extracting facial features...")
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
            print("\nStep 3: Clustering faces...")
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
                
                if len(adjusted_clusters) > 0 and len(original_clusters) > 0:
                    original_avg_size = np.mean([len(c) for c in original_clusters])
                    adjusted_avg_size = np.mean([len(c) for c in adjusted_clusters])
                    
                    if (len(adjusted_clusters) <= len(original_clusters) * 0.7 or 
                        (len(adjusted_clusters) <= len(original_clusters) and 
                        adjusted_std < original_std * 0.8)):
                        print(f"Selected adjusted clustering: {len(adjusted_clusters)} clusters")
                        clusters = adjusted_clusters
                    else:
                        print(f"Selected original clustering: {len(original_clusters)} clusters")
                        clusters = original_clusters
                elif len(original_clusters) > 0:
                    clusters = original_clusters
                else:
                    clusters = adjusted_clusters
            else:
                clusters = clustering.cluster_facial_encodings(
                    facial_encodings, threshold=args.cluster_threshold
                )
            
            # Step 4: Post-processing
            print("\nStep 4: Post-processing clusters...")
            
            processed_clusters, merge_actions = cluster_post_processing.post_process_clusters(
                clusters, facial_encodings,
                min_large_cluster_size=50,
                small_cluster_percentage=0.08,
                merge_threshold=0.4,
                max_merges_per_cluster=15,
                safety_checks=True
            )
            
            clusters = processed_clusters
            
            print(f"[OK] Post-processing completed:")
            print(f"   Merge actions: {len(merge_actions)}")
            for action in merge_actions:
                source_cluster = action.get('cluster_j', 'unknown')
                target_cluster = action.get('cluster_i', 'unknown')
                faces_added = action.get('faces_added', 'unknown')
                
                score = action.get('confidence', action.get('similarity', 0))
                action_type = action.get('type', 'merge')
                
                print(f"   {action_type}: Cluster {source_cluster} -> Cluster {target_cluster} "
                      f"(+{faces_added} faces, score: {score:.3f})")

            # Step 5: Save clustering results
            print(f"\nSaving {len(clusters)} clusters...")
            for idx, cluster in enumerate(clusters):
                cluster_dir = os.path.join(dirs['clusters'], f"cluster_{idx}")
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                for face_path in cluster:
                    face_name = os.path.basename(face_path)
                    dst_path = os.path.join(cluster_dir, face_name)
                    shutil.copy2(face_path, dst_path)

            # Step 6: Calculate cluster centers
            print("\nStep 6: Calculating cluster centers...")
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
            
            print(f"[OK] Cluster centers saved to: {centers_data_path}")
            
            # Visualization
            if args.visualize:
                print("\nStep 7: Creating visualizations...")
                visualization.visualize_clusters(
                    clusters, facial_encodings, cluster_centers, 
                    dirs['visualization']
                )
    
    # ========================================================================
    # PHASE 2: FACE RETRIEVAL (Optional)
    # ========================================================================
    if args.do_retrieval:
        print("\n" + "="*70)
        print("PHASE 2: FACE RETRIEVAL")
        print("="*70)
        
        if args.enhanced_retrieval:
            print("\nUsing enhanced face retrieval...")
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
    
    # ========================================================================
    # PHASE 3: VIDEO ANNOTATION WITH SPEAKER-AWARE SUBTITLES
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 3: VIDEO ANNOTATION")
    print("="*70)
    
    if args.srt:
        # Initialize final subtitle offset
        final_subtitle_offset = args.subtitle_offset
        
        # Perform automatic alignment if requested
        if args.auto_align:
            print("\n" + "-"*70)
            print("AUTOMATIC SUBTITLE-AUDIO ALIGNMENT")
            print("-"*70)
            print(f"VAD Mode: {args.vad_mode}")
            if args.vad_mode == 'webrtc':
                print(f"VAD Aggressiveness: {args.vad_aggressiveness}")
            
            try:
                # Import alignment class
                from speaking_ground_truth_tool import AudioSubtitleAligner
                
                # Create aligner instance
                aligner = AudioSubtitleAligner(
                    video_path=args.input_video,
                    srt_path=args.srt,
                    vad_mode=args.vad_mode,
                    aggressiveness=args.vad_aggressiveness
                )
                
                # Run alignment process
                calculated_offset = aligner.run_alignment()
                
                # Handle alignment result
                if calculated_offset != 0.0:
                    print(f"\n" + "="*70)
                    print(f"CALCULATED OFFSET: {calculated_offset:.3f} seconds")
                    print("="*70)
                    
                    # In automated pipeline, use calculated offset if quality is good
                    final_subtitle_offset = calculated_offset
                    print(f"Applying calculated offset: {final_subtitle_offset:.3f}s")
                    
                else:
                    print("\nAuto-alignment could not determine a reliable offset.")
                    print(f"Using manual offset: {final_subtitle_offset:.3f}s")
                
            except ImportError as e:
                print(f"\nWarning: Could not import AudioSubtitleAligner")
                print(f"Error: {e}")
                print(f"Continuing with manual offset: {final_subtitle_offset:.3f}s")
                
            except Exception as e:
                print(f"\nWarning: Auto-alignment failed with error:")
                print(f"Error: {e}")
                print(f"Continuing with manual offset: {final_subtitle_offset:.3f}s")
        
        else:
            # Auto-align not requested
            if final_subtitle_offset != 0.0:
                print(f"\nUsing manual subtitle offset: {final_subtitle_offset:.3f}s")
            else:
                print(f"\nNo subtitle offset applied (using original SRT timestamps)")
        
        # Create annotated video with speaker-aware subtitles
        print("\n" + "-"*70)
        print("CREATING SPEAKER-AWARE SUBTITLE ANNOTATION")
        print("-"*70)
        print(f"Input video: {args.input_video}")
        print(f"Subtitle file: {args.srt}")
        print(f"Final subtitle offset: {final_subtitle_offset:.3f}s")
        print(f"Face matching threshold: {args.similarity_threshold}")
        print(f"Speaking detection threshold: {args.speaking_threshold}")
        print(f"BBox smoothing alpha: {args.smoothing_alpha}")
        print(f"Generate JSON annotation: {args.generate_json}")
        
        output_video = os.path.join(args.output_dir, 'speaker_subtitle_annotated_video.mp4')
        
        try:
            speaker_subtitle_annotation.annotate_video_with_speaker_subtitles(
                input_video=args.input_video,
                output_video=output_video,
                centers_data_path=centers_data_path,
                subtitle_path=args.srt,
                model_dir=args.model_dir,
                detection_interval=args.detection_interval,
                similarity_threshold=args.similarity_threshold,
                speaking_threshold=args.speaking_threshold,
                preserve_audio=args.preserve_audio,
                smoothing_alpha=args.smoothing_alpha,
                generate_json=args.generate_json,
                subtitle_offset=final_subtitle_offset
            )
            
            print(f"\n[SUCCESS] Speaker-aware subtitle annotation completed!")
            print(f"Output video: {output_video}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to create speaker-aware subtitle annotation: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.use_legacy_annotation:
        # Use legacy annotation without subtitles
        print("\nCreating video with legacy annotation (no subtitles)...")
        
        output_video = os.path.join(args.output_dir, 'annotated_video.avi')
        
        try:
            enhanced_video_annotation.annotate_video_with_enhanced_detection(
                input_video=args.input_video,
                output_video=output_video,
                centers_data_path=centers_data_path,
                model_dir=args.model_dir,
                detection_interval=args.detection_interval,
                similarity_threshold=args.similarity_threshold,
                temporal_weight=args.temporal_weight
            )
            
            print(f"\n[SUCCESS] Legacy annotation completed!")
            print(f"Output video: {output_video}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to create legacy annotation: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("\n[INFO] No video annotation requested.")
        print("To create annotated video:")
        print("  - Provide --subtitle_file for speaker-aware subtitle annotation (recommended)")
        print("  - Or use --use_legacy_annotation for basic face annotation")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "="*70)
    print("PROCESSING COMPLETED")
    print("="*70)
    
    # Final summary
    print("\nFinal Output:")
    print(f"- Clustering results: {dirs['clusters']}")
    print(f"- Cluster centers: {centers_data_path}")
    
    if args.visualize:
        print(f"- Visualizations: {dirs['visualization']}")
    
    if args.do_retrieval:
        print(f"- Face retrieval: {dirs['retrieval']}")
    
    if args.srt:
        output_video = os.path.join(args.output_dir, 'speaker_subtitle_annotated_video.mp4')
        if os.path.exists(output_video):
            size = os.path.getsize(output_video) / (1024*1024)
            print(f"- Annotated video: {output_video} ({size:.1f} MB)")
            
            # Check for JSON annotation file
            json_path = output_video.replace('.mp4', '_annotation.json')
            if args.generate_json and os.path.exists(json_path):
                json_size = os.path.getsize(json_path) / 1024
                print(f"- JSON annotation: {json_path} ({json_size:.1f} KB)")
    elif args.use_legacy_annotation:
        output_video = os.path.join(args.output_dir, 'annotated_video.avi')
        if os.path.exists(output_video):
            size = os.path.getsize(output_video) / (1024*1024)
            print(f"- Annotated video: {output_video} ({size:.1f} MB)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ERROR] Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)