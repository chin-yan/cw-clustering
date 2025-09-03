import subprocess
import os
import enhanced_video_annotation

def merge_audio_with_video(input_video, silent_video, output_video):
    """
    Merge audio from original video with processed video using FFmpeg
    
    Args:
        input_video: Path to original video file (with audio)
        silent_video: Path to processed video file (without audio)
        output_video: Path to final output video file (with audio)
    """
    try:
        # Check if FFmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        
        # FFmpeg command: copy audio from original video to processed video
        cmd = [
            'ffmpeg', '-y',           # -y means overwrite output file
            '-i', silent_video,       # Input: processed video (visual part)
            '-i', input_video,        # Input: original video (audio part)
            '-c:v', 'copy',           # Copy video stream without re-encoding
            '-c:a', 'aac',            # Encode audio to AAC
            '-map', '0:v:0',          # Use video stream from first input
            '-map', '1:a:0',          # Use audio stream from second input
            '-shortest',              # Use duration of shorter stream
            output_video
        ]
        
        print("Merging audio...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully merged audio, output saved to: {output_video}")
            # Remove temporary silent video file
            try:
                os.remove(silent_video)
                print(f"Removed temporary file: {silent_video}")
            except:
                pass
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ FFmpeg is not installed or not available")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"❌ Audio merging failed: {e}")
        return False

def enhanced_annotate_video_with_audio(input_video, output_video, centers_data_path, model_dir,
                                     detection_interval=1, similarity_threshold=0.55, 
                                     temporal_weight=0.3):
    """
    Enhanced video annotation function that preserves audio
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        centers_data_path: Path to cluster centers data
        model_dir: FaceNet model directory
        detection_interval: Frame interval for face detection
        similarity_threshold: Similarity threshold for face matching
        temporal_weight: Weight for temporal consistency
    """
    # Create temporary video file name
    temp_video = output_video.replace('.avi', '_temp.avi').replace('.mp4', '_temp.mp4')
    
    # Call original video annotation function, but output to temporary file
    enhanced_video_annotation.annotate_video_with_enhanced_detection(
        input_video=input_video,
        output_video=temp_video,  # Output to temporary file
        centers_data_path=centers_data_path,
        model_dir=model_dir,
        detection_interval=detection_interval,
        similarity_threshold=similarity_threshold,
        temporal_weight=temporal_weight
    )
    
    # Merge audio
    success = merge_audio_with_video(input_video, temp_video, output_video)
    
    if not success:
        print("⚠️  Audio merging failed, but video annotation completed (without audio)")
        # If audio merging fails, rename temp file to final file
        try:
            if os.path.exists(temp_video):
                os.rename(temp_video, output_video)
                print(f"Silent video saved to: {output_video}")
        except:
            pass
    
    return success