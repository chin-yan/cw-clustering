# -*- coding: utf-8 -*-

import os
import subprocess

def check_ffmpeg_tools():
    """
    Check availability of FFmpeg tools
    
    Returns:
        dict: Status of ffmpeg and ffprobe
    """
    tools_status = {}
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        tools_status['ffmpeg'] = result.returncode == 0
        if tools_status['ffmpeg']:
            print("✅ FFmpeg is available")
        else:
            print("❌ FFmpeg command failed")
    except:
        tools_status['ffmpeg'] = False
        print("❌ FFmpeg not found")
    
    # Check ffprobe
    try:
        result = subprocess.run(['ffprobe', '-version'], 
                              capture_output=True, text=True, timeout=5)
        tools_status['ffprobe'] = result.returncode == 0
        if tools_status['ffprobe']:
            print("✅ ffprobe is available")
        else:
            print("❌ ffprobe command failed")
    except:
        tools_status['ffprobe'] = False
        print("❌ ffprobe not found")
    
    return tools_status

def simple_audio_test(video_path):
    """
    Simple test to check if video has audio using FFmpeg only
    
    Args:
        video_path: Path to video file
        
    Returns:
        bool: True if audio detected, False otherwise
    """
    try:
        # Use FFmpeg to extract just 1 second of audio to test
        temp_audio = "temp_audio_test.wav"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-t', '1',
            '-vn', '-acodec', 'pcm_s16le', temp_audio
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Check if temp audio file was created and has content
        has_audio = False
        if os.path.exists(temp_audio):
            size = os.path.getsize(temp_audio)
            has_audio = size > 1000  # At least 1KB for 1 second of audio
            
            # Clean up
            try:
                os.remove(temp_audio)
            except:
                pass
        
        if has_audio:
            print(f"✅ Audio detected in {os.path.basename(video_path)}")
        else:
            print(f"❌ No audio detected in {os.path.basename(video_path)}")
            
        return has_audio
        
    except Exception as e:
        print(f"❌ Error testing audio: {e}")
        return False

def robust_merge_audio_with_video(input_video, silent_video, output_video, verbose=True):
    """
    Robust audio merging without ffprobe dependency
    
    Args:
        input_video: Path to original video file (with audio)
        silent_video: Path to processed video file (without audio)
        output_video: Path to final output video file (with audio)
        verbose: Whether to print detailed progress information
    
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print(f"🎵 Robust audio merge process...")
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
    
    # Check FFmpeg tools
    tools_status = check_ffmpeg_tools()
    if not tools_status['ffmpeg']:
        print("❌ FFmpeg is required but not available")
        return False
    
    # Test if original video has audio
    print("🔍 Testing original video for audio...")
    if not simple_audio_test(input_video):
        print("❌ Original video appears to have no audio!")
        return False
    
    try:
        # Create output directory
        output_dir = os.path.dirname(output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Multiple strategies for audio merging
        strategies = [
            {
                'name': 'Copy both streams',
                'cmd': [
                    'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                    '-c:v', 'copy', '-c:a', 'copy',
                    '-map', '0:v:0', '-map', '1:a:0',
                    '-shortest', output_video
                ]
            },
            {
                'name': 'Re-encode audio to AAC',
                'cmd': [
                    'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                    '-map', '0:v:0', '-map', '1:a:0',
                    '-shortest', output_video
                ]
            },
            {
                'name': 'Safe compatibility mode',
                'cmd': [
                    'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                    '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-b:a', '128k',
                    '-map', '0:v', '-map', '1:a',
                    '-shortest', '-movflags', '+faststart',
                    output_video
                ]
            },
            {
                'name': 'Force stream selection',
                'cmd': [
                    'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                    '-c:v', 'copy', '-c:a', 'aac',
                    '-map', '0:0', '-map', '1:1',
                    '-shortest', output_video
                ]
            }
        ]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"🔄 Trying strategy {i}: {strategy['name']}")
            
            if verbose:
                print(f"   Command: {' '.join(strategy['cmd'])}")
            
            result = subprocess.run(strategy['cmd'], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Check if output file exists and has reasonable size
                if os.path.exists(output_video):
                    size = os.path.getsize(output_video)
                    if size > 1024:  # At least 1KB
                        print(f"✅ Strategy {i} succeeded!")
                        
                        # Test if output has audio
                        if simple_audio_test(output_video):
                            print("✅ Audio successfully merged!")
                            
                            file_size_mb = size / (1024*1024)
                            print(f"📁 Output file: {output_video}")
                            print(f"📊 File size: {file_size_mb:.2f} MB")
                            
                            # Clean up temporary file
                            try:
                                if os.path.exists(silent_video) and '_temp_silent' in silent_video:
                                    os.remove(silent_video)
                                    if verbose:
                                        print(f"🗑️  Removed temporary file: {silent_video}")
                            except Exception as e:
                                if verbose:
                                    print(f"⚠️  Could not remove temporary file: {e}")
                            
                            return True
                        else:
                            print(f"⚠️  Strategy {i} completed but output has no audio")
                            # Try next strategy
                            continue
                    else:
                        print(f"❌ Strategy {i} produced empty or tiny file")
                else:
                    print(f"❌ Strategy {i} - output file not created")
            else:
                print(f"❌ Strategy {i} failed with return code: {result.returncode}")
                if verbose and result.stderr:
                    # Show only the last few lines of error to avoid spam
                    error_lines = result.stderr.strip().split('\n')
                    print(f"   Error: {error_lines[-1]}")
        
        print("❌ All strategies failed!")
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg operation timed out")
        return False
    except Exception as e:
        print(f"❌ Audio merging failed: {e}")
        return False

def fix_existing_video(problematic_video, original_video, output_video):
    """
    Fix an existing video that has no audio
    """
    print("🔧 Fixing existing video with missing audio")
    print("=" * 50)
    
    return robust_merge_audio_with_video(original_video, problematic_video, output_video)

def test_video_audio_simple(video_path):
    """
    Simple test function
    """
    print(f"🧪 Testing video: {video_path}")
    print("-" * 50)
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return False
    
    file_size = os.path.getsize(video_path)
    print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
    
    # Test for audio
    has_audio = simple_audio_test(video_path)
    
    return has_audio

if __name__ == "__main__":
    # Test your files
    print("🚀 Audio Diagnosis and Repair Tool")
    print("=" * 50)
    
    # Check tools first
    print("1. Checking available tools...")
    check_ffmpeg_tools()
    
    # Test original video
    print("\n2. Testing original video...")
    original_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    test_video_audio_simple(original_video)
    
    # Test problematic video
    print("\n3. Testing problematic video...")
    problematic_video = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result_0831\enhanced_annotated_video.avi"
    if os.path.exists(problematic_video):
        test_video_audio_simple(problematic_video)
        
        # Try to fix it
        print("\n4. Attempting to fix the video...")
        fixed_output = problematic_video.replace('.mp4', '_FIXED.mp4')
        success = fix_existing_video(problematic_video, original_video, fixed_output)
        
        if success:
            print(f"🎉 Fixed video saved to: {fixed_output}")
        else:
            print("❌ Failed to fix the video")
    else:
        print(f"❌ Problematic video not found: {problematic_video}")