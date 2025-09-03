# -*- coding: utf-8 -*-

import os
import subprocess
import json

def diagnose_video_audio(video_path):
    """
    Diagnose video file for audio information using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Audio information or None if no audio
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-select_streams', 'a', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get('streams'):
                audio_stream = data['streams'][0]
                print(f"📊 Audio info for {os.path.basename(video_path)}:")
                print(f"   Codec: {audio_stream.get('codec_name', 'unknown')}")
                print(f"   Sample Rate: {audio_stream.get('sample_rate', 'unknown')} Hz")
                print(f"   Channels: {audio_stream.get('channels', 'unknown')}")
                print(f"   Duration: {audio_stream.get('duration', 'unknown')} seconds")
                return audio_stream
            else:
                print(f"❌ No audio stream found in {os.path.basename(video_path)}")
                return None
        else:
            print(f"❌ ffprobe failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error diagnosing audio: {e}")
        return None

def enhanced_merge_audio_with_video(input_video, silent_video, output_video, verbose=True):
    """
    Enhanced audio merging with detailed diagnostics
    
    Args:
        input_video: Path to original video file (with audio)
        silent_video: Path to processed video file (without audio)
        output_video: Path to final output video file (with audio)
        verbose: Whether to print detailed progress information
    
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print(f"🎵 Enhanced audio merge process...")
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
    
    # Diagnose audio in original video
    print("🔍 Diagnosing original video audio...")
    original_audio = diagnose_video_audio(input_video)
    if not original_audio:
        print("❌ Original video has no audio to extract!")
        return False
    
    # Diagnose silent video (should have no audio)
    print("🔍 Diagnosing processed video...")
    silent_audio = diagnose_video_audio(silent_video)
    if silent_audio:
        print("⚠️  Processed video unexpectedly has audio")
    
    # Check FFmpeg availability with more detailed info
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ FFmpeg not working properly")
            return False
        print("✅ FFmpeg is available")
    except:
        print("❌ FFmpeg not found")
        return False
    
    try:
        # Create output directory
        output_dir = os.path.dirname(output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Enhanced FFmpeg command with multiple fallback strategies
        commands_to_try = [
            # Strategy 1: Copy video, copy audio
            [
                'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                '-c:v', 'copy', '-c:a', 'copy',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', output_video
            ],
            # Strategy 2: Re-encode audio to AAC
            [
                'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', output_video
            ],
            # Strategy 3: Force audio stream mapping
            [
                'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                '-c:v', 'copy', '-c:a', 'aac',
                '-map', '0:0', '-map', '1:1',
                '-shortest', output_video
            ],
            # Strategy 4: Most compatible settings
            [
                'ffmpeg', '-y', '-i', silent_video, '-i', input_video,
                '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '128k',
                '-map', '0:v', '-map', '1:a',
                '-shortest', '-movflags', '+faststart',
                output_video
            ]
        ]
        
        success = False
        for i, cmd in enumerate(commands_to_try, 1):
            print(f"🔄 Trying audio merge strategy {i}/{len(commands_to_try)}...")
            if verbose:
                print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Verify output file
                if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                    print(f"✅ Strategy {i} succeeded!")
                    
                    # Verify audio in output
                    print("🔍 Verifying audio in output video...")
                    output_audio = diagnose_video_audio(output_video)
                    if output_audio:
                        print("✅ Audio successfully merged!")
                        success = True
                        break
                    else:
                        print(f"⚠️  Strategy {i} completed but no audio detected in output")
                        # Try next strategy
                        continue
                else:
                    print(f"❌ Strategy {i} failed - no valid output file")
            else:
                print(f"❌ Strategy {i} failed with return code: {result.returncode}")
                if verbose:
                    print(f"Error: {result.stderr}")
        
        if success:
            file_size = os.path.getsize(output_video)
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
            print("❌ All audio merge strategies failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg operation timed out")
        return False
    except Exception as e:
        print(f"❌ Audio merging failed with exception: {e}")
        return False

def test_audio_merge(input_video, silent_video=None, output_video=None):
    """
    Standalone function to test audio merging
    
    Args:
        input_video: Path to original video with audio
        silent_video: Path to processed video without audio (optional)
        output_video: Path for output (optional)
    """
    print("🧪 Testing Audio Merge Function")
    print("=" * 50)
    
    # If no silent video provided, create a test one
    if not silent_video:
        print("Creating test silent video...")
        silent_video = input_video.replace('.mp4', '_test_silent.avi')
        
        # Create a short silent video from original
        cmd = [
            'ffmpeg', '-y', '-i', input_video, '-t', '10',  # First 10 seconds
            '-c:v', 'copy', '-an',  # Copy video, remove audio
            silent_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to create test silent video: {result.stderr}")
            return False
    
    # Set default output if not provided
    if not output_video:
        output_video = input_video.replace('.mp4', '_test_with_audio.mp4')
    
    # Test the enhanced merge function
    success = enhanced_merge_audio_with_video(input_video, silent_video, output_video)
    
    if success:
        print(f"🎉 Test successful! Check output: {output_video}")
    else:
        print("❌ Test failed!")
    
    return success

def quick_audio_fix(problematic_video, original_video, fixed_output):
    """
    Quick fix for videos that lost their audio
    
    Args:
        problematic_video: Video file without audio
        original_video: Original video with audio
        fixed_output: Path for fixed output
    """
    print("🔧 Quick Audio Fix")
    print("-" * 30)
    
    return enhanced_merge_audio_with_video(original_video, problematic_video, fixed_output)

if __name__ == "__main__":
    # Test with your actual files
    input_video = r"C:\Users\VIPLAB\Desktop\Yan\Drama_FresfOnTheBoat\FreshOnTheBoatOnYoutube\0.mp4"
    
    # Option 1: Test audio merge function
    print("Testing audio merge functionality...")
    test_audio_merge(input_video)
