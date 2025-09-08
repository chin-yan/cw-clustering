#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影片音訊合併工具
使用 FFmpeg 將音訊檔案與影片檔案合併
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class VideoAudioMerger:
    def __init__(self):
        self.ffmpeg_path = self.check_ffmpeg()
    
    def check_ffmpeg(self):
        """檢查 FFmpeg 是否已安裝"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ FFmpeg 已安裝")
                return 'ffmpeg'
        except FileNotFoundError:
            pass
        
        # 檢查常見的 FFmpeg 安裝路徑
        possible_paths = [
            'ffmpeg.exe',
            './ffmpeg.exe',
            '/usr/local/bin/ffmpeg',
            '/usr/bin/ffmpeg'
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ 找到 FFmpeg: {path}")
                    return path
            except FileNotFoundError:
                continue
        
        print("❌ 未找到 FFmpeg，請先安裝 FFmpeg")
        print("下載地址: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    def validate_files(self, video_path, audio_path):
        """驗證檔案是否存在"""
        video_file = Path(video_path)
        audio_file = Path(audio_path)
        
        if not video_file.exists():
            raise FileNotFoundError(f"找不到影片檔案: {video_path}")
        
        if not audio_file.exists():
            raise FileNotFoundError(f"找不到音訊檔案: {audio_path}")
        
        # 檢查檔案格式
        video_ext = video_file.suffix.lower()
        audio_ext = audio_file.suffix.lower()
        
        supported_video = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
        supported_audio = ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma']
        
        if video_ext not in supported_video:
            print(f"警告: 影片格式 {video_ext} 可能不被支援")
        
        if audio_ext not in supported_audio:
            print(f"警告: 音訊格式 {audio_ext} 可能不被支援")
        
        return True
    
    def get_duration(self, file_path):
        """獲取媒體檔案的時長"""
        try:
            cmd = [
                self.ffmpeg_path, '-i', file_path,
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 從 stderr 中解析時長資訊
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    duration_str = line.split('Duration:')[1].split(',')[0].strip()
                    return duration_str
            
            return "未知"
        except:
            return "未知"
    
    def merge_replace_audio(self, video_path, audio_path, output_path, 
                           video_quality='medium'):
        """
        替換影片的音訊軌道
        
        Args:
            video_path: 影片檔案路徑
            audio_path: 音訊檔案路徑
            output_path: 輸出檔案路徑
            video_quality: 影片品質 ('high', 'medium', 'low', 'copy')
        """
        try:
            self.validate_files(video_path, audio_path)
            
            print(f"影片時長: {self.get_duration(video_path)}")
            print(f"音訊時長: {self.get_duration(audio_path)}")
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-i', audio_path,
                '-map', '0:v',  # 使用第一個檔案的影片軌
                '-map', '1:a',  # 使用第二個檔案的音訊軌
            ]
            
            # 影片編碼設定
            if video_quality == 'copy':
                cmd.extend(['-c:v', 'copy'])
            elif video_quality == 'high':
                cmd.extend(['-c:v', 'libx264', '-crf', '18'])
            elif video_quality == 'medium':
                cmd.extend(['-c:v', 'libx264', '-crf', '23'])
            elif video_quality == 'low':
                cmd.extend(['-c:v', 'libx264', '-crf', '28'])
            
            # 音訊編碼設定
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            
            # 時長處理：使用較短的媒體檔案長度
            cmd.extend(['-shortest'])
            
            # 輸出檔案
            cmd.extend(['-y', output_path])
            
            print(f"開始合併影片和音訊...")
            print(f"影片: {video_path}")
            print(f"音訊: {audio_path}")
            print(f"輸出: {output_path}")
            print(f"影片品質: {video_quality}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 影片音訊合併完成！")
                return True
            else:
                print("❌ 合併失敗:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 發生錯誤: {str(e)}")
            return False
    
    def merge_mix_audio(self, video_path, audio_path, output_path, 
                       video_quality='medium', audio_volume=1.0, bg_volume=0.3):
        """
        混合影片原音訊和新音訊
        
        Args:
            video_path: 影片檔案路徑
            audio_path: 音訊檔案路徑
            output_path: 輸出檔案路徑
            video_quality: 影片品質
            audio_volume: 新音訊音量 (0.0-2.0)
            bg_volume: 原音訊音量 (0.0-2.0)
        """
        try:
            self.validate_files(video_path, audio_path)
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-i', audio_path,
            ]
            
            # 音訊混合濾鏡
            audio_filter = f"[0:a]volume={bg_volume}[a0];[1:a]volume={audio_volume}[a1];[a0][a1]amix=inputs=2:duration=shortest[aout]"
            cmd.extend(['-filter_complex', audio_filter])
            
            # 映射影片和混合後的音訊
            cmd.extend(['-map', '0:v', '-map', '[aout]'])
            
            # 影片編碼設定
            if video_quality == 'copy':
                cmd.extend(['-c:v', 'copy'])
            elif video_quality == 'high':
                cmd.extend(['-c:v', 'libx264', '-crf', '18'])
            elif video_quality == 'medium':
                cmd.extend(['-c:v', 'libx264', '-crf', '23'])
            elif video_quality == 'low':
                cmd.extend(['-c:v', 'libx264', '-crf', '28'])
            
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            cmd.extend(['-y', output_path])
            
            print(f"開始混合音訊...")
            print(f"影片: {video_path}")
            print(f"背景音訊: {audio_path}")
            print(f"原音量: {bg_volume}, 新音量: {audio_volume}")
            print(f"輸出: {output_path}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 音訊混合完成！")
                return True
            else:
                print("❌ 混合失敗:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 發生錯誤: {str(e)}")
            return False
    
    def extract_audio(self, video_path, output_path, audio_format='mp3'):
        """
        從影片中提取音訊
        """
        try:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"找不到影片檔案: {video_path}")
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vn',  # 不要影片
                '-acodec', 'mp3' if audio_format == 'mp3' else 'copy',
                '-y', output_path
            ]
            
            print(f"從影片提取音訊...")
            print(f"影片: {video_path}")
            print(f"輸出: {output_path}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 音訊提取完成！")
                return True
            else:
                print("❌ 提取失敗:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 發生錯誤: {str(e)}")
            return False
    
    def batch_merge(self, video_dir, audio_dir, output_dir, 
                   merge_type='replace', video_quality='medium'):
        """
        批次處理多個檔案
        """
        video_dir = Path(video_dir)
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        
        # 創建輸出目錄
        output_dir.mkdir(exist_ok=True)
        
        # 找到所有影片檔案
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.mov']:
            video_files.extend(video_dir.glob(ext))
        
        success_count = 0
        total_count = len(video_files)
        
        for video_file in video_files:
            # 尋找對應的音訊檔案
            audio_file = None
            for ext in ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a']:
                potential_audio = audio_dir / f"{video_file.stem}{ext}"
                if potential_audio.exists():
                    audio_file = potential_audio
                    break
            
            if audio_file is None:
                print(f"⚠️  找不到 {video_file.name} 的音訊檔案，跳過")
                continue
            
            # 輸出檔案名稱
            output_file = output_dir / f"{video_file.stem}_merged{video_file.suffix}"
            
            print(f"\n處理 ({success_count + 1}/{total_count}): {video_file.name}")
            
            # 合併音訊
            if merge_type == 'replace':
                success = self.merge_replace_audio(
                    str(video_file), str(audio_file), str(output_file), video_quality
                )
            else:
                success = self.merge_mix_audio(
                    str(video_file), str(audio_file), str(output_file), video_quality
                )
            
            if success:
                success_count += 1
        
        print(f"\n批次處理完成: {success_count}/{total_count} 個檔案處理成功")


def main():
    parser = argparse.ArgumentParser(description='影片音訊合併工具')
    parser.add_argument('video', help='影片檔案路徑')
    parser.add_argument('audio', help='音訊檔案路徑')
    parser.add_argument('-o', '--output', help='輸出檔案路徑')
    parser.add_argument('-q', '--quality', choices=['high', 'medium', 'low', 'copy'], 
                       default='medium', help='影片品質')
    parser.add_argument('-t', '--type', choices=['replace', 'mix'], 
                       default='replace', help='合併類型 (replace=替換音訊, mix=混合音訊)')
    parser.add_argument('--audio-volume', type=float, default=1.0, 
                       help='新音訊音量 (0.0-2.0，僅混合模式)')
    parser.add_argument('--bg-volume', type=float, default=0.3, 
                       help='原音訊音量 (0.0-2.0，僅混合模式)')
    parser.add_argument('--batch', action='store_true', help='批次處理模式')
    parser.add_argument('--extract', action='store_true', help='從影片提取音訊')
    
    args = parser.parse_args()
    
    merger = VideoAudioMerger()
    
    if args.extract:
        # 音訊提取模式
        if not args.output:
            video_path = Path(args.video)
            args.output = f"{video_path.stem}_audio.mp3"
        
        merger.extract_audio(args.video, args.output)
        
    elif args.batch:
        # 批次處理模式
        if not args.output:
            args.output = './output'
        
        merger.batch_merge(args.video, args.audio, args.output, 
                          args.type, args.quality)
    else:
        # 單檔處理模式
        if not args.output:
            video_path = Path(args.video)
            args.output = f"{video_path.stem}_merged{video_path.suffix}"
        
        if args.type == 'replace':
            merger.merge_replace_audio(args.video, args.audio, args.output, args.quality)
        else:
            merger.merge_mix_audio(args.video, args.audio, args.output, 
                                 args.quality, args.audio_volume, args.bg_volume)


if __name__ == '__main__':
    # 如果沒有命令列參數，進入互動模式
    if len(sys.argv) == 1:
        print("=== 影片音訊合併工具 ===\n")
        
        merger = VideoAudioMerger()
        
        try:
            print("請選擇功能:")
            print("1. 合併影片和音訊")
            print("2. 從影片提取音訊")
            
            function_choice = input("請選擇 (1-2): ").strip()
            
            if function_choice == '2':
                # 音訊提取功能
                video_path = input("請輸入影片檔案路徑: ").strip('"')
                output_path = input("請輸入輸出音訊檔案路徑 (按 Enter 使用預設): ").strip('"')
                
                if not output_path:
                    video_file = Path(video_path)
                    output_path = f"{video_file.stem}_audio.mp3"
                
                merger.extract_audio(video_path, output_path)
                
            else:
                # 影片音訊合併功能
                video_path = input("請輸入影片檔案路徑: ").strip('"')
                audio_path = input("請輸入音訊檔案路徑: ").strip('"')
                output_path = input("請輸入輸出檔案路徑 (按 Enter 使用預設): ").strip('"')
                
                if not output_path:
                    video_file = Path(video_path)
                    output_path = f"{video_file.stem}_merged{video_file.suffix}"
                
                # 選擇合併類型
                print("\n合併類型:")
                print("1. 替換音訊 (移除原音訊，使用新音訊)")
                print("2. 混合音訊 (原音訊+新音訊)")
                
                merge_choice = input("請選擇 (1-2，預設為1): ").strip() or "1"
                
                if merge_choice == '2':
                    # 混合音訊模式
                    print("\n音量設定:")
                    audio_volume = input("新音訊音量 (0.0-2.0，預設1.0): ").strip() or "1.0"
                    bg_volume = input("原音訊音量 (0.0-2.0，預設0.3): ").strip() or "0.3"
                    
                    try:
                        audio_volume = float(audio_volume)
                        bg_volume = float(bg_volume)
                    except:
                        audio_volume, bg_volume = 1.0, 0.3
                    
                    # 選擇影片品質
                    print("\n影片品質:")
                    print("1. 複製原檔 (最快，品質不變)")
                    print("2. 高品質")
                    print("3. 中等品質 (推薦)")
                    print("4. 低品質")
                    
                    quality_choice = input("請選擇 (1-4，預設為3): ").strip() or "3"
                    quality_map = {'1': 'copy', '2': 'high', '3': 'medium', '4': 'low'}
                    quality = quality_map.get(quality_choice, 'medium')
                    
                    merger.merge_mix_audio(video_path, audio_path, output_path, 
                                         quality, audio_volume, bg_volume)
                else:
                    # 替換音訊模式
                    print("\n影片品質:")
                    print("1. 複製原檔 (最快，品質不變)")
                    print("2. 高品質")
                    print("3. 中等品質 (推薦)")
                    print("4. 低品質")
                    
                    quality_choice = input("請選擇 (1-4，預設為3): ").strip() or "3"
                    quality_map = {'1': 'copy', '2': 'high', '3': 'medium', '4': 'low'}
                    quality = quality_map.get(quality_choice, 'medium')
                    
                    merger.merge_replace_audio(video_path, audio_path, output_path, quality)
                    
        except KeyboardInterrupt:
            print("\n\n程式已取消")
        except Exception as e:
            print(f"\n發生錯誤: {str(e)}")
    else:
        main()