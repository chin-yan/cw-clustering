#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影片合併工具
使用 FFmpeg 合併兩個或多個影片檔案
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import tempfile


class VideoMerger:
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
    
    def validate_files(self, video_paths):
        """驗證影片檔案是否存在"""
        for video_path in video_paths:
            video_file = Path(video_path)
            
            if not video_file.exists():
                raise FileNotFoundError(f"找不到影片檔案: {video_path}")
            
            # 檢查檔案格式
            video_ext = video_file.suffix.lower()
            supported_video = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
            
            if video_ext not in supported_video:
                print(f"警告: 影片格式 {video_ext} 可能不被支援")
        
        return True
    
    def get_video_info(self, video_path):
        """獲取影片資訊"""
        try:
            cmd = [
                self.ffmpeg_path, '-i', video_path,
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            info = {}
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    duration = line.split('Duration:')[1].split(',')[0].strip()
                    info['duration'] = duration
                elif 'Video:' in line:
                    # 解析解析度
                    if 'x' in line:
                        resolution_part = line.split(',')[2].strip() if ',' in line else line
                        for part in line.split(','):
                            if 'x' in part and any(char.isdigit() for char in part):
                                resolution = part.strip().split()[0]
                                if 'x' in resolution:
                                    info['resolution'] = resolution
                                    break
                elif 'fps' in line:
                    for part in line.split(','):
                        if 'fps' in part:
                            fps = part.strip().split()[0]
                            info['fps'] = fps
                            break
            
            return info
        except:
            return {}
    
    def concatenate_videos(self, video_paths, output_path, quality='medium', 
                          resize_mode='none', target_resolution=None):
        """
        連接多個影片檔案（順序播放）
        
        Args:
            video_paths: 影片檔案路徑列表
            output_path: 輸出檔案路徑
            quality: 影片品質 ('high', 'medium', 'low', 'copy')
            resize_mode: 調整大小模式 ('none', 'first', 'largest', 'smallest', 'custom')
            target_resolution: 自訂解析度 (如 '1920x1080')
        """
        try:
            self.validate_files(video_paths)
            
            # 顯示影片資訊
            print("影片資訊:")
            video_infos = []
            for i, video_path in enumerate(video_paths, 1):
                info = self.get_video_info(video_path)
                video_infos.append(info)
                print(f"  影片 {i}: {Path(video_path).name}")
                print(f"    時長: {info.get('duration', '未知')}")
                print(f"    解析度: {info.get('resolution', '未知')}")
                print(f"    幀率: {info.get('fps', '未知')} fps")
            
            # 決定輸出解析度
            if resize_mode != 'none':
                target_res = self._determine_target_resolution(
                    video_infos, resize_mode, target_resolution
                )
                if target_res:
                    print(f"目標解析度: {target_res}")
            
            # 創建臨時檔案列表
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                temp_list_file = f.name
                for video_path in video_paths:
                    # 使用絕對路徑並轉換路徑分隔符
                    abs_path = os.path.abspath(video_path).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")
            
            try:
                cmd = [self.ffmpeg_path, '-f', 'concat', '-safe', '0', '-i', temp_list_file]
                
                # 如果需要調整大小
                if resize_mode != 'none' and target_res:
                    width, height = target_res.split('x')
                    cmd.extend(['-vf', f'scale={width}:{height}'])
                
                # 品質設定
                if quality == 'copy':
                    # 只有在所有影片格式相同時才能使用 copy
                    cmd.extend(['-c', 'copy'])
                else:
                    # 重新編碼
                    if quality == 'high':
                        cmd.extend(['-c:v', 'libx264', '-crf', '18'])
                    elif quality == 'medium':
                        cmd.extend(['-c:v', 'libx264', '-crf', '23'])
                    elif quality == 'low':
                        cmd.extend(['-c:v', 'libx264', '-crf', '28'])
                    
                    cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
                
                cmd.extend(['-y', output_path])
                
                print(f"\n開始合併影片...")
                print(f"輸出檔案: {output_path}")
                print(f"影片品質: {quality}")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✓ 影片合併完成！")
                    return True
                else:
                    print("❌ 合併失敗:")
                    print(result.stderr)
                    return False
            
            finally:
                # 清理臨時檔案
                try:
                    os.unlink(temp_list_file)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ 發生錯誤: {str(e)}")
            return False
    
    def side_by_side_videos(self, video1_path, video2_path, output_path, 
                           quality='medium', layout='horizontal'):
        """
        並排顯示兩個影片
        
        Args:
            video1_path: 第一個影片路徑
            video2_path: 第二個影片路徑
            output_path: 輸出檔案路徑
            quality: 影片品質
            layout: 佈局 ('horizontal', 'vertical')
        """
        try:
            self.validate_files([video1_path, video2_path])
            
            # 獲取影片資訊
            info1 = self.get_video_info(video1_path)
            info2 = self.get_video_info(video2_path)
            
            print("影片資訊:")
            print(f"  影片1: {Path(video1_path).name} - {info1.get('resolution', '未知')}")
            print(f"  影片2: {Path(video2_path).name} - {info2.get('resolution', '未知')}")
            
            cmd = [
                self.ffmpeg_path,
                '-i', video1_path,
                '-i', video2_path
            ]
            
            # 設定濾鏡
            if layout == 'horizontal':
                # 水平排列 (左右)
                filter_complex = '[0:v]scale=iw/2:ih[v0];[1:v]scale=iw/2:ih[v1];[v0][v1]hstack=inputs=2[v]'
            else:
                # 垂直排列 (上下)
                filter_complex = '[0:v]scale=iw:ih/2[v0];[1:v]scale=iw:ih/2[v1];[v0][v1]vstack=inputs=2[v]'
            
            cmd.extend(['-filter_complex', filter_complex])
            cmd.extend(['-map', '[v]'])
            
            # 音訊處理 (混合兩個音訊)
            cmd.extend(['-filter_complex', 
                       f'{filter_complex};[0:a][1:a]amix=inputs=2:duration=shortest[a]'])
            cmd.extend(['-map', '[a]'])
            
            # 品質設定
            if quality == 'high':
                cmd.extend(['-c:v', 'libx264', '-crf', '18'])
            elif quality == 'medium':
                cmd.extend(['-c:v', 'libx264', '-crf', '23'])
            elif quality == 'low':
                cmd.extend(['-c:v', 'libx264', '-crf', '28'])
            
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            cmd.extend(['-shortest'])  # 使用較短影片的長度
            cmd.extend(['-y', output_path])
            
            print(f"\n開始建立{layout}並排影片...")
            print(f"輸出檔案: {output_path}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 並排影片建立完成！")
                return True
            else:
                print("❌ 建立失敗:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 發生錯誤: {str(e)}")
            return False
    
    def picture_in_picture(self, main_video_path, overlay_video_path, output_path,
                          quality='medium', position='top-right', scale=0.25):
        """
        子母畫面 (Picture in Picture)
        
        Args:
            main_video_path: 主影片路徑
            overlay_video_path: 疊加影片路徑  
            output_path: 輸出檔案路徑
            quality: 影片品質
            position: 位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
            scale: 小影片縮放比例 (0.1-1.0)
        """
        try:
            self.validate_files([main_video_path, overlay_video_path])
            
            # 位置對應
            position_map = {
                'top-left': '10:10',
                'top-right': 'main_w-overlay_w-10:10',
                'bottom-left': '10:main_h-overlay_h-10',
                'bottom-right': 'main_w-overlay_w-10:main_h-overlay_h-10',
                'center': '(main_w-overlay_w)/2:(main_h-overlay_h)/2'
            }
            
            overlay_pos = position_map.get(position, 'main_w-overlay_w-10:10')
            
            cmd = [
                self.ffmpeg_path,
                '-i', main_video_path,
                '-i', overlay_video_path
            ]
            
            # 濾鏡：縮放小影片並疊加
            filter_complex = f'[1:v]scale=iw*{scale}:ih*{scale}[overlay];[0:v][overlay]overlay={overlay_pos}'
            cmd.extend(['-filter_complex', filter_complex])
            
            # 音訊使用主影片的音訊
            cmd.extend(['-c:a', 'copy'])
            
            # 品質設定
            if quality == 'high':
                cmd.extend(['-c:v', 'libx264', '-crf', '18'])
            elif quality == 'medium':
                cmd.extend(['-c:v', 'libx264', '-crf', '23'])
            elif quality == 'low':
                cmd.extend(['-c:v', 'libx264', '-crf', '28'])
            
            cmd.extend(['-shortest'])
            cmd.extend(['-y', output_path])
            
            print(f"\n開始建立子母畫面...")
            print(f"主影片: {Path(main_video_path).name}")
            print(f"疊加影片: {Path(overlay_video_path).name}")
            print(f"位置: {position}, 縮放: {scale}")
            print(f"輸出檔案: {output_path}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 子母畫面建立完成！")
                return True
            else:
                print("❌ 建立失敗:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 發生錯誤: {str(e)}")
            return False
    
    def _determine_target_resolution(self, video_infos, resize_mode, custom_resolution):
        """決定目標解析度"""
        if resize_mode == 'custom' and custom_resolution:
            return custom_resolution
        
        resolutions = []
        for info in video_infos:
            res = info.get('resolution')
            if res and 'x' in res:
                try:
                    w, h = map(int, res.split('x'))
                    resolutions.append((w, h, w*h))  # width, height, pixels
                except:
                    continue
        
        if not resolutions:
            return None
        
        if resize_mode == 'first':
            return f"{resolutions[0][0]}x{resolutions[0][1]}"
        elif resize_mode == 'largest':
            largest = max(resolutions, key=lambda x: x[2])
            return f"{largest[0]}x{largest[1]}"
        elif resize_mode == 'smallest':
            smallest = min(resolutions, key=lambda x: x[2])
            return f"{smallest[0]}x{smallest[1]}"
        
        return None
    
    def batch_concatenate(self, video_dir, output_dir, pattern='*.mp4'):
        """批次合併目錄中的影片"""
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 找到所有符合條件的影片檔案
        video_files = sorted(list(video_dir.glob(pattern)))
        
        if len(video_files) < 2:
            print(f"❌ 在 {video_dir} 中找到的影片檔案少於2個")
            return False
        
        print(f"找到 {len(video_files)} 個影片檔案:")
        for i, video_file in enumerate(video_files, 1):
            print(f"  {i}. {video_file.name}")
        
        output_file = output_dir / f"merged_{video_dir.name}.mp4"
        
        return self.concatenate_videos(
            [str(f) for f in video_files], 
            str(output_file)
        )


def main():
    parser = argparse.ArgumentParser(description='影片合併工具')
    parser.add_argument('videos', nargs='+', help='影片檔案路徑（2個或更多）')
    parser.add_argument('-o', '--output', required=True, help='輸出檔案路徑')
    parser.add_argument('-m', '--mode', choices=['concat', 'side-by-side', 'pip'], 
                       default='concat', help='合併模式')
    parser.add_argument('-q', '--quality', choices=['high', 'medium', 'low', 'copy'], 
                       default='medium', help='影片品質')
    parser.add_argument('--layout', choices=['horizontal', 'vertical'], 
                       default='horizontal', help='並排佈局（僅side-by-side模式）')
    parser.add_argument('--position', choices=['top-left', 'top-right', 'bottom-left', 
                       'bottom-right', 'center'], default='top-right', 
                       help='子母畫面位置（僅pip模式）')
    parser.add_argument('--scale', type=float, default=0.25, 
                       help='子母畫面縮放比例（僅pip模式）')
    parser.add_argument('--resize', choices=['none', 'first', 'largest', 'smallest', 'custom'], 
                       default='none', help='解析度調整模式')
    parser.add_argument('--resolution', help='自訂解析度（如1920x1080）')
    
    args = parser.parse_args()
    
    merger = VideoMerger()
    
    if args.mode == 'concat':
        # 連接模式
        merger.concatenate_videos(args.videos, args.output, args.quality, 
                                args.resize, args.resolution)
    elif args.mode == 'side-by-side':
        # 並排模式
        if len(args.videos) != 2:
            print("❌ 並排模式需要恰好2個影片檔案")
            sys.exit(1)
        merger.side_by_side_videos(args.videos[0], args.videos[1], args.output, 
                                 args.quality, args.layout)
    elif args.mode == 'pip':
        # 子母畫面模式
        if len(args.videos) != 2:
            print("❌ 子母畫面模式需要恰好2個影片檔案")
            sys.exit(1)
        merger.picture_in_picture(args.videos[0], args.videos[1], args.output,
                                args.quality, args.position, args.scale)


if __name__ == '__main__':
    # 如果沒有命令列參數，進入互動模式
    if len(sys.argv) == 1:
        print("=== 影片合併工具 ===\n")
        
        merger = VideoMerger()
        
        try:
            print("請選擇合併模式:")
            print("1. 連接合併 (影片依序播放)")
            print("2. 並排顯示 (兩個影片同時播放)")
            print("3. 子母畫面 (一個影片疊加在另一個上)")
            print("4. 批次合併 (合併資料夾中的所有影片)")
            
            mode_choice = input("請選擇 (1-4): ").strip()
            
            if mode_choice == '1':
                # 連接合併
                print("\n請輸入要合併的影片檔案路徑 (每行一個，完成後輸入空行):")
                video_paths = []
                while True:
                    path = input(f"影片 {len(video_paths)+1}: ").strip('"')
                    if not path:
                        break
                    video_paths.append(path)
                
                if len(video_paths) < 2:
                    print("❌ 至少需要2個影片檔案")
                    sys.exit(1)
                
                output_path = input("輸出檔案路徑: ").strip('"')
                
                # 選擇品質
                print("\n影片品質:")
                print("1. 高品質 (檔案較大)")
                print("2. 中等品質 (推薦)")
                print("3. 低品質 (檔案較小)")
                print("4. 複製模式 (最快，但需要相同格式)")
                
                quality_choice = input("請選擇 (1-4，預設為2): ").strip() or "2"
                quality_map = {'1': 'high', '2': 'medium', '3': 'low', '4': 'copy'}
                quality = quality_map.get(quality_choice, 'medium')
                
                merger.concatenate_videos(video_paths, output_path, quality)
                
            elif mode_choice == '2':
                # 並排顯示
                video1 = input("第一個影片路徑: ").strip('"')
                video2 = input("第二個影片路徑: ").strip('"')
                output_path = input("輸出檔案路徑: ").strip('"')
                
                print("\n佈局選擇:")
                print("1. 水平排列 (左右)")
                print("2. 垂直排列 (上下)")
                
                layout_choice = input("請選擇 (1-2，預設為1): ").strip() or "1"
                layout = 'horizontal' if layout_choice == '1' else 'vertical'
                
                merger.side_by_side_videos(video1, video2, output_path, 'medium', layout)
                
            elif mode_choice == '3':
                # 子母畫面
                main_video = input("主影片路徑: ").strip('"')
                overlay_video = input("疊加影片路徑: ").strip('"')
                output_path = input("輸出檔案路徑: ").strip('"')
                
                print("\n疊加位置:")
                print("1. 右上角")
                print("2. 左上角")
                print("3. 右下角")
                print("4. 左下角")
                print("5. 中央")
                
                pos_choice = input("請選擇 (1-5，預設為1): ").strip() or "1"
                pos_map = {'1': 'top-right', '2': 'top-left', '3': 'bottom-right', 
                          '4': 'bottom-left', '5': 'center'}
                position = pos_map.get(pos_choice, 'top-right')
                
                scale_input = input("小影片縮放比例 (0.1-1.0，預設0.25): ").strip() or "0.25"
                try:
                    scale = float(scale_input)
                    scale = max(0.1, min(1.0, scale))
                except:
                    scale = 0.25
                
                merger.picture_in_picture(main_video, overlay_video, output_path,
                                        'medium', position, scale)
                
            elif mode_choice == '4':
                # 批次合併
                video_dir = input("影片資料夾路徑: ").strip('"')
                output_dir = input("輸出資料夾路徑: ").strip('"')
                pattern = input("檔案類型 (如 *.mp4，預設 *.mp4): ").strip() or "*.mp4"
                
                merger.batch_concatenate(video_dir, output_dir, pattern)
            
            else:
                print("❌ 無效的選擇")
                
        except KeyboardInterrupt:
            print("\n\n程式已取消")
        except Exception as e:
            print(f"\n發生錯誤: {str(e)}")
    else:
        main()