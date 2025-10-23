#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT Multi-Speaker Dialogue Splitter Tool
Splits subtitles with multiple speaker dialogues in the same time frame into separate entries.
Fully compatible with speaking_ground_truth_tool.py
"""

import os
import re
import argparse
from pathlib import Path
from datetime import timedelta


class SRTDialogueSplitter:
    """SRT Dialogue Splitter"""

    def __init__(self):
        # Common dialogue separators
        self.dialogue_patterns = [
            r'^-\s*(.+)$',          # - I chose piccolo.
            r'^\s*-\s*(.+)$',       # - with leading space
            r'^([A-Z\s]+):\s*(.+)$',  # NAME: dialogue
            r'^【([^】]+)】\s*(.+)$',  # 【Character Name】dialogue
            r'^《([^》]+)》\s*(.+)$',  # 《Character Name》dialogue
        ]

    def parse_time(self, time_str):
        """
        Parse SRT time format
        '00:00:32,478' -> timedelta
        """
        # Remove potential BOM
        time_str = time_str.strip().lstrip('\ufeff')

        # SRT format: HH:MM:SS,mmm or HH:MM:SS.mmm
        match = re.match(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})', time_str)
        if match:
            hours, minutes, seconds, milliseconds = map(int, match.groups())
            return timedelta(
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                milliseconds=milliseconds
            )
        raise ValueError(f"Invalid time format: {time_str}")

    def format_time(self, td):
        """
        Convert timedelta to SRT time format
        timedelta -> '00:00:32,478'
        """
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = td.microseconds // 1000

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def detect_multi_dialogue(self, text):
        """
        Detect if the text contains multiple dialogues (only detects dialogues starting with a hyphen).

        Returns:
            list: A list of split dialogues, each element as (character, dialogue).
                  Returns None if there are not multiple hyphen-prefixed dialogues.
        """
        lines = text.strip().split('\n')
        dialogues = []

        # Only detect lines starting with a hyphen
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Only process dialogues starting with a hyphen
            if line.startswith('-'):
                dialogue_text = line[1:].strip()
                dialogues.append((None, dialogue_text))

        # If there are 0 or 1 hyphen-prefixed dialogues, return None (do not split)
        if len(dialogues) <= 1:
            return None

        return dialogues

    def split_time_by_text_length(self, start_time, end_time, dialogues, gap_seconds=0.15):
        """
        Distribute time for multiple dialogues based on text length and a gap.

        Args:
            start_time: Start time (timedelta)
            end_time: End time (timedelta)
            dialogues: List of dialogues [(character, text), ...]
            gap_seconds: Gap in seconds between dialogues (default 0.15s)

        Returns:
            list: [(start1, end1), (start2, end2), ...]
        """
        total_duration = end_time - start_time
        num_dialogues = len(dialogues)

        # Calculate total gap time
        total_gap_time = timedelta(seconds=gap_seconds * (num_dialogues - 1))

        # Actual time available for dialogues
        available_time = total_duration - total_gap_time

        # If the total gap time is too long, reduce the gap
        if available_time.total_seconds() <= 0:
            gap_seconds = 0.05  # Reduce to a minimum gap
            total_gap_time = timedelta(seconds=gap_seconds * (num_dialogues - 1))
            available_time = total_duration - total_gap_time

        # Calculate the length of each dialogue
        text_lengths = []
        for char, text in dialogues:
            # Calculate effective characters (removing spaces and common punctuation)
            effective_length = len(re.sub(r'[\s\?!,\.]', '', text))
            text_lengths.append(max(effective_length, 1))  # At least 1 to avoid division by zero

        total_chars = sum(text_lengths)
        if total_chars == 0: # Prevent division by zero if all texts are empty/punctuation
            total_chars = len(text_lengths)


        # Allocate time proportionally based on text length
        time_ranges = []
        current_time = start_time

        for i, text_len in enumerate(text_lengths):
            # Calculate the time ratio this sentence should occupy
            time_ratio = text_len / total_chars
            dialogue_duration = available_time * time_ratio

            # Set start and end times
            dialogue_start = current_time
            dialogue_end = current_time + dialogue_duration

            time_ranges.append((dialogue_start, dialogue_end))

            # Start time for the next sentence = current end time + gap
            if i < num_dialogues - 1:  # Not the last sentence
                current_time = dialogue_end + timedelta(seconds=gap_seconds)
        
        return time_ranges

    def parse_srt(self, srt_path):
        """
        Parse an SRT file.

        Returns:
            list: A list of subtitle entries.
        """
        try:
            with open(srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"❌ Error: File not found at {srt_path}")
            return []
        
        # Split into subtitle blocks
        subtitle_blocks = re.split(r'\n\s*\n', content.strip())

        subtitles = []
        for block in subtitle_blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue

            try:
                # First line: index
                if not lines[0].strip().lstrip('\ufeff').isdigit():
                    continue
                index = int(lines[0].strip().lstrip('\ufeff'))

                # Second line: timestamp
                time_line = lines[1].strip()
                if '-->' not in time_line:
                    continue
                time_parts = time_line.split(' --> ')
                if len(time_parts) != 2:
                    continue

                start_time = self.parse_time(time_parts[0])
                end_time = self.parse_time(time_parts[1])

                # Remaining lines: text
                text = '\n'.join(lines[2:])

                subtitles.append({
                    'index': index,
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })

            except (ValueError, IndexError) as e:
                print(f"⚠️ Skipping invalid subtitle block: {e}\nBlock content: '''{block}'''")
                continue

        return subtitles

    def split_subtitles(self, subtitles):
        """
        Split subtitles containing multiple dialogues.

        Returns:
            list: A list of split subtitles.
        """
        split_subtitles_list = []
        current_index = 1

        for subtitle in subtitles:
            # Detect if it contains multiple dialogues
            dialogues = self.detect_multi_dialogue(subtitle['text'])

            if dialogues is None:
                # Single dialogue or non-hyphen dialogue, keep as is
                split_subtitles_list.append({
                    'index': current_index,
                    'start': subtitle['start'],
                    'end': subtitle['end'],
                    'text': subtitle['text'],
                    'original_index': subtitle['index']
                })
                current_index += 1
            else:
                # Multiple dialogues found, perform splitting
                time_ranges = self.split_time_by_text_length(
                    subtitle['start'],
                    subtitle['end'],
                    dialogues
                )

                for i, ((char, dialogue), (start, end)) in enumerate(zip(dialogues, time_ranges)):
                    # Construct the new text (prepending the hyphen back)
                    new_text = f"- {dialogue}"

                    split_subtitles_list.append({
                        'index': current_index,
                        'start': start,
                        'end': end,
                        'text': new_text,
                        'original_index': subtitle['index'],
                        'split_part': f"{i+1}/{len(dialogues)}"
                    })
                    current_index += 1

        return split_subtitles_list

    def save_srt(self, subtitles, output_path):
        """
        Save as SRT format.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for subtitle in subtitles:
                # Index
                f.write(f"{subtitle['index']}\n")

                # Timestamp
                start_str = self.format_time(subtitle['start'])
                end_str = self.format_time(subtitle['end'])
                f.write(f"{start_str} --> {end_str}\n")

                # Text
                f.write(f"{subtitle['text']}\n")

                # Blank line separator
                f.write("\n")

        print(f"✅ Saved to: {output_path}")

    def save_report(self, original_subtitles, split_subtitles_list, output_dir):
        """
        Save the processing report.
        """
        report_path = output_dir / 's2_ep24_split_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        split_count = 0
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("SRT Multi-Speaker Dialogue Split Report\n")
            f.write("="*70 + "\n\n")

            f.write(f"Original number of subtitle entries: {len(original_subtitles)}\n")
            f.write(f"Number of entries after splitting: {len(split_subtitles_list)}\n")
            f.write(f"Number of new entries added: {len(split_subtitles_list) - len(original_subtitles)}\n\n")

            f.write("-"*70 + "\n")
            f.write("Details of Splitting\n")
            f.write("-"*70 + "\n\n")

            # Find the subtitles that were split
            split_groups = {}
            for subtitle in split_subtitles_list:
                orig_idx = subtitle.get('original_index')
                if orig_idx:
                    if orig_idx not in split_groups:
                        split_groups[orig_idx] = []
                    split_groups[orig_idx].append(subtitle)

            for orig_idx in sorted(split_groups.keys()):
                group = split_groups[orig_idx]
                if len(group) > 1:
                    split_count += 1
                    # Find the original subtitle
                    orig = next((s for s in original_subtitles if s['index'] == orig_idx), None)
                    if orig:
                        original_text_oneline = orig['text'].replace('\n', ' / ')
                        f.write(f"Original #{orig_idx}:\n")
                        f.write(f"  Time: {self.format_time(orig['start'])} --> {self.format_time(orig['end'])}\n")
                        f.write(f"  Text: {original_text_oneline}\n\n")

                        f.write(f"  Split into {len(group)} entries:\n")
                        for sub in group:
                            f.write(f"    #{sub['index']}: {self.format_time(sub['start'])} --> {self.format_time(sub['end'])}\n")
                            f.write(f"          {sub['text']}\n")
                        f.write("-" * 50 + "\n\n")
            
            if split_count == 0:
                 f.write("No subtitle entries were split.\n")

        print(f"✅ Report saved to: {report_path}")

    def process_srt(self, input_path, output_path=None, save_report=True):
        """
        Process an SRT file.

        Args:
            input_path: Path to the input SRT file.
            output_path: Path to the output SRT file (optional).
            save_report: Whether to save a processing report.
        """
        input_path = Path(input_path)
        if not input_path.is_file():
            print(f"❌ Error: Input file not found at '{input_path}'")
            return

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_split.srt"
        else:
            output_path = Path(output_path)
            
        print(f"📖 Reading SRT: {input_path}")

        # Parse original subtitles
        original_subtitles = self.parse_srt(input_path)
        if not original_subtitles:
            print("No valid subtitles found to process.")
            return
            
        print(f"  Found original entries: {len(original_subtitles)}")

        # Split dialogues
        print(f"🔄 Splitting multi-speaker dialogues...")
        split_subtitles_list = self.split_subtitles(original_subtitles)
        print(f"  Entries after split: {len(split_subtitles_list)}")
        
        added_entries = len(split_subtitles_list) - len(original_subtitles)
        print(f"  New entries added: {added_entries}")

        # Save results
        self.save_srt(split_subtitles_list, output_path)

        # Save report
        if save_report:
            if added_entries > 0:
                self.save_report(original_subtitles, split_subtitles_list, output_path.parent)
            else:
                print("ℹ️ No dialogues were split, so no detailed report was generated.")

        return split_subtitles_list


def main():
    parser = argparse.ArgumentParser(
        description='SRT Multi-Speaker Dialogue Splitter - Splits dialogues from multiple speakers in the same time frame into separate entries.'
    )
    parser.add_argument('--input', help='Input SRT file path')
    parser.add_argument('-o', '--output', help='Output SRT file path (optional)')
    parser.add_argument('--no-report', action='store_true', help='Do not save the processing report')

    args = parser.parse_args()

    splitter = SRTDialogueSplitter()
    splitter.process_srt(args.input, args.output, not args.no_report)


if __name__ == '__main__':
    import sys

    # If no command-line arguments are provided, enter interactive mode
    if len(sys.argv) == 1:
        print("="*70)
        print("SRT Multi-Speaker Dialogue Splitter Tool")
        print("="*70)
        print("\nThis tool splits subtitle entries that contain multiple hyphen-prefixed lines.")
        print("For example:")
        print("  Original: - I chose piccolo.")
        print("            - Piccolo?! But I wanted leatherwork!")
        print("  After splitting: Two separate subtitle entries, each with its own time duration\n")

        try:
            input_path_str = input("Please enter the path to the SRT file: ").strip().strip('"')
            if not input_path_str:
                print("\n❌ Error: Input path cannot be empty.")
            else:
                output_path_str = input("Output path (press Enter for default): ").strip().strip('"') or None
                
                splitter = SRTDialogueSplitter()
                splitter.process_srt(input_path_str, output_path_str, save_report=True)
                
                print("\n✨ Processing complete!")

        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled.")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
    else:
        main()